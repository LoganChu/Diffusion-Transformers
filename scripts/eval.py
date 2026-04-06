"""Evaluation harness for the OptiWorld-FM MBRL agent.

Loads a model checkpoint, runs N greedy episodes with the CEM planner in
ManiSkill PickCube-v1, and reports success rate, mean return, and mean
episode length.

Usage
-----
    python scripts/eval.py --checkpoint checkpoints/online_best.pt --n_episodes 20

    # Verbose: print per-episode stats
    python scripts/eval.py --checkpoint checkpoints/online_best.pt --verbose
"""

from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
import mani_skill.envs  # noqa: F401
import gymnasium as gym

from inference.planner import cem_plan, cube_height_score_fn
from models.dit        import ACTION_DIM, DiTSmall


# ---------------------------------------------------------------------------
# Shared helpers (mirrors training/train_online.py — kept separate to allow
# running eval without importing the full training stack)
# ---------------------------------------------------------------------------

def _load_encoder(ckpt_dir: str):
    from data.ingest import ensure_cosmos_weights, CosmosLatentEncoder
    ensure_cosmos_weights(ckpt_dir)
    return CosmosLatentEncoder(ckpt_dir)


@torch.no_grad()
def _encode(encoder, frame) -> torch.Tensor:
    """[H,W,3] or [1,H,W,3] uint8 → [1, 16, 8, 8] float16 on encoder device."""
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if frame.dim() == 4:
        frame = frame[0]
    frame = frame[..., :3].permute(2, 0, 1).float().div_(255.0)
    if frame.shape[-2:] != (128, 128):
        frame = F.interpolate(
            frame.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze(0)
    buf    = frame.unsqueeze(0).to(torch.float16)
    latent = encoder.encode(buf)
    encoder.sync()
    return latent.to(torch.float16)


def _parse_success(info: dict) -> bool:
    s = info.get("success", False)
    if isinstance(s, torch.Tensor):
        return bool(s.any().item())
    return bool(s)


def _parse_reward(r) -> float:
    if isinstance(r, torch.Tensor):
        return float(r[0].item())
    if hasattr(r, "__len__"):
        return float(r[0])
    return float(r)


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(env, model, encoder, args, device: torch.device) -> dict:
    """Run one greedy episode. Returns stats dict."""
    model.eval()
    obs, _info = env.reset()

    ctx_deque: collections.deque[torch.Tensor] = collections.deque(maxlen=args.n_ctx)
    total_r   = 0.0
    steps     = 0
    success   = False

    for step in range(args.max_steps):
        frame = env.render()
        z_t   = _encode(encoder, frame).to(device)
        ctx_deque.append(z_t.squeeze(0))

        frames = list(ctx_deque)
        if len(frames) < args.n_ctx:
            pad    = [torch.zeros_like(frames[0])] * (args.n_ctx - len(frames))
            frames = pad + frames
        ctx_input = torch.stack(frames, dim=0).unsqueeze(0)    # [1, n_ctx, 16, 8, 8]
        ctx_act   = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float16)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            a_4d = cem_plan(
                model, ctx_input, ctx_act,
                score_fn      = cube_height_score_fn,
                horizon       = args.horizon,
                n_candidates  = args.n_candidates,
                n_elites      = args.n_elites,
                n_cem_iters   = args.n_cem_iters,
                num_ode_steps = args.num_ode_steps,
            )

        obs, reward, terminated, truncated, info = env.step(
            a_4d.cpu().numpy().reshape(1, -1)
        )
        total_r += _parse_reward(reward)
        steps   += 1

        if _parse_success(info):
            success = True

        if bool(terminated or truncated):
            break

    return {"success": success, "total_reward": total_r, "steps": steps}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    model = DiTSmall().to(device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ep = ckpt.get("episode", "?")
    print(f"Loaded checkpoint from {args.checkpoint}  (episode {ep})")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Encoder ----
    encoder = _load_encoder(args.cosmos_ckpt)

    # ---- Environment ----
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_ee_delta_pos",
    )

    # ---- Run episodes ----
    results = []
    for ep_idx in range(args.n_episodes):
        stats = run_episode(env, model, encoder, args, device)
        results.append(stats)
        if args.verbose:
            print(
                f"  ep {ep_idx+1:>3d}  "
                f"success={stats['success']}  "
                f"reward={stats['total_reward']:>7.2f}  "
                f"steps={stats['steps']}"
            )

    env.close()

    # ---- Summary ----
    n            = len(results)
    success_rate = sum(r["success"]       for r in results) / n
    mean_reward  = sum(r["total_reward"]  for r in results) / n
    mean_steps   = sum(r["steps"]         for r in results) / n

    print(f"\n{'='*50}")
    print(f"Episodes:      {n}")
    print(f"Success rate:  {success_rate:.1%}  ({int(success_rate*n)}/{n})")
    print(f"Mean reward:   {mean_reward:.3f}")
    print(f"Mean steps:    {mean_steps:.1f}")
    print(f"{'='*50}")

    return {"success_rate": success_rate, "mean_reward": mean_reward, "mean_steps": mean_steps}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CEM planner + DiT world model")
    p.add_argument("--checkpoint",    type=str,   required=True)
    p.add_argument("--cosmos_ckpt",   type=str,   default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16")
    p.add_argument("--n_episodes",    type=int,   default=20)
    p.add_argument("--max_steps",     type=int,   default=200)
    # Planner
    p.add_argument("--horizon",       type=int,   default=6)
    p.add_argument("--n_candidates",  type=int,   default=32)
    p.add_argument("--n_elites",      type=int,   default=6)
    p.add_argument("--n_cem_iters",   type=int,   default=3)
    p.add_argument("--num_ode_steps", type=int,   default=10)
    p.add_argument("--n_ctx",         type=int,   default=4)
    p.add_argument("--verbose",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
