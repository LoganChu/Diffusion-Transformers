"""Online model-based RL training loop.

Phases
------
1. Seed the replay buffer from offline HDF5 (heuristic rewards via env/reward.py).
2. Warm-start the model from a pretrained offline checkpoint (recommended).
3. Main loop:
   a. Collect one episode with the CEM planner in the real ManiSkill env.
   b. Push transitions — labelled with env.step() rewards — into the replay buffer.
   c. Run K gradient updates on sampled replay batches (world model + task heads).
   d. Periodically evaluate success rate and save checkpoints.

Usage
-----
    # Warm-start from offline checkpoint, run 500 online episodes
    python -m training.train_online \
        --hdf5        trajectories_10k_v2.h5 \
        --resume      checkpoints/best.pt \
        --n_episodes  500

    # From scratch (slower convergence)
    python -m training.train_online --n_episodes 1000
"""

from __future__ import annotations

import argparse
import collections
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mani_skill.envs   # noqa: F401 — register ManiSkill envs
import gymnasium as gym
from torch.profiler import record_function

from data.replay       import ReplayBuffer
from inference.planner import (
    cem_plan, cube_height_score_fn, maniskill_reward_score_fn,
    reward_only_score_fn, reward_value_score_fn,
)
from models.dit        import ACTION_DIM, DiTSmall, IN_CHANNELS, LATENT_H, LATENT_W, NUM_PATCHES
from training.loss     import WorldModelLoss


# ---------------------------------------------------------------------------
# Cosmos encoder
# ---------------------------------------------------------------------------

def load_encoder(ckpt_dir: str, device: torch.device):
    """Load the Cosmos-Tokenizer-CI16x16 encoder JIT model."""
    from data.ingest import ensure_cosmos_weights, CosmosLatentEncoder
    ensure_cosmos_weights(ckpt_dir)
    return CosmosLatentEncoder(ckpt_dir)


@torch.no_grad()
def encode_frame(encoder, frame_hwc: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Encode a single [H, W, 3] uint8 frame → [1, 16, 8, 8] float16 latent.

    Handles the [1, H, W, 3] batched output that ManiSkill returns for
    num_envs=1 by squeezing the leading dimension before encoding.
    """
    if isinstance(frame_hwc, np.ndarray):
        frame_hwc = torch.from_numpy(frame_hwc)
    if frame_hwc.dim() == 4:
        frame_hwc = frame_hwc[0]                       # [H, W, 3]
    frame = frame_hwc[..., :3].permute(2, 0, 1).float().div_(255.0)  # [3, H, W]
    if frame.shape[-2:] != (128, 128):
        frame = F.interpolate(
            frame.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze(0)
    buf = frame.unsqueeze(0).to("cuda", dtype=torch.float16)  # [1, 3, 128, 128]
    latent = encoder.encode(buf)                        # [1, 16, 8, 8]
    encoder.sync()
    return latent.to(torch.float16)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(control_mode: str = "pd_ee_delta_pos", max_episode_steps: int = 200) -> gym.Env:
    return gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
    )


def parse_success(info: dict) -> bool:
    """Extract scalar success flag from ManiSkill info dict."""
    s = info.get("success", False)
    if isinstance(s, torch.Tensor):
        return bool(s.any().item())
    return bool(s)


def parse_reward(reward) -> float:
    if isinstance(reward, torch.Tensor):
        return float(reward[0].item())
    if hasattr(reward, "__len__"):
        return float(reward[0])
    return float(reward)


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _build_score_fn(args):
    """Return the CEM score function selected by --score_fn."""
    if args.score_fn == "reward_value":
        return lambda model, z, t, **kw: reward_value_score_fn(
            model, z, t, horizon=args.horizon, gamma=args.gamma
        )
    if args.score_fn == "reward_only":
        return reward_only_score_fn
    if args.score_fn == "maniskill_reward":
        return maniskill_reward_score_fn
    return cube_height_score_fn


def collect_episode(
    env,
    model:   nn.Module,
    encoder,
    replay:  ReplayBuffer,
    args,
    device:  torch.device,
    score_fn=None,
) -> dict:
    """Run one episode with the CEM planner and push transitions to replay.

    Exploration noise is added on top of the planned action so the agent
    visits states outside the planner's current knowledge.

    Returns a stats dict with keys: steps, total_reward, success.
    """
    model.eval()
    obs, _info = env.reset()

    ctx_deque: collections.deque[torch.Tensor] = collections.deque(maxlen=args.n_ctx)

    ep_latents:    list[torch.Tensor] = []   # [16, 8, 8] float16
    ep_a_conds:    list[torch.Tensor] = []   # [4] float32
    ep_rewards:    list[float]        = []
    ep_terminated: list[bool]         = []
    ep_truncated:  list[bool]         = []
    ep_success_t:  list[bool]         = []   # per-step success flag
    ep_success                        = False

    for _ in range(args.max_steps):
        # ---- Encode current observation ----
        frame   = env.render()                         # [1, H, W, 3] uint8
        z_t     = encode_frame(encoder, frame)         # [1, 16, 8, 8] float16
        z_t_dev = z_t.to(device)

        ctx_deque.append(z_t_dev.squeeze(0))           # [16, 8, 8]

        # ---- Build [1, n_ctx, 16, 8, 8] context with zero-padding ----
        frames  = list(ctx_deque)
        n_have  = len(frames)
        if n_have < args.n_ctx:
            pad    = [torch.zeros_like(frames[0])] * (args.n_ctx - n_have)
            frames = pad + frames
        ctx_input = torch.stack(frames, dim=0).unsqueeze(0)    # [1, n_ctx, 16, 8, 8]
        ctx_act   = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float16)

        # ---- Extract ee_pos for maniskill_reward score function ----
        try:
            ee_pos = env.unwrapped.agent.tcp.pose.p[0].to(device).float()   # [3]
        except Exception:
            ee_pos = None

        # ---- Plan ----
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            a_4d = cem_plan(
                model,
                ctx_input,
                ctx_act,
                score_fn        = score_fn,
                horizon         = args.horizon,
                n_candidates    = args.n_candidates,
                n_elites        = args.n_elites,
                n_cem_iters     = args.n_cem_iters,
                num_ode_steps   = args.num_ode_steps,
                noise_std_init  = args.noise_std_init,
                ee_pos          = ee_pos,
            )   # [4] float32

        # ---- Exploration noise ----
        if args.expl_noise > 0:
            noise = torch.randn_like(a_4d) * args.expl_noise
            a_4d  = (a_4d + noise).clamp(-0.08, 0.08)

        # ---- Step environment ----
        action_np = a_4d.cpu().numpy().reshape(1, -1)      # [1, 4] for num_envs=1
        obs, reward, terminated, truncated, info = env.step(action_np)
        step_success = parse_success(info)
        done = bool(terminated or truncated)
        r    = parse_reward(reward)
        if step_success:
            ep_success = True

        ep_latents.append(z_t.squeeze(0).cpu())
        ep_a_conds.append(a_4d.cpu())                  # [4] — pure action, no ee_pos
        ep_rewards.append(r)
        ep_terminated.append(bool(terminated))
        ep_truncated.append(bool(truncated))
        ep_success_t.append(step_success)

        if done:
            break

    # ---- Push episode to replay ----
    if len(ep_latents) > 1:
        replay.push_episode(
            latents    = torch.stack(ep_latents).to(torch.float16),
            a_conds    = torch.stack(ep_a_conds),
            rewards    = torch.tensor(ep_rewards,    dtype=torch.float32),
            terminated = torch.tensor(ep_terminated, dtype=torch.bool),
            truncated  = torch.tensor(ep_truncated,  dtype=torch.bool),
            success    = torch.tensor(ep_success_t,  dtype=torch.bool),
        )

    model.train()
    return {
        "steps":        len(ep_latents),
        "total_reward": sum(ep_rewards),
        "success":      ep_success,
    }


# ---------------------------------------------------------------------------
# Evaluation (no exploration noise, no gradient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    env,
    model:   nn.Module,
    encoder,
    args,
    device:  torch.device,
    score_fn=None,
) -> dict:
    """Run args.eval_episodes greedy episodes and return success rate + mean reward."""
    model.eval()
    successes    = 0
    total_reward = 0.0

    for _ in range(args.eval_episodes):
        obs, _info = env.reset()
        ctx_deque: collections.deque[torch.Tensor] = collections.deque(maxlen=args.n_ctx)
        ep_reward  = 0.0

        for _ in range(args.max_steps):
            frame   = env.render()
            z_t     = encode_frame(encoder, frame).to(device)
            ctx_deque.append(z_t.squeeze(0))

            frames = list(ctx_deque)
            n_have = len(frames)
            if n_have < args.n_ctx:
                pad    = [torch.zeros_like(frames[0])] * (args.n_ctx - n_have)
                frames = pad + frames
            ctx_input = torch.stack(frames, dim=0).unsqueeze(0)
            ctx_act   = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float16)

            try:
                ee_pos = env.unwrapped.agent.tcp.pose.p[0].to(device).float()
            except Exception:
                ee_pos = None

            with torch.amp.autocast("cuda", dtype=torch.float16):
                a_4d = cem_plan(
                    model, ctx_input, ctx_act,
                    score_fn        = score_fn,
                    horizon         = args.horizon,
                    n_candidates    = args.n_candidates,
                    n_elites        = args.n_elites,
                    n_cem_iters     = args.n_cem_iters,
                    num_ode_steps   = args.num_ode_steps,
                    noise_std_init  = args.noise_std_init,
                    ee_pos          = ee_pos,
                )

            obs, reward, terminated, truncated, info = env.step(
                a_4d.cpu().numpy().reshape(1, -1)
            )
            ep_reward += parse_reward(reward)

            if parse_success(info):
                successes += 1
                break
            if bool(terminated or truncated):
                break

        total_reward += ep_reward

    model.train()
    n = args.eval_episodes
    return {
        "success_rate": successes / n,
        "mean_reward":  total_reward / n,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, episode, path,
                    replay=None, best_success_rate=0.0):
    torch.save(
        {
            "model":              model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "scheduler":          scheduler.state_dict(),
            "scaler":             scaler.state_dict(),
            "episode":            episode,
            "best_success_rate":  best_success_rate,
        },
        path,
    )
    if replay is not None:
        replay.save(path + ".replay")
    print(f"  >> saved {path}")


# ---------------------------------------------------------------------------
# Offline head pretraining
# ---------------------------------------------------------------------------

def pretrain_heads(
    model:     nn.Module,
    replay:    ReplayBuffer,
    args,
    device:    torch.device,
    amp_dtype: torch.dtype,
) -> None:
    """Pre-train reward / done / value heads on offline replay data.

    The backbone is frozen so offline-learned flow-matching dynamics are not
    disturbed.  Only Pass 2 of WorldModelLoss runs (clean input, task heads)
    because the offline HDF5 has reward / done / success labels.

    Call this once after seeding the replay buffer from HDF5 and loading the
    offline checkpoint — before the online collection loop begins.
    """
    n = args.pretrain_heads_steps
    if n <= 0 or len(replay) < args.batch_size:
        return

    print(f"\nPre-training task heads for {n} steps on offline data "
          f"(backbone frozen) ...")

    # Freeze everything except the three task heads
    head_ids = set()
    for head in (model.reward_head, model.done_head, model.value_head):
        head_ids.update(id(p) for p in head.parameters())
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)

    head_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = args.lr,
        betas        = (0.9, 0.95),
        weight_decay = args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    log_every = max(1, n // 5)   # print 5 progress lines
    for step in range(n):
        batch   = replay.sample(args.batch_size, device=device)
        z_next  = batch["z_next"]
        ctx     = batch["ctx"]
        a_cond  = batch["a_cond"]
        r_gt    = batch["r"]
        done_gt = (batch["terminated"] | batch["truncated"]).float()
        mc_ret  = batch["mc_return"]
        trunc   = batch["truncated"].float()
        B       = z_next.shape[0]

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            t_ones = torch.ones(B, device=device, dtype=z_next.dtype)
            _, r_pred, d_pred, v_pred = model(
                z_next, t_ones, a_cond, ctx_latents=ctx, return_heads=True
            )
            loss_r = F.mse_loss(r_pred.squeeze(-1), r_gt)
            loss_d = F.binary_cross_entropy_with_logits(d_pred.squeeze(-1), done_gt)
            # Bootstrap truncated transitions exactly as WorldModelLoss does
            v_target = mc_ret + trunc * WorldModelLoss.GAMMA * v_pred.squeeze(-1).detach()
            loss_v   = F.mse_loss(v_pred.squeeze(-1), v_target)
            loss     = loss_r + WorldModelLoss.DONE_WEIGHT * loss_d + WorldModelLoss.VALUE_WEIGHT * loss_v

        head_opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(head_opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(head_opt)
        scaler.update()

        if (step + 1) % log_every == 0:
            print(f"  [{step+1:>5d}/{n}]  "
                  f"r={loss_r.item():.4f}  "
                  f"done={loss_d.item():.4f}  "
                  f"val={loss_v.item():.4f}")

    # Restore all params to trainable for the online phase
    for p in model.parameters():
        p.requires_grad_(True)

    print("Head pretraining complete.\n")


# ---------------------------------------------------------------------------
# Main online training loop
# ---------------------------------------------------------------------------

def train_online(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- WandB (optional) ----
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project  = args.wandb_project,
            name     = args.wandb_run_name,
            config   = vars(args),
        )

    # ---- Model ----
    model = DiTSmall().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer / scheduler ----
    # Differential learning rates: backbone gets a small lr to avoid overwriting
    # offline-learned flow-matching dynamics; task heads get the full lr so they
    # can learn quickly from online env rewards.
    _head_params = set()
    for head in (model.reward_head, model.done_head, model.value_head):
        _head_params.update(id(p) for p in head.parameters())
    backbone_params = [p for p in model.parameters() if id(p) not in _head_params]
    head_params     = [p for p in model.parameters() if id(p) in _head_params]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
            {"params": head_params,     "lr": args.lr},
        ],
        betas        = (0.9, 0.95),
        weight_decay = args.weight_decay,
    )
    total_updates = args.n_episodes * args.updates_per_ep
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=args.lr * 0.1
    )
    print(f"Backbone lr: {args.lr * args.backbone_lr_scale:.2e}  "
          f"Head lr: {args.lr:.2e}")

    use_bf16  = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    # ---- Loss ----
    loss_fn = WorldModelLoss()

    # ---- Replay buffer ----
    replay = ReplayBuffer(
        capacity   = args.replay_capacity,
        n_ctx      = args.n_ctx,
        action_dim = ACTION_DIM,   # 4: [dx, dy, dz, gripper]
        gamma      = args.gamma,
    )

    # ---- Resume or cold start ----
    start_episode     = 0
    best_success_rate = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict)
        is_online_ckpt = "episode" in ckpt   # False for offline checkpoints
        if is_online_ckpt:
            # Full resume: restore optimizer, scheduler, scaler from a previous online run.
            # Guard against param-group count mismatch (e.g. old single-group checkpoints).
            saved_opt = ckpt["optimizer"]
            n_saved   = len(saved_opt.get("param_groups", []))
            n_current = len(optimizer.param_groups)
            if n_saved == n_current:
                optimizer.load_state_dict(saved_opt)
                scheduler.load_state_dict(ckpt["scheduler"])
                scaler.load_state_dict(ckpt["scaler"])
            else:
                print(f"  WARNING: optimizer param groups mismatch "
                      f"(saved={n_saved}, current={n_current}). "
                      f"Skipping optimizer/scheduler restore — fresh optimizer used.")
        # For offline warm-starts, keep fresh optimizer + scheduler (lr starts at args.lr)
        start_episode     = ckpt.get("episode", -1) + 1   # 0 when loading offline ckpt
        best_success_rate = ckpt.get("best_success_rate", 0.0)
        # Load replay buffer: for online resumes use the saved replay so online
        # experience is preserved.  For offline warm-starts seed from HDF5.
        replay_path = args.resume + ".replay"
        if is_online_ckpt and os.path.isfile(replay_path):
            replay.load(replay_path)
        elif args.hdf5:
            replay.seed_from_hdf5(args.hdf5)
        print(f"Resumed from {args.resume} (episode {start_episode})")
    elif args.hdf5:
        replay.seed_from_hdf5(args.hdf5)

    # ---- Encoder + environment ----
    encoder = load_encoder(args.cosmos_ckpt, device)
    env     = make_env(max_episode_steps=args.max_steps)

    # ---- Score function ----
    score_fn = _build_score_fn(args)
    print(f"CEM score function: {args.score_fn}")

    # ---- Training loop ----
    global_update = 0

    for episode in range(start_episode, args.n_episodes):
        # -- Collect --
        ep_stats = collect_episode(env, model, encoder, replay, args, device,
                                   score_fn=score_fn)

        # -- Update (skip if replay too small) --
        update_stats: dict[str, float] = {}
        if len(replay) >= args.min_replay_size:
            accum: dict[str, float] = collections.defaultdict(float)
            for _ in range(args.updates_per_ep):
                batch = replay.sample(args.batch_size, device=device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss, loss_dict = loss_fn(model, batch)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_update += 1

                for k, v in loss_dict.items():
                    accum[k] += v / args.updates_per_ep

            update_stats = dict(accum)

        # -- Logging --
        if (episode + 1) % args.log_every == 0:
            lr  = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(
                f"[ep {episode+1:>4d}]  "
                f"steps={ep_stats['steps']:>3d}  "
                f"r={ep_stats['total_reward']:>6.2f}  "
                f"success={ep_stats['success']}  "
                f"replay={len(replay):,}  "
                + (
                    f"loss={update_stats.get('loss', 0):.4f}  "
                    f"cfm={update_stats.get('loss_cfm', 0):.4f}  "
                    f"rew={update_stats.get('loss_reward', 0):.4f}  "
                    f"done={update_stats.get('loss_done', 0):.4f}  "
                    f"val={update_stats.get('loss_value', 0):.4f}  "
                    if update_stats else ""
                )
                + f"lr={lr:.2e}  mem={mem:.0f}MB"
            )
            if run is not None:
                log = {
                    "collect/steps":        ep_stats["steps"],
                    "collect/total_reward": ep_stats["total_reward"],
                    "collect/success":      int(ep_stats["success"]),
                    "replay/size":          len(replay),
                    "train/lr":             lr,
                    "gpu/mem_peak_mb":      mem,
                }
                log.update({f"train/{k}": v for k, v in update_stats.items()})
                run.log(log, step=episode + 1)

        # -- Evaluate + checkpoint --
        if (episode + 1) % args.eval_every == 0 and len(replay) >= args.min_replay_size:
            eval_stats = evaluate(env, model, encoder, args, device,
                                  score_fn=score_fn)
            sr = eval_stats["success_rate"]
            print(
                f"  >> eval  success_rate={sr:.1%}  "
                f"mean_reward={eval_stats['mean_reward']:.2f}"
            )
            if run is not None:
                run.log(
                    {
                        "eval/success_rate": sr,
                        "eval/mean_reward":  eval_stats["mean_reward"],
                    },
                    step=episode + 1,
                )
            if sr > best_success_rate:
                best_success_rate = sr
                save_checkpoint(
                    model, optimizer, scheduler, scaler, episode,
                    os.path.join(args.ckpt_dir, "online_best.pt"),
                    replay=replay, best_success_rate=best_success_rate,
                )
                print(f"  >> new best success_rate={sr:.1%}")

        # Periodic checkpoint
        if (episode + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, episode,
                os.path.join(args.ckpt_dir, f"online_ep{episode+1:05d}.pt"),
                replay=replay, best_success_rate=best_success_rate,
            )

    env.close()
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.n_episodes - 1,
        os.path.join(args.ckpt_dir, "online_final.pt"),
        replay=replay, best_success_rate=best_success_rate,
    )
    if run is not None:
        run.finish()
    print(f"Done. Best success rate: {best_success_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Online MBRL training loop")
    # Data
    p.add_argument("--hdf5",             type=str,   default="trajectories_10k_v2.h5")
    p.add_argument("--cosmos_ckpt",      type=str,   default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16")
    p.add_argument("--resume",           type=str,   default=None)
    # Replay
    p.add_argument("--replay_capacity",  type=int,   default=50_000)
    p.add_argument("--min_replay_size",  type=int,   default=1_000)
    p.add_argument("--n_ctx",            type=int,   default=4)
    p.add_argument("--gamma",            type=float, default=0.99)
    # Collection
    p.add_argument("--n_episodes",       type=int,   default=500)
    p.add_argument("--max_steps",        type=int,   default=200)
    p.add_argument("--expl_noise",       type=float, default=0.02)
    # Planner
    p.add_argument("--horizon",          type=int,   default=6)
    p.add_argument("--n_candidates",     type=int,   default=128)
    p.add_argument("--n_elites",         type=int,   default=16)
    p.add_argument("--n_cem_iters",      type=int,   default=5)
    p.add_argument("--num_ode_steps",    type=int,   default=10)
    p.add_argument("--noise_std_init",   type=float, default=0.05,
                   help="Initial std of the CEM action distribution. "
                        "Increase (e.g. 0.1) if the arm barely moves.")
    p.add_argument("--score_fn",         type=str,   default="cube_pos",
                   choices=["cube_pos", "reward_value", "reward_only", "maniskill_reward"],
                   help="CEM score function. "
                        "maniskill_reward: reach+lift using CubePosHead + ee_pos from env "
                        "(best signal from episode 0, no task head training required). "
                        "reward_value: RewardHead+ValueHead bootstrap (best after online training). "
                        "reward_only: RewardHead only (avoids value overestimation).")
    # Training
    p.add_argument("--updates_per_ep",   type=int,   default=20)
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale",type=float, default=0.1,
                   help="Backbone lr = lr * backbone_lr_scale. Default 0.1 means "
                        "backbone gets 10x smaller lr than task heads to prevent "
                        "catastrophic forgetting of offline-learned dynamics.")
    p.add_argument("--weight_decay",     type=float, default=0.05)
    p.add_argument("--seed",             type=int,   default=42)
    # Evaluation
    p.add_argument("--eval_every",       type=int,   default=20)
    p.add_argument("--eval_episodes",    type=int,   default=10)
    # Logging / checkpoints
    p.add_argument("--log_every",        type=int,   default=5)
    p.add_argument("--save_every",       type=int,   default=10)
    p.add_argument("--ckpt_dir",         type=str,   default="checkpoints")
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--wandb_project",    type=str,   default="optiworld-online")
    p.add_argument("--wandb_run_name",   type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train_online(parse_args())
