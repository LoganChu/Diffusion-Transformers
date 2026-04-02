"""Record a few guided-policy trajectories as MP4s.

Usage:
    conda run -n ml_env python scripts/record_trajectories.py
    conda run -n ml_env python scripts/record_trajectories.py --n-episodes 5 --fps 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

# Pull GuidedPolicy and wrist-fix helper from ingest pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy, ManiSkillCollector

VIDEO_DIR = Path("outputs/videos")
TASK      = "PickCube-v1"
MAX_STEPS = 400   # covers 5-phase budget (80+120+30+100+60=390)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes",           type=int,   default=3)
    p.add_argument("--fps",                  type=int,   default=20)
    p.add_argument("--seed",                 type=int,   default=0)
    p.add_argument("--noise",                type=float, default=None,
                   help="Fixed action noise std. If omitted, noise is sampled per episode "
                        "from a log-normal distribution.")
    p.add_argument("--noise-mu-log",         type=float, default=-2.35,
                   help="Log-space mean for per-episode noise sampling (default: -2.35)")
    p.add_argument("--noise-sigma-log",      type=float, default=0.75,
                   help="Log-space std for per-episode noise sampling (default: 0.75)")
    p.add_argument("--noise-max",            type=float, default=0.22,
                   help="Upper truncation for per-episode noise sampling (default: 0.22)")
    p.add_argument("--robot_init_qpos_noise", type=float, default=0.10,
                   help="Std (rad) of shoulder/elbow joint noise (wrist is fixed straight down)")
    p.add_argument("--cube_spawn_half_size",  type=float, default=0.15,
                   help="Half-side (m) of XY region where cube and goal spawn")
    return p.parse_args()


def sample_noise(rng, mu_log, sigma_log, sigma_max):
    """Sample one noise value from a truncated log-normal distribution."""
    while True:
        s = rng.lognormal(mean=mu_log, sigma=sigma_log)
        if s <= sigma_max:
            return float(s)


def fix_wrist(env):
    """Reset Panda wrist joints to canonical straight-down orientation."""
    uwenv = env.unwrapped
    _WRIST = torch.tensor([0.0, np.pi * 3 / 4, np.pi / 4],
                           dtype=torch.float32, device=uwenv.device)
    qpos = uwenv.agent.robot.get_qpos().clone()
    qpos[:, 4:7] = _WRIST
    uwenv.agent.robot.set_qpos(qpos)


def run_episode(env, policy, seed: int, max_steps: int):
    """Roll out one episode. Returns list of [H,W,3] uint8 frames."""
    obs, _ = env.reset(seed=seed)
    fix_wrist(env)
    policy.phases.zero_()
    policy.phase_steps.zero_()

    frames = []
    for _ in range(max_steps):
        action = policy()
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]   # squeeze num_envs dim
        frames.append(frame)

        if terminated.any() or truncated.any():
            break

    return frames


def main():
    args = parse_args()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating env  task={TASK}  control_mode=pd_ee_delta_pos  render_backend=cpu")
    env = gym.make(
        TASK,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=MAX_STEPS + 10,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
    )
    env.unwrapped.cube_spawn_half_size = args.cube_spawn_half_size

    rng = np.random.default_rng(args.seed)
    fixed_noise = args.noise is not None

    policy = GuidedPolicy(
        env=env,
        num_envs=1,
        noise_scale=args.noise if fixed_noise else 0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if fixed_noise:
        print(f"Noise mode: fixed  sigma={args.noise}")
    else:
        print(f"Noise mode: sampled  LogNormal(mu={args.noise_mu_log}, s={args.noise_sigma_log})  max={args.noise_max}")

    # Find next free episode index
    existing = sorted(VIDEO_DIR.glob("episode_*.mp4"))
    ep_idx   = len(existing)

    for i in range(args.n_episodes):
        seed = args.seed + i

        if fixed_noise:
            sigma = args.noise
        else:
            sigma = sample_noise(rng, args.noise_mu_log, args.noise_sigma_log, args.noise_max)
            policy.noise_scale = sigma

        print(f"  Episode {i+1}/{args.n_episodes}  seed={seed}  sigma={sigma:.4f} …", end=" ", flush=True)

        frames = run_episode(env, policy, seed=seed, max_steps=MAX_STEPS)

        vid_path = VIDEO_DIR / f"episode_{ep_idx:04d}.mp4"
        imageio.mimsave(str(vid_path), frames, fps=args.fps)
        ep_idx += 1
        print(f"{len(frames)} frames → {vid_path}")

    env.close()
    print(f"\nDone. {args.n_episodes} videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
