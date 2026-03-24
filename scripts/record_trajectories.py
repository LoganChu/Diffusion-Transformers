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

# Pull GuidedPolicy from ingest pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy

VIDEO_DIR = Path("outputs/videos")
TASK      = "PickCube-v1"
MAX_STEPS = 300   # slightly over the 4-phase budget (80+120+40+50=290)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--fps",        type=int, default=20)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--noise",      type=float, default=0.15,
                   help="Action noise std (lower = cleaner demo)")
    return p.parse_args()


def run_episode(env, policy, seed: int, max_steps: int):
    """Roll out one episode. Returns list of [H,W,3] uint8 frames."""
    obs, _ = env.reset(seed=seed)
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
    )

    policy = GuidedPolicy(
        env=env,
        num_envs=1,
        noise_scale=args.noise,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Find next free episode index
    existing = sorted(VIDEO_DIR.glob("episode_*.mp4"))
    ep_idx   = len(existing)

    for i in range(args.n_episodes):
        seed = args.seed + i
        print(f"  Episode {i+1}/{args.n_episodes}  seed={seed} …", end=" ", flush=True)

        frames = run_episode(env, policy, seed=seed, max_steps=MAX_STEPS)

        vid_path = VIDEO_DIR / f"episode_{ep_idx:04d}.mp4"
        imageio.mimsave(str(vid_path), frames, fps=args.fps)
        ep_idx += 1
        print(f"{len(frames)} frames → {vid_path}")

    env.close()
    print(f"\nDone. {args.n_episodes} videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
