"""Generate one video per noise level from 0.00 to 0.25 (step 0.01).

Usage:
    conda run -n ml_env python scripts/noise_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy

VIDEO_DIR = Path("outputs/videos/noise_sweep")
TASK      = "PickCube-v1"
MAX_STEPS = 400


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    sigmas = np.round(np.arange(0.00, 0.26, 0.01), 2)
    print(f"Generating {len(sigmas)} videos  sigma={sigmas}\n")

    env = gym.make(
        TASK,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=MAX_STEPS + 10,
        robot_init_qpos_noise=0.10,
    )
    env.unwrapped.cube_spawn_half_size = 0.15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = GuidedPolicy(env=env, num_envs=1, noise_scale=0.0, device=device)
    _WRIST = torch.tensor([0.0, np.pi * 3 / 4, np.pi / 4], dtype=torch.float32)

    for i, sigma in enumerate(sigmas):
        policy.noise_scale = float(sigma)

        obs, _ = env.reset(seed=42)
        uwenv  = env.unwrapped
        qpos   = uwenv.agent.robot.get_qpos().clone()
        qpos[:, 4:7] = _WRIST.to(qpos.device)
        uwenv.agent.robot.set_qpos(qpos)
        policy.phases.zero_()
        policy.phase_steps.zero_()

        frames = []
        for _ in range(MAX_STEPS):
            action = policy()
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame)
            if terminated.any() or truncated.any():
                break

        vid_path = VIDEO_DIR / f"sigma_{sigma:.2f}.mp4"
        imageio.mimsave(str(vid_path), frames, fps=20)
        print(f"  [{i+1:02d}/{len(sigmas)}]  sigma={sigma:.2f}  {len(frames)} frames -> {vid_path}")

    env.close()
    print(f"\nDone. {len(sigmas)} videos saved to {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()
