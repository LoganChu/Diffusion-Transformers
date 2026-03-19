"""scripts/verify_env.py
────────────────────────────────────────────────────────────────────────────────
ManiSkill 3.0 validation and visualisation script.

DESIGN NOTE — Vulkan workaround
  render_mode='rgb_array' hard-crashes (exit 127) on RTX 50-series Windows when
  CUDA is already initialised in the same process (SAPIEN Vulkan-CUDA interop
  bug — see project MEMORY.md).  We therefore use render_mode=None and implement
  our own imageio-based MP4 writer with synthetic 512×512 frames derived from the
  observation state vector.  The camera constant (CAM_W × CAM_H = 512×512) is
  preserved as the output frame resolution.

Capabilities
  • obs_mode='state' + render_mode=None → safe on all tested hardware
  • Synthetic 512×512 frame generation from state tensor (consistent with ingest.py)
  • imageio MP4 output → outputs/videos/episode_<N>.mp4
  • render_human() behind --viewer flag with try/except fallback
  • Dual-path latency profiling: headless step() vs step()+render_human()
  • NVTX ranges + torch.profiler.record_function for Nsight Systems

Usage
  conda run -n ml_env python scripts/verify_env.py
  conda run -n ml_env python scripts/verify_env.py --viewer
  nsys profile -w true -t cuda,nvtx,osrt -o profiler/rep_%b ^
      conda run -n ml_env python scripts/verify_env.py
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from typing import List

import imageio
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import record_function

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers PickCube-v1 etc.

# ─── Constants ────────────────────────────────────────────────────────────────
TASK      = "PickCube-v1"
CAM_W     = 512          # target output frame width  (static base_camera)
CAM_H     = 512          # target output frame height
MAX_STEPS = 200          # episode length (random-action rollout)
N_WARMUP  = 20           # steps excluded from latency measurement
VIDEO_DIR = Path("outputs/videos")
FPS       = 20


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ManiSkill env validation + latency profiler")
    p.add_argument("--task",     default=TASK,        help="ManiSkill task ID")
    p.add_argument("--viewer",        action="store_true", help="Enable render_human() (requires display)")
    p.add_argument("--viewer-delay",  type=float, default=0.05,
                   help="Seconds to sleep after each render_human() call (default 0.05 = 20 fps). "
                        "Increase to slow the viewer, 0 to run at full speed.")
    p.add_argument("--n-warmup", type=int, default=N_WARMUP)
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


# ─── Synthetic frame generator ────────────────────────────────────────────────
def state_to_frame(obs: torch.Tensor, step: int, reward: float) -> np.ndarray:
    """
    Convert a state observation tensor → [CAM_H, CAM_W, 3] uint8 frame.

    Strategy (consistent with ingest.py's random-projection fallback):
      1. Flatten state → 1-D signal of length D
      2. Normalise to [0, 255]
      3. Nearest-neighbour stretch to CAM_W via np.linspace indexing
      4. Tile the row CAM_H times → [CAM_H, CAM_W]
      5. Phase-shift the row by ±CAM_W//3 for G and B channels

    This is O(D) and always produces exactly (CAM_H, CAM_W, 3) regardless of D.
    """
    flat  = obs.flatten().float().cpu().numpy().astype(np.float32)  # [D]
    D     = len(flat)
    vmin, vmax = float(flat.min()), float(flat.max())
    normed = ((flat - vmin) / (vmax - vmin + 1e-6) * 255).clip(0, 255).astype(np.uint8)

    # Stretch D-element signal to CAM_W via nearest-neighbour sampling
    idx = np.floor(np.linspace(0, D - 1, CAM_W)).astype(int)
    row = normed[idx]                                    # [CAM_W]

    ch_r = np.tile(row,                              (CAM_H, 1))  # [CAM_H, CAM_W]
    ch_g = np.tile(np.roll(row,     CAM_W // 3),     (CAM_H, 1))
    ch_b = np.tile(np.roll(row, 2 * CAM_W // 3),     (CAM_H, 1))

    return np.stack([ch_r, ch_g, ch_b], axis=-1)  # [CAM_H, CAM_W, 3] uint8


# ─── Environment factory ──────────────────────────────────────────────────────
def make_env(task: str) -> gym.Env:
    """
    Create a BS-1 ManiSkill env.

    render_mode is intentionally omitted:  on RTX 50-series Windows,
    render_mode='rgb_array' triggers SAPIEN Vulkan initialisation in-process
    which hard-crashes (exit 127) once the CUDA context is live.  Frame
    generation is handled separately by state_to_frame().
    """
    nvtx.range_push("verify_env/init")
    with record_function("verify_env.init"):
        env = gym.make(task, num_envs=1, obs_mode="state")
    nvtx.range_pop()
    return env


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── 1. Build environment ──────────────────────────────────────────────────
    env = make_env(args.task)
    print(f"[OK] env created  task={args.task}  obs_mode=state")
    print(f"     Synthetic camera : base_camera @ {CAM_W}×{CAM_H}  (state→frame)")

    # ── 2. Reset ──────────────────────────────────────────────────────────────
    nvtx.range_push("verify_env/reset")
    with record_function("verify_env.reset"):
        obs, _ = env.reset(seed=args.seed)
    nvtx.range_pop()

    # ManiSkill with obs_mode='state' returns a flat Tensor [num_envs, D]
    # Squeeze batch dim for single-env use
    obs_vec = obs.squeeze(0) if obs.ndim > 1 else obs
    print(f"[OK] env.reset()  obs.shape={tuple(obs.shape)}")

    # Sanity-check the synthetic frame pipeline
    test_frame = state_to_frame(obs_vec, step=0, reward=0.0)
    assert test_frame.shape == (CAM_H, CAM_W, 3), f"Frame shape error: {test_frame.shape}"
    print(f"[OK] synthetic frame verified @ {CAM_W}×{CAM_H}  dtype={test_frame.dtype}")

    # ── 3. Probe render_human() ───────────────────────────────────────────────
    viewer_ok = False
    if args.viewer:
        nvtx.range_push("verify_env/render_human_probe")
        try:
            with record_function("verify_env.render_human_probe"):
                env.unwrapped.render_human()
            viewer_ok = True
            print("[OK] render_human() initialised — 3-D viewer active.")
        except Exception as exc:
            print(f"[WARN] render_human() unavailable: {exc}")
            print("       Continuing in headless mode.")
        finally:
            nvtx.range_pop()

    # ── 4. Prepare frame buffer (collect-then-write avoids codec quirks) ──────
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ep_idx   = sum(1 for f in VIDEO_DIR.glob("episode_*.mp4"))
    vid_path = VIDEO_DIR / f"episode_{ep_idx:04d}.mp4"
    frames: List[np.ndarray] = []
    print(f"[  ] Recording → {vid_path}  ({CAM_W}×{CAM_H} @ {FPS} fps)")

    # ── 5. Episode loop: MAX_STEPS steps with random actions ─────────────────
    print(f"\n[RUN] {MAX_STEPS} steps  warmup={args.n_warmup}  seed={args.seed} …")

    step_ms:         list[float] = []
    render_human_ms: list[float] = []

    for step_i in range(MAX_STEPS):
        action = env.action_space.sample()

        # ── env.step() ──────────────────────────────────────────────────────
        nvtx.range_push("verify_env/step")
        with record_function("verify_env.step"):
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        nvtx.range_pop()

        if step_i >= args.n_warmup:
            step_ms.append((t1 - t0) * 1e3)

        # ── Synthetic frame → frame buffer ───────────────────────────────────
        obs_vec = obs.squeeze(0) if obs.ndim > 1 else obs
        rew_val = float(reward.squeeze()) if hasattr(reward, "squeeze") else float(reward)
        frames.append(state_to_frame(obs_vec, step_i, rew_val))

        # ── render_human() ──────────────────────────────────────────────────
        if viewer_ok:
            nvtx.range_push("verify_env/render_human")
            with record_function("verify_env.render_human"):
                try:
                    tr0 = time.perf_counter()
                    env.unwrapped.render_human()
                    tr1 = time.perf_counter()
                    if step_i >= args.n_warmup:
                        render_human_ms.append((tr1 - tr0) * 1e3)
                    # Throttle viewer to human-observable speed (excluded from timing)
                    if args.viewer_delay > 0:
                        time.sleep(args.viewer_delay)
                except Exception as exc:
                    print(f"[WARN] render_human() failed at step {step_i}: {exc}")
                    print("       Viewer disabled for remaining steps.")
                    viewer_ok = False
            nvtx.range_pop()

        # Episode auto-reset
        if terminated or truncated:
            nvtx.range_push("verify_env/reset")
            with record_function("verify_env.reset"):
                obs, _ = env.reset()
            nvtx.range_pop()

    # ── 6. Write MP4 + close env ──────────────────────────────────────────────
    nvtx.range_push("verify_env/close")
    imageio.mimsave(str(vid_path), frames, fps=FPS)   # collect-then-write
    env.close()
    nvtx.range_pop()
    assert frames[0].shape == (CAM_H, CAM_W, 3), f"Frame shape error: {frames[0].shape}"
    print(f"[OK] env.close() — {len(frames)} frames @ {frames[0].shape} written to {vid_path}")

    # ── 7. Latency report ─────────────────────────────────────────────────────
    hs = np.asarray(step_ms,         dtype=np.float64)
    rs = np.asarray(render_human_ms, dtype=np.float64)
    n  = len(hs)

    print("  verify_env  Latency Report")
    print(f"  Camera    : base_camera @ {CAM_W}×{CAM_H}  (synthetic, state→RGB)")
    print(f"  Task      : {args.task}  |  obs_mode=state  |  num_envs=1")
    print(f"  Samples   : {n} steps measured (first {args.n_warmup} discarded as warmup)")
    print()

    if n:
        print(f"  Headless step()        :  {np.mean(hs):7.3f} ± {np.std(hs):.3f} ms"
              f"  [p50={np.median(hs):.3f}  p99={np.percentile(hs,99):.3f}]")
    if rs.size:
        overhead = np.mean(rs)
        total    = np.mean(hs) + overhead
        pct      = 100.0 * overhead / total if total > 0 else 0.0
        print(f"  render_human() call    :  {overhead:7.3f} ± {np.std(rs):.3f} ms")
        print(f"  Total (step + render)  :  {total:7.3f} ms")
        print(f"  Viewer overhead        :  {pct:.1f}% of combined wall time")
    else:
        print("  render_human()         :  N/A — headless (use --viewer to enable)")

    print(f"\n  MP4 → {vid_path}  ({MAX_STEPS} frames @ {FPS} fps)")
    print("\nProfile with Nsight Systems:")
    print("  nsys profile -w true -t cuda,nvtx,osrt -o profiler/rep_%b \\")
    print("      conda run -n ml_env python scripts/verify_env.py")


if __name__ == "__main__":
    main()
