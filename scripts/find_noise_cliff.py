"""Empirically find the noise cliff for GuidedPolicy on PickCube-v1.

Sweeps sigma from --sigma-low to --sigma-high in --sigma-step increments,
runs --episodes episodes per level, fits a logistic curve to the success-rate
data, then derives log-uniform sampling bounds:

  sigma_min = sigma_c - ln(19)/k   [95% success rate]
  sigma_max = sigma_c + ln(19)/k   [5%  success rate]

No video or HDF5 output — pure simulation.

Usage:
    conda run -n ml_env python scripts/find_noise_cliff.py
    conda run -n ml_env python scripts/find_noise_cliff.py --episodes 50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy

TASK      = "PickCube-v1"
MAX_STEPS = 400
_WRIST    = None   # initialised after torch import (needs pi)


def parse_args():
    p = argparse.ArgumentParser(
        description="Find the noise cliff and derive log-uniform sampling bounds."
    )
    p.add_argument("--episodes",          type=int,   default=30,
                   help="Episodes per sigma level (default: 30)")
    p.add_argument("--sigma-low",         type=float, default=0.00,
                   help="Sweep start (default: 0.00)")
    p.add_argument("--sigma-high",        type=float, default=0.30,
                   help="Sweep end — set wider than expected cliff (default: 0.30)")
    p.add_argument("--sigma-step",        type=float, default=0.01,
                   help="Sweep step (default: 0.01)")
    p.add_argument("--seed",              type=int,   default=0,
                   help="First episode seed; uses seed..seed+episodes-1 per level")
    p.add_argument("--robot-qpos-noise",  type=float, default=0.10)
    p.add_argument("--cube-spawn-half",   type=float, default=0.15)
    return p.parse_args()


def fix_wrist(env):
    wrist = torch.tensor([0.0, math.pi * 3 / 4, math.pi / 4],
                          dtype=torch.float32)
    uwenv = env.unwrapped
    qpos  = uwenv.agent.robot.get_qpos().clone()
    qpos[:, 4:7] = wrist.to(qpos.device)
    uwenv.agent.robot.set_qpos(qpos)


def run_level(env, policy, sigma: float, n_episodes: int, base_seed: int) -> int:
    """Run n_episodes at this sigma. Returns number of successes."""
    policy.noise_scale = sigma
    successes = 0
    for i in range(n_episodes):
        env.reset(seed=base_seed + i)
        fix_wrist(env)
        policy.phases.zero_()
        policy.phase_steps.zero_()

        episode_success = False
        for _ in range(MAX_STEPS):
            action = policy()
            _, _, terminated, truncated, info = env.step(action)

            raw = info.get("success", False)
            if isinstance(raw, torch.Tensor):
                step_success = bool(raw[0].item())
            else:
                step_success = bool(raw)
            if step_success:
                episode_success = True

            if terminated.any() or truncated.any():
                break

        if episode_success:
            successes += 1

        print(f"    ep {i+1:02d}/{n_episodes}  success={episode_success}  "
              f"running={successes}/{i+1}", flush=True)

    return successes


def fit_logistic(sigmas: np.ndarray, rates: np.ndarray):
    """Fit logistic curve. Returns (k, sigma_c, sigma_c_err) or raises."""
    from scipy.optimize import curve_fit

    def logistic(sigma, k, sigma_c):
        return 1.0 / (1.0 + np.exp(k * (sigma - sigma_c)))

    popt, pcov = curve_fit(
        logistic, sigmas, rates,
        p0=[30.0, float(sigmas[len(sigmas) // 2])],
        bounds=([0.1, float(sigmas[0])], [500.0, float(sigmas[-1])]),
        maxfev=10000,
    )
    k, sigma_c = float(popt[0]), float(popt[1])
    perr = np.sqrt(np.diag(pcov))
    return k, sigma_c, float(perr[1])


def fallback_cliff(sigmas: np.ndarray, rates: np.ndarray):
    """Linear interpolation at 50% crossing + slope-based k estimate."""
    rates_arr = np.array(rates)
    below = np.where(rates_arr < 0.50)[0]
    if len(below) == 0:
        sigma_c = float(sigmas[-1])
        k = 1.0
    elif below[0] == 0:
        sigma_c = float(sigmas[0])
        k = 1.0
    else:
        i = below[0]
        r0, r1 = rates_arr[i - 1], rates_arr[i]
        s0, s1 = sigmas[i - 1], sigmas[i]
        sigma_c = float(s0 + (0.50 - r0) * (s1 - s0) / (r1 - r0))
        slope = (r1 - r0) / (s1 - s0)   # negative
        k = float(-4.0 * slope)          # logistic slope at 50% = -k/4
        k = max(k, 0.1)
    return k, sigma_c


def bar(rate: float, width: int = 30) -> str:
    filled = round(rate * width)
    return "█" * filled + " " * (width - filled)


def main():
    args = parse_args()

    sigmas = np.round(
        np.arange(args.sigma_low, args.sigma_high + 1e-9, args.sigma_step), 4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Creating env  task={TASK}  device={device}")
    env = gym.make(
        TASK,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=MAX_STEPS,
        robot_init_qpos_noise=args.robot_qpos_noise,
    )
    env.unwrapped.cube_spawn_half_size = args.cube_spawn_half

    policy = GuidedPolicy(env=env, num_envs=1, noise_scale=0.0, device=device)

    print(f"\nSweeping {len(sigmas)} sigma levels  "
          f"({args.episodes} episodes each, {len(sigmas) * args.episodes} total)\n")
    print(f"{'sigma':>6} | {'succ/tot':>8} | {'rate':>5} | {'bar':<30}")
    print("-" * 7 + "+" + "-" * 10 + "+" + "-" * 7 + "+" + "-" * 32)

    counts = []
    rates  = []
    for sigma in sigmas:
        n = run_level(env, policy, float(sigma), args.episodes, args.seed)
        r = n / args.episodes
        counts.append(n)
        rates.append(r)
        print(f" {sigma:5.2f} | {n:4d}/{args.episodes:<3d} | {r:5.2f} | {bar(r)}")

    env.close()

    # ── cliff detection ──────────────────────────────────────────────────────
    rates_arr = np.array(rates)
    method = ""
    try:
        k, sigma_c, sigma_c_err = fit_logistic(sigmas, rates_arr)
        method = f"logistic fit  σ_c={sigma_c:.4f} ± {sigma_c_err:.4f}  k={k:.1f}"
    except Exception as e:
        print(f"\n[warn] logistic fit failed ({e}), falling back to interpolation")
        k, sigma_c = fallback_cliff(sigmas, rates_arr)
        sigma_c_err = float("nan")
        method = f"linear interpolation  σ_c={sigma_c:.4f}  k={k:.1f}"

    # ── derived bounds ───────────────────────────────────────────────────────
    half_width = math.log(19) / k
    sigma_min  = max(0.005, sigma_c - half_width)
    sigma_max  = sigma_c + half_width

    # ── output ───────────────────────────────────────────────────────────────
    print(f"\nCliff detection: {method}")
    print(f"\nDerived log-uniform bounds:")
    print(f"  sigma_min = σ_c - ln(19)/k"
          f" = {sigma_c:.4f} - {half_width:.4f} = {sigma_min:.4f}   [95% success]")
    print(f"  sigma_max = σ_c + ln(19)/k"
          f" = {sigma_c:.4f} + {half_width:.4f} = {sigma_max:.4f}   [ 5% success]")
    print(f"\n  log-uniform range : [{sigma_min:.4f}, {sigma_max:.4f}]")
    print(f"  sigma_max/sigma_min = {sigma_max / sigma_min:.2f}"
          f"  (~{math.log(sigma_max / sigma_min) / math.log(10):.1f} decades)")
    print(f"\nUse with collect_parallel.py:")
    print(f"  --noise-sigma-min {sigma_min:.4f} --noise-sigma-max {sigma_max:.4f}")


if __name__ == "__main__":
    main()
