"""Benchmark: KV-cached Heun vs recompute-baseline Heun at BS=1.

Usage:
    python -m inference.bench [--num_steps 8] [--n_ctx 2] [--warmup 5] [--repeats 50]

Outputs per-step latency (ms) for both paths and the speedup ratio.
"""

from __future__ import annotations

import argparse
import time

import torch

from models.cache import KVCache
from models.dit import (
    DEPTH,
    HEAD_DIM,
    IN_CHANNELS,
    LATENT_H,
    LATENT_W,
    NUM_HEADS,
    NUM_PATCHES,
    DiTSmall,
)


def _make_cache(n_ctx_tokens: int, device, dtype):
    return KVCache(
        DEPTH, NUM_HEADS, HEAD_DIM,
        n_ctx_tokens, NUM_PATCHES,
        device=device, dtype=dtype,
    )


@torch.no_grad()
def bench_heun_cached(model, ctx_latents, ctx_actions, action, num_steps, warmup, repeats):
    """Heun with persistent cache (prefill once)."""
    B = 1
    device = action.device
    dtype = next(model.parameters()).dtype
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
    dt = 1.0 / num_steps
    t_buf = torch.empty(B, device=device, dtype=dtype)

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            cache = _make_cache(n_ctx_tokens, device, dtype)
            model.prefill_cache(ctx_latents, ctx_actions, cache)
            x = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
            x_euler = torch.empty_like(x)

            for i in range(num_steps):
                t_buf.fill_(i * dt)
                v1 = model(x, t_buf, action, cache=cache)
                if i < num_steps - 1:
                    torch.add(x, v1, alpha=dt, out=x_euler)
                    t_buf.fill_((i + 1) * dt)
                    v2 = model(x_euler, t_buf, action, cache=cache)
                    v1.add_(v2)
                    x.add_(v1, alpha=dt * 0.5)
                else:
                    x.add_(v1, alpha=dt)
            return x

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_heun_recompute(model, ctx_latents, ctx_actions, action, num_steps, warmup, repeats):
    """Heun with fresh cache per model evaluation (recompute baseline)."""
    B = 1
    device = action.device
    dtype = next(model.parameters()).dtype
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
    dt = 1.0 / num_steps

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            x = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)

            for i in range(num_steps):
                cache = _make_cache(n_ctx_tokens, device, dtype)
                model.prefill_cache(ctx_latents, ctx_actions, cache)
                t_val = torch.full((B,), i * dt, device=device, dtype=dtype)
                v1 = model(x, t_val, action, cache=cache)

                if i < num_steps - 1:
                    x_euler = x + dt * v1
                    cache2 = _make_cache(n_ctx_tokens, device, dtype)
                    model.prefill_cache(ctx_latents, ctx_actions, cache2)
                    t_next = torch.full((B,), (i + 1) * dt, device=device, dtype=dtype)
                    v2 = model(x_euler, t_next, action, cache=cache2)
                    x = x + dt * 0.5 * (v1 + v2)
                else:
                    x = x + dt * v1
            return x

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    model = DiTSmall().to(device=device, dtype=dtype).eval()
    B = 1
    ctx_latents = torch.randn(B, args.n_ctx, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
    ctx_actions = torch.randn(B, 8, device=device, dtype=dtype)
    action = torch.randn(B, 8, device=device, dtype=dtype)

    print(f"Config: BS=1, n_ctx={args.n_ctx}, num_steps={args.num_steps}, "
          f"warmup={args.warmup}, repeats={args.repeats}, dtype={dtype}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Heun cached
    times_cached = bench_heun_cached(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )
    # Heun recompute
    times_recompute = bench_heun_recompute(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )

    # Heun uses (2*num_steps - 1) model evals
    n_evals = 2 * args.num_steps - 1

    mean_cached = sum(times_cached) / len(times_cached)
    mean_recompute = sum(times_recompute) / len(times_recompute)

    ms_per_step_cached = mean_cached / args.num_steps
    ms_per_step_recompute = mean_recompute / args.num_steps
    ms_per_eval_cached = mean_cached / n_evals
    ms_per_eval_recompute = mean_recompute / n_evals

    speedup = mean_recompute / mean_cached

    print("=" * 60)
    print(f"{'Metric':<35} {'Cached':>10} {'Recompute':>10}")
    print("-" * 60)
    print(f"{'Total time (ms)':<35} {mean_cached:>10.2f} {mean_recompute:>10.2f}")
    print(f"{'ms/step':<35} {ms_per_step_cached:>10.2f} {ms_per_step_recompute:>10.2f}")
    print(f"{'ms/model_eval':<35} {ms_per_eval_cached:>10.2f} {ms_per_eval_recompute:>10.2f}")
    print(f"{'Speedup':<35} {speedup:>10.2f}x")
    print("=" * 60)

    # Output markdown for results.md
    print("\n--- Markdown (copy to results.md) ---\n")
    print(f"| Solver | Total (ms) | ms/step | ms/eval | Speedup |")
    print(f"|--------|-----------|---------|---------|---------|")
    print(f"| Heun (KV-cached) | {mean_cached:.2f} | {ms_per_step_cached:.2f} | {ms_per_eval_cached:.2f} | **{speedup:.2f}x** |")
    print(f"| Heun (recompute) | {mean_recompute:.2f} | {ms_per_step_recompute:.2f} | {ms_per_eval_recompute:.2f} | 1.00x |")


if __name__ == "__main__":
    main()
