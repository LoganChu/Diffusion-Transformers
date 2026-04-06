"""Benchmark: KV-cache speedup for single-step Heun inference and CEM-MPC planning.

Two comparisons, both measuring cached vs. recompute:
  1. Single Heun step (BS=1, 8 ODE steps) — high-quality inference regime.
  2. CEM planning step (N=64, H=6, 3 iters, 4 ODE steps) — planning regime.

Usage:
    python -m inference.bench [--num_steps 8] [--n_ctx 2] [--warmup 5] [--repeats 50]
                              [--horizon 6] [--n_candidates 64] [--n_elites 8]
                              [--n_cem_iters 3] [--cem_ode_steps 4]
"""

from __future__ import annotations

import argparse

import torch

from inference.graph_solver import GraphedEulerStep, GraphedHeunSolver
from inference.planner import cem_plan, cube_height_score_fn, _euler_rollout_step
from models.cache import KVCache, RingKVCache
from models.dit import (
    ACTION_DIM,
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


@torch.no_grad()
def bench_cem_recompute(model, ctx_latents, ctx_actions, warmup, repeats,
                        horizon, n_candidates, n_elites, n_cem_iters, num_ode_steps):
    """CEM planning step — recompute baseline (context K/V recomputed every ODE step)."""
    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return cem_plan(
                model, ctx_latents, ctx_actions,
                score_fn=cube_height_score_fn,
                horizon=horizon,
                n_candidates=n_candidates,
                n_elites=n_elites,
                n_cem_iters=n_cem_iters,
                num_ode_steps=num_ode_steps,
            )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


def _cached_rollout_step(model, ctx_roll, a_cond, n_ctx_tokens, num_ode_steps, dtype):
    """One horizon step: prefill shared BS=1 context K/V once, reuse for all N candidates.

    All N candidates observe identical context (ctx_roll is an expand, not a copy),
    so a single BS=1 cache suffices. SDPA broadcasts [1, heads, n_ctx, hd] context
    K/V against [N, heads, n_denoise, hd] query tensors automatically.
    """
    N      = a_cond.shape[0]
    device = a_cond.device
    cache  = _make_cache(n_ctx_tokens, device, dtype)
    model.prefill_cache(ctx_roll[0:1], a_cond[0:1], cache)

    x     = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
    dt    = 1.0 / num_ode_steps
    t_buf = torch.empty(N, device=device, dtype=dtype)
    for i in range(num_ode_steps):
        t_buf.fill_(i * dt)
        v = model(x, t_buf, a_cond, cache=cache)
        x.add_(v, alpha=dt)
    return x


@torch.no_grad()
def bench_cem_cached(model, ctx_latents, ctx_actions, warmup, repeats,
                     horizon, n_candidates, n_elites, n_cem_iters, num_ode_steps):
    """CEM planning step — KV-cached (context prefilled once per horizon step)."""
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    N, H   = n_candidates, horizon
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES

    _ACTION_LO = torch.tensor([-0.08, -0.08, -0.08, -1.0], device=device, dtype=dtype)
    _ACTION_HI = torch.tensor([ 0.08,  0.08,  0.08,  1.0], device=device, dtype=dtype)

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ctx_base = ctx_latents.expand(N, -1, -1, -1, -1)
            mean = torch.zeros(H, ACTION_DIM, device=device, dtype=dtype)
            std  = torch.full((H, ACTION_DIM), 0.05, device=device, dtype=dtype)
            t_ones = torch.ones(N, device=device, dtype=dtype)

            for _ in range(n_cem_iters):
                eps     = torch.randn(N, H, ACTION_DIM, device=device, dtype=dtype)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(_ACTION_LO, _ACTION_HI)
                ctx_roll = ctx_base.clone()
                returns  = torch.zeros(N, device=device, dtype=torch.float32)

                for h in range(H):
                    a_cond = actions[:, h, :]
                    z_next = _cached_rollout_step(
                        model, ctx_roll, a_cond,
                        n_ctx_tokens, num_ode_steps, dtype,
                    )
                    step_score = cube_height_score_fn(model, z_next, t_ones)
                    returns.add_(step_score * float(0.99 ** h))
                    ctx_roll = torch.cat([ctx_roll[:, 1:], z_next.unsqueeze(1)], dim=1)

                _, elite_idx  = returns.topk(n_elites)
                elite_actions = actions[elite_idx].float()
                mean = elite_actions.mean(dim=0).to(dtype)
                std  = elite_actions.std(dim=0).clamp(min=0.01).to(dtype)

            return mean[0].float()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_graphed_euler_step(model, ctx_latents, ctx_actions, warmup, repeats,
                              n_candidates, num_ode_steps, cache_type="kv"):
    """Graph-replayed Euler step: 1 graph launch replaces num_ode_steps kernel storms."""
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    n_ctx  = ctx_latents.shape[1]
    N      = n_candidates

    solver = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                               num_ode_steps=num_ode_steps,
                               cache_type=cache_type, dtype=dtype)

    ctx_latents_1 = ctx_latents          # [1, n_ctx, C, H, W]
    g = torch.Generator(device=device).manual_seed(1)
    a_cond = torch.randn(N, ACTION_DIM, device=device, dtype=dtype, generator=g)
    x_init = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype, generator=g)

    def run():
        solver.run(model, ctx_latents_1, a_cond, x_init=x_init)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_ungraphed_euler_step(model, ctx_latents, ctx_actions, warmup, repeats,
                                n_candidates, num_ode_steps):
    """Ungraphed Euler step baseline (current _euler_rollout_step behaviour)."""
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    N      = n_candidates
    ctx_N  = ctx_latents.expand(N, -1, -1, -1, -1)

    g = torch.Generator(device=device).manual_seed(1)
    a_cond = torch.randn(N, ACTION_DIM, device=device, dtype=dtype, generator=g)

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _euler_rollout_step(model, ctx_N, a_cond, num_ode_steps, dtype)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_graphed_heun(model, ctx_latents, ctx_actions, action, warmup, repeats, num_steps):
    """Graph-replayed Heun solver: 1 graph launch replaces 2*num_steps-1 kernel storms."""
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    n_ctx  = ctx_latents.shape[1]

    solver = GraphedHeunSolver(model, n_ctx=n_ctx, num_steps=num_steps, dtype=dtype)

    g = torch.Generator(device=device).manual_seed(2)
    x_init = torch.randn(1, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype, generator=g)

    def run():
        solver.run(model, ctx_latents, ctx_actions, action, x_init=x_init)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_slide_physical(model, ctx_latents, ctx_actions, action, n_roll_frames, warmup, repeats):
    """Rolling context with physical-shift slide (KVCache.slide).

    Simulates K sequential frame generations each followed by a context-window
    update.  Pre-computed random K/V tensors stand in for encoded new frames so
    the ODE step cost is identical to bench_slide_ring — only the slide
    operation differs.
    """
    device = action.device
    dtype  = next(model.parameters()).dtype
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
    n_frame_shape = (1, NUM_HEADS, NUM_PATCHES, HEAD_DIM)

    # Pre-compute new-frame K/V (same for both benchmarks — eliminates noise)
    g = torch.Generator(device=device).manual_seed(0)
    new_frame_kvs = [
        (
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
        )
        for _ in range(n_roll_frames)
    ]

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            cache = _make_cache(n_ctx_tokens, device, dtype)
            model.prefill_cache(ctx_latents, ctx_actions, cache)
            for frame_idx in range(n_roll_frames):
                k_new, v_new = new_frame_kvs[frame_idx]
                for layer_idx in range(DEPTH):
                    cache.slide(layer_idx, k_new, v_new)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return elapsed


@torch.no_grad()
def bench_slide_ring(model, ctx_latents, ctx_actions, action, n_roll_frames, warmup, repeats):
    """Rolling context with ring-buffer slide (RingKVCache.slide_ring).

    Identical workload to bench_slide_physical — only the slide operation
    differs (O(n_frame) pointer write vs O(n_ctx) memory shift).
    """
    device = action.device
    dtype  = next(model.parameters()).dtype
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
    n_frame_shape = (1, NUM_HEADS, NUM_PATCHES, HEAD_DIM)

    g = torch.Generator(device=device).manual_seed(0)
    new_frame_kvs = [
        (
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
        )
        for _ in range(n_roll_frames)
    ]

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            cache = RingKVCache(
                DEPTH, NUM_HEADS, HEAD_DIM,
                n_ctx_tokens, NUM_PATCHES, NUM_PATCHES,
                device=device, dtype=dtype,
            )
            model.prefill_cache(ctx_latents, ctx_actions, cache)
            for frame_idx in range(n_roll_frames):
                k_new, v_new = new_frame_kvs[frame_idx]
                for layer_idx in range(DEPTH):
                    cache.slide_ring(layer_idx, k_new, v_new)
                cache.advance_head()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
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
    parser.add_argument("--num_steps",    type=int, default=8)
    parser.add_argument("--n_ctx",        type=int, default=2)
    parser.add_argument("--warmup",       type=int, default=5)
    parser.add_argument("--repeats",      type=int, default=50)
    # CEM-MPC params
    parser.add_argument("--horizon",       type=int, default=6)
    parser.add_argument("--n_candidates",  type=int, default=64)
    parser.add_argument("--n_elites",      type=int, default=8)
    parser.add_argument("--n_cem_iters",   type=int, default=3)
    parser.add_argument("--cem_ode_steps",   type=int, default=4)
    parser.add_argument("--n_roll_frames",   type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    model = DiTSmall().to(device=device, dtype=dtype).eval()
    B = 1
    ctx_latents = torch.randn(B, args.n_ctx, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
    ctx_actions = torch.randn(B, ACTION_DIM, device=device, dtype=dtype)
    action      = torch.randn(B, ACTION_DIM, device=device, dtype=dtype)

    print(f"Config: BS=1, n_ctx={args.n_ctx}, num_steps={args.num_steps}, "
          f"warmup={args.warmup}, repeats={args.repeats}, dtype={dtype}")
    print(f"CEM config: N={args.n_candidates}, H={args.horizon}, K={args.n_elites}, "
          f"iters={args.n_cem_iters}, ode_steps={args.cem_ode_steps}")
    print(f"Roll config: n_roll_frames={args.n_roll_frames}, n_ctx={args.n_ctx}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # --- Heun benchmarks ---
    times_heun_cached = bench_heun_cached(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )
    times_heun_recompute = bench_heun_recompute(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )

    n_heun_evals = 2 * args.num_steps - 1
    mean_heun_cached    = sum(times_heun_cached)    / len(times_heun_cached)
    mean_heun_recompute = sum(times_heun_recompute) / len(times_heun_recompute)
    heun_speedup        = mean_heun_recompute / mean_heun_cached

    # --- CEM benchmarks ---
    times_cem_recompute = bench_cem_recompute(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        horizon=args.horizon,
        n_candidates=args.n_candidates,
        n_elites=args.n_elites,
        n_cem_iters=args.n_cem_iters,
        num_ode_steps=args.cem_ode_steps,
    )
    times_cem_cached = bench_cem_cached(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        horizon=args.horizon,
        n_candidates=args.n_candidates,
        n_elites=args.n_elites,
        n_cem_iters=args.n_cem_iters,
        num_ode_steps=args.cem_ode_steps,
    )

    n_cem_evals      = args.n_cem_iters * args.horizon * args.cem_ode_steps
    mean_cem_recompute = sum(times_cem_recompute) / len(times_cem_recompute)
    mean_cem_cached    = sum(times_cem_cached)    / len(times_cem_cached)
    cem_speedup        = mean_cem_recompute / mean_cem_cached

    W = 62
    print("=" * W)
    print(f"  Single Inference Step  (Heun, BS=1, {n_heun_evals} evals)")
    print(f"{'Metric':<35} {'Cached':>10} {'Recompute':>10}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_heun_cached:>10.2f} {mean_heun_recompute:>10.2f}")
    print(f"{'ms/step':<35} {mean_heun_cached/args.num_steps:>10.2f} {mean_heun_recompute/args.num_steps:>10.2f}")
    print(f"{'ms/model_eval':<35} {mean_heun_cached/n_heun_evals:>10.2f} {mean_heun_recompute/n_heun_evals:>10.2f}")
    print(f"{'Speedup':<35} {heun_speedup:>10.2f}x")
    print("=" * W)
    print(f"  CEM Planning Step  (N={args.n_candidates}, H={args.horizon}, "
          f"iters={args.n_cem_iters}, {n_cem_evals} evals @ BS={args.n_candidates})")
    print(f"{'Metric':<35} {'Cached':>10} {'Recompute':>10}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_cem_cached:>10.2f} {mean_cem_recompute:>10.2f}")
    print(f"{'ms/CEM_iter':<35} {mean_cem_cached/args.n_cem_iters:>10.2f} {mean_cem_recompute/args.n_cem_iters:>10.2f}")
    print(f"{'ms/model_eval (@ BS={:d})'.format(args.n_candidates):<35} "
          f"{mean_cem_cached/n_cem_evals:>10.2f} {mean_cem_recompute/n_cem_evals:>10.2f}")
    print(f"{'Speedup':<35} {cem_speedup:>10.2f}x")
    print("=" * W)

    # Markdown block for results.md
    print("\n--- Markdown (copy to results.md) ---\n")
    print("### Heun Solver (BS=1)")
    print(f"| Solver | Total (ms) | ms/step | ms/eval | Speedup |")
    print(f"|--------|-----------|---------|---------|---------|")
    print(f"| Heun (KV-cached)   | {mean_heun_cached:.2f} | "
          f"{mean_heun_cached/args.num_steps:.2f} | "
          f"{mean_heun_cached/n_heun_evals:.2f} | **{heun_speedup:.2f}x** |")
    print(f"| Heun (recompute)   | {mean_heun_recompute:.2f} | "
          f"{mean_heun_recompute/args.num_steps:.2f} | "
          f"{mean_heun_recompute/n_heun_evals:.2f} | 1.00x |")
    print()
    print(f"### CEM-MPC Planning Step (N={args.n_candidates}, H={args.horizon}, "
          f"iters={args.n_cem_iters}, ode={args.cem_ode_steps})")
    print(f"| Planner | Total (ms) | ms/iter | ms/eval | Speedup |")
    print(f"|---------|-----------|---------|---------|---------|")
    print(f"| CEM (KV-cached)    | {mean_cem_cached:.2f} | "
          f"{mean_cem_cached/args.n_cem_iters:.2f} | "
          f"{mean_cem_cached/n_cem_evals:.2f} | **{cem_speedup:.2f}x** |")
    print(f"| CEM (recompute)    | {mean_cem_recompute:.2f} | "
          f"{mean_cem_recompute/args.n_cem_iters:.2f} | "
          f"{mean_cem_recompute/n_cem_evals:.2f} | 1.00x |")

    # --- CUDA graph benchmarks ---
    times_graphed_euler = bench_graphed_euler_step(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )
    times_ungraphed_euler = bench_ungraphed_euler_step(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )
    times_graphed_heun = bench_graphed_heun(
        model, ctx_latents, ctx_actions, action,
        args.warmup, args.repeats,
        num_steps=args.num_steps,
    )

    mean_graphed_euler   = sum(times_graphed_euler)   / len(times_graphed_euler)
    mean_ungraphed_euler = sum(times_ungraphed_euler) / len(times_ungraphed_euler)
    mean_graphed_heun    = sum(times_graphed_heun)    / len(times_graphed_heun)
    euler_graph_speedup  = mean_ungraphed_euler / mean_graphed_euler
    heun_graph_speedup   = mean_heun_recompute  / mean_graphed_heun

    n_euler_evals = args.cem_ode_steps

    print()
    print("=" * W)
    print(f"  CUDA Graph vs Eager  (Euler step, N={args.n_candidates}, ode={args.cem_ode_steps})")
    print(f"{'Metric':<35} {'Graphed':>10} {'Eager':>10}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_graphed_euler:>10.2f} {mean_ungraphed_euler:>10.2f}")
    print(f"{'ms/model_eval':<35} {mean_graphed_euler/n_euler_evals:>10.2f} {mean_ungraphed_euler/n_euler_evals:>10.2f}")
    print(f"{'Speedup':<35} {euler_graph_speedup:>10.2f}x")
    print("=" * W)
    print(f"  CUDA Graph vs Eager  (Heun BS=1, steps={args.num_steps}, {n_heun_evals} evals)")
    print(f"{'Metric':<35} {'Graphed':>10} {'Eager KV':>10}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_graphed_heun:>10.2f} {mean_heun_cached:>10.2f}")
    print(f"{'ms/model_eval':<35} {mean_graphed_heun/n_heun_evals:>10.2f} {mean_heun_cached/n_heun_evals:>10.2f}")
    print(f"{'Speedup vs KV-cached eager':<35} {heun_graph_speedup:>10.2f}x")
    print("=" * W)

    print()
    print(f"### CUDA Graph Speedup")
    print(f"| Solver | Total (ms) | ms/eval | Speedup vs eager |")
    print(f"|--------|-----------|---------|-----------------|")
    print(f"| Euler step graphed (N={args.n_candidates}) | {mean_graphed_euler:.2f} | "
          f"{mean_graphed_euler/n_euler_evals:.2f} | **{euler_graph_speedup:.2f}x** |")
    print(f"| Euler step eager               | {mean_ungraphed_euler:.2f} | "
          f"{mean_ungraphed_euler/n_euler_evals:.2f} | 1.00x |")
    print(f"| Heun graphed (BS=1)            | {mean_graphed_heun:.2f} | "
          f"{mean_graphed_heun/n_heun_evals:.2f} | **{heun_graph_speedup:.2f}x** |")
    print(f"| Heun KV-cached eager           | {mean_heun_cached:.2f} | "
          f"{mean_heun_cached/n_heun_evals:.2f} | 1.00x |")

    # --- Slide benchmarks ---
    times_slide_phys = bench_slide_physical(
        model, ctx_latents, ctx_actions, action,
        args.n_roll_frames, args.warmup, args.repeats,
    )
    times_slide_ring = bench_slide_ring(
        model, ctx_latents, ctx_actions, action,
        args.n_roll_frames, args.warmup, args.repeats,
    )

    mean_slide_phys = sum(times_slide_phys) / len(times_slide_phys)
    mean_slide_ring = sum(times_slide_ring) / len(times_slide_ring)
    slide_speedup   = mean_slide_phys / mean_slide_ring

    print()
    print("=" * W)
    print(f"  Rolling Context Slide  ({args.n_roll_frames} frames, n_ctx={args.n_ctx})")
    print(f"{'Metric':<35} {'Ring':>10} {'Physical':>10}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_slide_ring:>10.2f} {mean_slide_phys:>10.2f}")
    print(f"{'ms/frame':<35} {mean_slide_ring/args.n_roll_frames:>10.2f} {mean_slide_phys/args.n_roll_frames:>10.2f}")
    print(f"{'Speedup':<35} {slide_speedup:>10.2f}x")
    print("=" * W)

    print()
    print(f"### Rolling Context Slide ({args.n_roll_frames} frames, n_ctx={args.n_ctx})")
    print(f"| Slide strategy | Total (ms) | ms/frame | Speedup |")
    print(f"|----------------|-----------|----------|---------|")
    print(f"| Ring buffer    | {mean_slide_ring:.2f} | "
          f"{mean_slide_ring/args.n_roll_frames:.2f} | **{slide_speedup:.2f}x** |")
    print(f"| Physical shift | {mean_slide_phys:.2f} | "
          f"{mean_slide_phys/args.n_roll_frames:.2f} | 1.00x |")


if __name__ == "__main__":
    main()
