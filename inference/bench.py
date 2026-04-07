"""Benchmark: KV-cache speedup for single-step inference and MPC candidate rollouts.

Two comparisons, both measuring cached vs. recompute:
  1. Single Euler step (BS=1, 8 ODE steps) — reactive control regime.
  2. Single MPC step (N=64, 4 ODE steps) — planning regime.
     One prefill → N candidates × ode_steps evals is the atomic unit the cache
     accelerates. Horizon and CEM iteration loops are excluded — they add no
     additional cache utilization (same speedup ratio at every step).

Usage:
    python -m inference.bench [--num_steps 8] [--n_ctx 2] [--warmup 5] [--repeats 50]
                              [--n_candidates 64] [--cem_ode_steps 4]
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


def _make_cache(n_ctx_tokens: int, device, dtype, cache_type: str = "kv"):
    if cache_type == "ring":
        return RingKVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES, NUM_PATCHES,
            device=device, dtype=dtype,
        )
    return KVCache(
        DEPTH, NUM_HEADS, HEAD_DIM,
        n_ctx_tokens, NUM_PATCHES,
        device=device, dtype=dtype,
    )



@torch.no_grad()
def bench_euler_cached_bs1(model, ctx_latents, ctx_actions, action, num_steps, warmup, repeats):
    """Euler (BS=1) with persistent cache (prefill once)."""
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
            for i in range(num_steps):
                t_buf.fill_(i * dt)
                v = model(x, t_buf, action, cache=cache)
                x.add_(v, alpha=dt)
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
def bench_euler_cached_bs1_ring(model, ctx_latents, ctx_actions, action, num_steps, warmup, repeats):
    """Euler (BS=1) with persistent ring-buffer cache (prefill once)."""
    B = 1
    device = action.device
    dtype = next(model.parameters()).dtype
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
    dt = 1.0 / num_steps
    t_buf = torch.empty(B, device=device, dtype=dtype)

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            cache = _make_cache(n_ctx_tokens, device, dtype, cache_type="ring")
            model.prefill_cache(ctx_latents, ctx_actions, cache)
            x = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
            for i in range(num_steps):
                t_buf.fill_(i * dt)
                v = model(x, t_buf, action, cache=cache)
                x.add_(v, alpha=dt)
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
def bench_euler_recompute_bs1(model, ctx_latents, ctx_actions, action, num_steps, warmup, repeats):
    """Euler (BS=1) with fresh cache per step (recompute baseline)."""
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
                v = model(x, t_val, action, cache=cache)
                x = x + dt * v
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
def bench_mpc_recompute(model, ctx_latents, ctx_actions, warmup, repeats,
                        n_candidates, num_ode_steps):
    """Single MPC step — recompute baseline (context K/V recomputed every ODE step).

    Times one rollout of N candidates with no caching: context is re-encoded
    at every ODE step for every candidate via _euler_rollout_step.
    One MPC step is the atomic unit — the speedup ratio is the same at every horizon step.
    """
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    N      = n_candidates
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            a_cond   = torch.randn(N, ACTION_DIM, device=device, dtype=dtype)
            ctx_N    = ctx_latents.expand(N, -1, -1, -1, -1)
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


def _cached_rollout_step(model, ctx_roll, a_cond, n_ctx_tokens, num_ode_steps, dtype, cache_type: str = "kv"):
    """One horizon step: prefill shared BS=1 context K/V once, reuse for all N candidates.

    All N candidates observe identical context (ctx_roll is an expand, not a copy),
    so a single BS=1 cache suffices. SDPA broadcasts [1, heads, n_ctx, hd] context
    K/V against [N, heads, n_denoise, hd] query tensors automatically.
    """
    N      = a_cond.shape[0]
    device = a_cond.device
    cache  = _make_cache(n_ctx_tokens, device, dtype, cache_type=cache_type)
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
def bench_mpc_cached(model, ctx_latents, ctx_actions, warmup, repeats,
                     n_candidates, num_ode_steps):
    """Single MPC step — KV-cached (context prefilled once, broadcast to N candidates).

    Times one rollout of N candidates with shared BS=1 context cache:
    prefill once, SDPA broadcasts to all N candidates across all ODE steps.
    One MPC step is the atomic unit — the speedup ratio is the same at every horizon step.
    """
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    N      = n_candidates
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            a_cond = torch.randn(N, ACTION_DIM, device=device, dtype=dtype)
            _cached_rollout_step(model, ctx_latents, a_cond, n_ctx_tokens, num_ode_steps, dtype, cache_type="kv")

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
def bench_mpc_cached_ring(model, ctx_latents, ctx_actions, warmup, repeats,
                          n_candidates, num_ode_steps):
    """Single MPC step — Ring-buffer cached (context prefilled once, broadcast to N candidates).

    Times one rollout of N candidates with shared BS=1 ring-buffer context cache:
    prefill once, SDPA broadcasts to all N candidates across all ODE steps.
    One MPC step is the atomic unit — the speedup ratio is the same at every horizon step.
    """
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    N      = n_candidates
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES

    def run():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            a_cond = torch.randn(N, ACTION_DIM, device=device, dtype=dtype)
            _cached_rollout_step(model, ctx_latents, a_cond, n_ctx_tokens, num_ode_steps, dtype, cache_type="ring")

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
def bench_graphed_euler_step_both(model, ctx_latents, ctx_actions, warmup, repeats,
                                   n_candidates, num_ode_steps):
    """Benchmark graphed Euler step for both KV and ring-buffer cache types."""
    results = {}
    for cache_type in ["kv", "ring"]:
        results[cache_type] = bench_graphed_euler_step(
            model, ctx_latents, ctx_actions, warmup, repeats,
            n_candidates, num_ode_steps, cache_type=cache_type
        )
    return results


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
def bench_graphed_euler_bs1(model, ctx_latents, ctx_actions, action, warmup, repeats, num_steps, cache_type="kv"):
    """Graph-replayed Euler solver (BS=1): 1 graph launch replaces num_steps kernel storms."""
    device = ctx_latents.device
    dtype  = next(model.parameters()).dtype
    n_ctx  = ctx_latents.shape[1]

    solver = GraphedHeunSolver(model, n_ctx=n_ctx, num_steps=num_steps, cache_type=cache_type, dtype=dtype)

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
def bench_graphed_euler_bs1_both(model, ctx_latents, ctx_actions, action, warmup, repeats, num_steps):
    """Benchmark graphed Euler BS=1 for both KV and ring-buffer cache types."""
    results = {}
    for cache_type in ["kv", "ring"]:
        results[cache_type] = bench_graphed_euler_bs1(
            model, ctx_latents, ctx_actions, action, warmup, repeats,
            num_steps, cache_type=cache_type
        )
    return results


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
    # MPC params
    parser.add_argument("--n_candidates",  type=int, default=64)
    parser.add_argument("--cem_ode_steps", type=int, default=4)
    parser.add_argument("--n_roll_frames", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    model = DiTSmall().to(device=device, dtype=dtype).eval()
    ckpt = torch.load("offline_best.pt", map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)

    B = 1
    ctx_latents = torch.randn(B, args.n_ctx, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
    ctx_actions = torch.randn(B, ACTION_DIM, device=device, dtype=dtype)
    action      = torch.randn(B, ACTION_DIM, device=device, dtype=dtype)

    print(f"Config: BS=1, n_ctx={args.n_ctx}, num_steps={args.num_steps}, "
          f"warmup={args.warmup}, repeats={args.repeats}, dtype={dtype}")
    print(f"MPC config: N={args.n_candidates}, ode_steps={args.cem_ode_steps}")
    print(f"Roll config: n_roll_frames={args.n_roll_frames}, n_ctx={args.n_ctx}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # --- Euler BS=1 benchmarks ---
    times_euler_cached = bench_euler_cached_bs1(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )
    times_euler_cached_ring = bench_euler_cached_bs1_ring(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )
    times_euler_recompute = bench_euler_recompute_bs1(
        model, ctx_latents, ctx_actions, action,
        args.num_steps, args.warmup, args.repeats,
    )

    n_euler_evals_bs1    = args.num_steps
    mean_euler_cached    = sum(times_euler_cached)    / len(times_euler_cached)
    mean_euler_cached_ring = sum(times_euler_cached_ring) / len(times_euler_cached_ring)
    mean_euler_recompute = sum(times_euler_recompute) / len(times_euler_recompute)
    euler_bs1_speedup    = mean_euler_recompute / mean_euler_cached
    euler_bs1_kv_vs_ring = mean_euler_cached_ring / mean_euler_cached

    # --- Single MPC step benchmarks ---
    times_mpc_recompute = bench_mpc_recompute(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )
    times_mpc_cached = bench_mpc_cached(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )
    times_mpc_cached_ring = bench_mpc_cached_ring(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )

    n_mpc_evals        = args.cem_ode_steps
    mean_mpc_recompute = sum(times_mpc_recompute) / len(times_mpc_recompute)
    mean_mpc_cached    = sum(times_mpc_cached)    / len(times_mpc_cached)
    mean_mpc_cached_ring = sum(times_mpc_cached_ring) / len(times_mpc_cached_ring)
    mpc_speedup        = mean_mpc_recompute / mean_mpc_cached
    mpc_kv_vs_ring     = mean_mpc_cached_ring / mean_mpc_cached

    W = 75
    print("=" * W)
    print(f"  Single Inference Step  (Euler, BS=1, {n_euler_evals_bs1} evals)")
    print(f"{'Metric':<35} {'KV Cache':>12} {'Ring Cache':>12} {'Recompute':>12}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_euler_cached:>12.2f} {mean_euler_cached_ring:>12.2f} {mean_euler_recompute:>12.2f}")
    print(f"{'ms/step':<35} {mean_euler_cached/args.num_steps:>12.2f} {mean_euler_cached_ring/args.num_steps:>12.2f} {mean_euler_recompute/args.num_steps:>12.2f}")
    print(f"{'ms/model_eval':<35} {mean_euler_cached/n_euler_evals_bs1:>12.2f} {mean_euler_cached_ring/n_euler_evals_bs1:>12.2f} {mean_euler_recompute/n_euler_evals_bs1:>12.2f}")
    print(f"{'Speedup vs recompute':<35} {euler_bs1_speedup:>12.2f}x {mean_euler_recompute/mean_euler_cached_ring:>12.2f}x")
    print(f"{'Ring/KV ratio':<35} {euler_bs1_kv_vs_ring:>12.2f}x")
    print("=" * W)
    print(f"  Single MPC Step  (N={args.n_candidates}, {n_mpc_evals} evals @ BS={args.n_candidates})")
    print(f"{'Metric':<35} {'KV Cache':>12} {'Ring Cache':>12} {'Recompute':>12}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_mpc_cached:>12.2f} {mean_mpc_cached_ring:>12.2f} {mean_mpc_recompute:>12.2f}")
    print(f"{'ms/model_eval (@ BS={:d})'.format(args.n_candidates):<35} "
          f"{mean_mpc_cached/n_mpc_evals:>12.2f} {mean_mpc_cached_ring/n_mpc_evals:>12.2f} {mean_mpc_recompute/n_mpc_evals:>12.2f}")
    print(f"{'Speedup vs recompute':<35} {mpc_speedup:>12.2f}x {mean_mpc_recompute/mean_mpc_cached_ring:>12.2f}x")
    print(f"{'Ring/KV ratio':<35} {mpc_kv_vs_ring:>12.2f}x")
    print("=" * W)

    # Markdown block for results.md
    print("\n--- Markdown (copy to results.md) ---\n")
    print("### Euler Solver (BS=1)")
    print(f"| Solver | Total (ms) | ms/step | ms/eval | Speedup vs recompute |")
    print(f"|--------|-----------|---------|---------|------------|")
    print(f"| Euler (KV-cached)  | {mean_euler_cached:.2f} | "
          f"{mean_euler_cached/args.num_steps:.2f} | "
          f"{mean_euler_cached/n_euler_evals_bs1:.2f} | **{euler_bs1_speedup:.2f}x** |")
    print(f"| Euler (Ring-cached)  | {mean_euler_cached_ring:.2f} | "
          f"{mean_euler_cached_ring/args.num_steps:.2f} | "
          f"{mean_euler_cached_ring/n_euler_evals_bs1:.2f} | **{mean_euler_recompute/mean_euler_cached_ring:.2f}x** |")
    print(f"| Euler (recompute)  | {mean_euler_recompute:.2f} | "
          f"{mean_euler_recompute/args.num_steps:.2f} | "
          f"{mean_euler_recompute/n_euler_evals_bs1:.2f} | 1.00x |")
    print(f"| **KV/Ring ratio** | **{euler_bs1_kv_vs_ring:.2f}x** | | | |")
    print()
    print(f"### Single MPC Step (N={args.n_candidates}, ode={args.cem_ode_steps})")
    print(f"| Planner | Total (ms) | ms/eval | Speedup vs recompute |")
    print(f"|---------|-----------|---------|------------|")
    print(f"| MPC (KV-cached)    | {mean_mpc_cached:.2f} | "
          f"{mean_mpc_cached/n_mpc_evals:.2f} | **{mpc_speedup:.2f}x** |")
    print(f"| MPC (Ring-cached)    | {mean_mpc_cached_ring:.2f} | "
          f"{mean_mpc_cached_ring/n_mpc_evals:.2f} | **{mean_mpc_recompute/mean_mpc_cached_ring:.2f}x** |")
    print(f"| MPC (recompute)    | {mean_mpc_recompute:.2f} | "
          f"{mean_mpc_recompute/n_mpc_evals:.2f} | 1.00x |")
    print(f"| **KV/Ring ratio** | **{mpc_kv_vs_ring:.2f}x** | | |")

    # --- CUDA graph benchmarks ---
    graphed_euler_results = bench_graphed_euler_step_both(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )
    times_graphed_euler_kv = graphed_euler_results["kv"]
    times_graphed_euler_ring = graphed_euler_results["ring"]

    times_ungraphed_euler = bench_ungraphed_euler_step(
        model, ctx_latents, ctx_actions,
        args.warmup, args.repeats,
        n_candidates=args.n_candidates,
        num_ode_steps=args.cem_ode_steps,
    )

    graphed_euler_bs1_results = bench_graphed_euler_bs1_both(
        model, ctx_latents, ctx_actions, action,
        args.warmup, args.repeats,
        num_steps=args.num_steps,
    )
    times_graphed_euler_bs1_kv = graphed_euler_bs1_results["kv"]
    times_graphed_euler_bs1_ring = graphed_euler_bs1_results["ring"]

    mean_graphed_euler_kv   = sum(times_graphed_euler_kv)      / len(times_graphed_euler_kv)
    mean_graphed_euler_ring = sum(times_graphed_euler_ring)    / len(times_graphed_euler_ring)
    mean_ungraphed_euler_N   = sum(times_ungraphed_euler)    / len(times_ungraphed_euler)
    mean_graphed_euler_bs1_kv   = sum(times_graphed_euler_bs1_kv)  / len(times_graphed_euler_bs1_kv)
    mean_graphed_euler_bs1_ring = sum(times_graphed_euler_bs1_ring) / len(times_graphed_euler_bs1_ring)
    euler_N_graph_speedup_kv    = mean_ungraphed_euler_N / mean_graphed_euler_kv
    euler_N_graph_speedup_ring  = mean_ungraphed_euler_N / mean_graphed_euler_ring
    euler_bs1_graph_speedup_kv  = mean_euler_cached      / mean_graphed_euler_bs1_kv
    euler_bs1_graph_speedup_ring = mean_euler_cached_ring / mean_graphed_euler_bs1_ring
    graphed_euler_n_kv_vs_ring = mean_graphed_euler_ring / mean_graphed_euler_kv
    graphed_euler_bs1_kv_vs_ring = mean_graphed_euler_bs1_ring / mean_graphed_euler_bs1_kv

    n_euler_evals_N = args.cem_ode_steps

    print()
    print("=" * W)
    print(f"  CUDA Graph vs Eager  (Euler step, N={args.n_candidates}, ode={args.cem_ode_steps})")
    print(f"{'Metric':<35} {'Graph KV':>12} {'Graph Ring':>12} {'Eager':>12}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_graphed_euler_kv:>12.2f} {mean_graphed_euler_ring:>12.2f} {mean_ungraphed_euler_N:>12.2f}")
    print(f"{'ms/model_eval':<35} {mean_graphed_euler_kv/n_euler_evals_N:>12.2f} {mean_graphed_euler_ring/n_euler_evals_N:>12.2f} {mean_ungraphed_euler_N/n_euler_evals_N:>12.2f}")
    print(f"{'Speedup vs eager':<35} {euler_N_graph_speedup_kv:>12.2f}x {euler_N_graph_speedup_ring:>12.2f}x")
    print(f"{'Ring/KV ratio':<35} {graphed_euler_n_kv_vs_ring:>12.2f}x")
    print("=" * W)
    print(f"  CUDA Graph vs Eager  (Euler BS=1, steps={args.num_steps}, {n_euler_evals_bs1} evals)")
    print(f"{'Metric':<35} {'Graph KV':>12} {'Graph Ring':>12} {'Eager KV':>12}")
    print("-" * W)
    print(f"{'Total (ms)':<35} {mean_graphed_euler_bs1_kv:>12.2f} {mean_graphed_euler_bs1_ring:>12.2f} {mean_euler_cached:>12.2f}")
    print(f"{'ms/model_eval':<35} {mean_graphed_euler_bs1_kv/n_euler_evals_bs1:>12.2f} {mean_graphed_euler_bs1_ring/n_euler_evals_bs1:>12.2f} {mean_euler_cached/n_euler_evals_bs1:>12.2f}")
    print(f"{'Speedup vs eager KV':<35} {euler_bs1_graph_speedup_kv:>12.2f}x {mean_euler_cached_ring/mean_graphed_euler_bs1_ring:>12.2f}x")
    print(f"{'Ring/KV ratio':<35} {graphed_euler_bs1_kv_vs_ring:>12.2f}x")
    print("=" * W)

    print()
    print(f"### CUDA Graph Speedup")
    print(f"| Solver | Total (ms) | ms/eval | Speedup vs eager |")
    print(f"|--------|-----------|---------|-----------------|")
    print(f"| Euler graphed KV (N={args.n_candidates})    | {mean_graphed_euler_kv:.2f} | "
          f"{mean_graphed_euler_kv/n_euler_evals_N:.2f} | **{euler_N_graph_speedup_kv:.2f}x** |")
    print(f"| Euler graphed Ring (N={args.n_candidates})  | {mean_graphed_euler_ring:.2f} | "
          f"{mean_graphed_euler_ring/n_euler_evals_N:.2f} | **{euler_N_graph_speedup_ring:.2f}x** |")
    print(f"| Euler eager   (N={args.n_candidates})    | {mean_ungraphed_euler_N:.2f} | "
          f"{mean_ungraphed_euler_N/n_euler_evals_N:.2f} | 1.00x |")
    print(f"| Euler graphed KV (BS=1)           | {mean_graphed_euler_bs1_kv:.2f} | "
          f"{mean_graphed_euler_bs1_kv/n_euler_evals_bs1:.2f} | **{euler_bs1_graph_speedup_kv:.2f}x** |")
    print(f"| Euler graphed Ring (BS=1)         | {mean_graphed_euler_bs1_ring:.2f} | "
          f"{mean_graphed_euler_bs1_ring/n_euler_evals_bs1:.2f} | **{mean_euler_cached_ring/mean_graphed_euler_bs1_ring:.2f}x** |")
    print(f"| Euler KV-cached eager (BS=1)   | {mean_euler_cached:.2f} | "
          f"{mean_euler_cached/n_euler_evals_bs1:.2f} | 1.00x |")

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
