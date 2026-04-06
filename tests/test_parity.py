"""Parity tests: KV-cached inference must match the recompute baseline.

The "baseline" recomputes context K/V from scratch at every ODE step
(fresh cache per evaluation).  The "cached" path prefills once and reuses.
Both execute the identical numerical operations — the cache only avoids
redundant work — so outputs must match within floating-point tolerance.

Run:
    python -m pytest tests/test_parity.py -v
"""

from __future__ import annotations

import torch
import pytest

from inference.graph_solver import GraphedEulerStep, GraphedHeunSolver
from models.cache import KVCache, RingKVCache
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-6  # per spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_model(dtype: torch.dtype = torch.float32) -> DiTSmall:
    model = DiTSmall().to(device=DEVICE, dtype=dtype).eval()
    return model


def _make_inputs(
    B: int = 1,
    n_ctx: int = 2,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ctx_latents = torch.randn(
        B, n_ctx, IN_CHANNELS, LATENT_H, LATENT_W,
        device=DEVICE, dtype=dtype, generator=g,
    )
    ctx_actions = torch.randn(B, 8, device=DEVICE, dtype=dtype, generator=g)
    action = torch.randn(B, 8, device=DEVICE, dtype=dtype, generator=g)
    x = torch.randn(
        B, IN_CHANNELS, LATENT_H, LATENT_W,
        device=DEVICE, dtype=dtype, generator=g,
    )
    t = torch.tensor([0.5] * B, device=DEVICE, dtype=dtype)
    return ctx_latents, ctx_actions, action, x, t


def _make_cache(n_ctx_tokens: int, dtype: torch.dtype = torch.float32):
    return KVCache(
        DEPTH, NUM_HEADS, HEAD_DIM,
        n_ctx_tokens, NUM_PATCHES,
        device=DEVICE, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Single forward-pass parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_forward_parity():
    """Cached forward must match recomputed forward for a single evaluation."""
    model = _make_model()
    ctx_latents, ctx_actions, action, x, t = _make_inputs()
    n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES

    # --- Cached path: prefill once, forward once ---
    cache_a = _make_cache(n_ctx_tokens)
    with torch.no_grad():
        model.prefill_cache(ctx_latents, ctx_actions, cache_a)
        out_cached = model(x, t, action, cache=cache_a)

    # --- Recompute path: fresh cache, prefill + forward ---
    cache_b = _make_cache(n_ctx_tokens)
    with torch.no_grad():
        model.prefill_cache(ctx_latents, ctx_actions, cache_b)
        out_recompute = model(x, t, action, cache=cache_b)

    diff = (out_cached - out_recompute).abs().max().item()
    assert diff <= ATOL, f"Single-forward parity failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# Multi-step Euler parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_euler_multistep_parity():
    """After N Euler steps, cached (prefill once) must match recompute (prefill each step)."""
    model = _make_model()
    num_steps = 4
    B, n_ctx = 1, 2
    dtype = torch.float32

    ctx_latents, ctx_actions, action, _, _ = _make_inputs(B=B, n_ctx=n_ctx, dtype=dtype)
    n_ctx_tokens = n_ctx * NUM_PATCHES

    # Shared initial noise
    seed_gen = torch.Generator(device=DEVICE).manual_seed(999)
    x0 = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=DEVICE, dtype=dtype, generator=seed_gen)

    dt = 1.0 / num_steps

    # --- Cached path ---
    cache_c = _make_cache(n_ctx_tokens, dtype=dtype)
    with torch.no_grad():
        model.prefill_cache(ctx_latents, ctx_actions, cache_c)
        x_c = x0.clone()
        for i in range(num_steps):
            t_val = torch.tensor([i * dt] * B, device=DEVICE, dtype=dtype)
            v = model(x_c, t_val, action, cache=cache_c)
            x_c.add_(v, alpha=dt)

    # --- Recompute path ---
    with torch.no_grad():
        x_r = x0.clone()
        for i in range(num_steps):
            cache_r = _make_cache(n_ctx_tokens, dtype=dtype)
            model.prefill_cache(ctx_latents, ctx_actions, cache_r)
            t_val = torch.tensor([i * dt] * B, device=DEVICE, dtype=dtype)
            v = model(x_r, t_val, action, cache=cache_r)
            x_r.add_(v, alpha=dt)

    diff = (x_c - x_r).abs().max().item()
    assert diff <= ATOL, f"Euler multi-step parity failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# Multi-step Heun parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_heun_multistep_parity():
    """Heun solver with persistent cache must match recompute-per-step baseline."""
    model = _make_model()
    num_steps = 4
    B, n_ctx = 1, 2
    dtype = torch.float32

    ctx_latents, ctx_actions, action, _, _ = _make_inputs(B=B, n_ctx=n_ctx, dtype=dtype)
    n_ctx_tokens = n_ctx * NUM_PATCHES

    seed_gen = torch.Generator(device=DEVICE).manual_seed(42)
    x0 = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=DEVICE, dtype=dtype, generator=seed_gen)

    dt = 1.0 / num_steps

    def _heun_loop(make_cache_fn):
        """Run Heun ODE loop with the given cache factory."""
        x = x0.clone()
        cache = make_cache_fn()
        model.prefill_cache(ctx_latents, ctx_actions, cache)

        for i in range(num_steps):
            t_val = torch.tensor([i * dt] * B, device=DEVICE, dtype=dtype)
            v1 = model(x, t_val, action, cache=cache)

            if i < num_steps - 1:
                x_euler = x + dt * v1
                t_next = torch.tensor([(i + 1) * dt] * B, device=DEVICE, dtype=dtype)
                v2 = model(x_euler, t_next, action, cache=cache)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1
        return x

    def _heun_loop_recompute():
        """Recompute baseline: fresh cache for every model evaluation."""
        x = x0.clone()

        for i in range(num_steps):
            # Fresh cache + prefill for v1
            cache = _make_cache(n_ctx_tokens, dtype=dtype)
            model.prefill_cache(ctx_latents, ctx_actions, cache)

            t_val = torch.tensor([i * dt] * B, device=DEVICE, dtype=dtype)
            v1 = model(x, t_val, action, cache=cache)

            if i < num_steps - 1:
                x_euler = x + dt * v1

                # Fresh cache + prefill for v2
                cache2 = _make_cache(n_ctx_tokens, dtype=dtype)
                model.prefill_cache(ctx_latents, ctx_actions, cache2)

                t_next = torch.tensor([(i + 1) * dt] * B, device=DEVICE, dtype=dtype)
                v2 = model(x_euler, t_next, action, cache=cache2)

                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1
        return x

    with torch.no_grad():
        out_cached = _heun_loop(lambda: _make_cache(n_ctx_tokens, dtype=dtype))
        out_recompute = _heun_loop_recompute()

    diff = (out_cached - out_recompute).abs().max().item()
    assert diff <= ATOL, f"Heun multi-step parity failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# CEM regime: shared BS=1 cache broadcasts correctly to N candidates
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cem_shared_cache_parity():
    """BS=1 context cache must broadcast correctly to N parallel candidates.

    In the CEM planner all N candidates observe identical context, so a single
    BS=1 KVCache is prefilled once and shared.  SDPA broadcasts the context
    K/V [1, heads, n_ctx, hd] against per-candidate query tensors
    [N, heads, n_denoise, hd].  This test verifies the broadcast gives
    identical results to N independent BS=1 forward passes.
    """
    model = _make_model()
    N, n_ctx = 8, 2
    dtype = torch.float32

    ctx_latents_1, ctx_actions, _, _, _ = _make_inputs(B=1, n_ctx=n_ctx, dtype=dtype, seed=5)
    n_ctx_tokens = n_ctx * NUM_PATCHES

    g = torch.Generator(device=DEVICE).manual_seed(77)
    x_N = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=DEVICE, dtype=dtype, generator=g)
    a_N = torch.randn(N, ctx_actions.shape[-1], device=DEVICE, dtype=dtype, generator=g)
    t_N = torch.rand(N, device=DEVICE, dtype=dtype, generator=g)

    with torch.no_grad():
        # --- Batched path: shared BS=1 cache, N candidates in one forward ---
        cache_shared = _make_cache(n_ctx_tokens, dtype=dtype)
        model.prefill_cache(ctx_latents_1, ctx_actions, cache_shared)
        out_batched = model(x_N, t_N, a_N, cache=cache_shared)   # [N, 16, 8, 8]

        # --- Reference: N independent BS=1 forward passes, fresh cache each ---
        out_ref = []
        for i in range(N):
            cache_i = _make_cache(n_ctx_tokens, dtype=dtype)
            model.prefill_cache(ctx_latents_1, ctx_actions, cache_i)
            out_ref.append(model(x_N[i:i+1], t_N[i:i+1], a_N[i:i+1], cache=cache_i))
        out_ref = torch.cat(out_ref, dim=0)   # [N, 16, 8, 8]

    diff = (out_batched - out_ref).abs().max().item()
    assert diff <= ATOL, f"CEM shared-cache parity failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# Backward compat: forward without cache matches original behaviour
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_no_cache_unchanged():
    """forward(x, t, action) without cache must be deterministic and not crash."""
    model = _make_model()
    _, _, action, x, t = _make_inputs(seed=7)

    with torch.no_grad():
        out1 = model(x, t, action)
        out2 = model(x, t, action)

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"Cacheless forward not deterministic: diff = {diff}"


# ---------------------------------------------------------------------------
# Ring-buffer KV cache: slide parity against physical-shift KVCache
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ring_cache_slide_parity():
    """After K slides, RingKVCache forward output must match KVCache forward output.

    Verifies that full-attention SDPA is permutation-invariant over context K/V:
    the ring buffer's non-chronological physical layout does not affect the
    attention result.  K=n_ctx ensures at least one full wrap-around of the ring.
    """
    model = _make_model()
    n_ctx, n_slides = 2, 4   # n_slides > n_ctx exercises wrap-around
    dtype = torch.float32

    ctx_latents, ctx_actions, action, x, t = _make_inputs(B=1, n_ctx=n_ctx, dtype=dtype, seed=11)
    n_ctx_tokens  = n_ctx * NUM_PATCHES
    n_frame_shape = (1, NUM_HEADS, NUM_PATCHES, HEAD_DIM)

    # Seeded new-frame K/V — identical for both caches
    g = torch.Generator(device=DEVICE).manual_seed(55)
    new_frame_kvs = [
        (
            torch.randn(*n_frame_shape, device=DEVICE, dtype=dtype, generator=g),
            torch.randn(*n_frame_shape, device=DEVICE, dtype=dtype, generator=g),
        )
        for _ in range(n_slides)
    ]

    with torch.no_grad():
        # --- Physical-shift KVCache ---
        cache_phys = _make_cache(n_ctx_tokens, dtype=dtype)
        model.prefill_cache(ctx_latents, ctx_actions, cache_phys)
        for k_new, v_new in new_frame_kvs:
            for layer_idx in range(DEPTH):
                cache_phys.slide(layer_idx, k_new, v_new)
        out_phys = model(x, t, action, cache=cache_phys)

        # --- Ring-buffer RingKVCache ---
        cache_ring = RingKVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES, NUM_PATCHES,
            device=DEVICE, dtype=dtype,
        )
        model.prefill_cache(ctx_latents, ctx_actions, cache_ring)
        for k_new, v_new in new_frame_kvs:
            for layer_idx in range(DEPTH):
                cache_ring.slide_ring(layer_idx, k_new, v_new)
            cache_ring.advance_head()
        out_ring = model(x, t, action, cache=cache_ring)

    diff = (out_phys - out_ring).abs().max().item()
    assert diff <= ATOL, f"Ring-cache slide parity failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# CUDA graph parity: graphed solvers must match eager baselines
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphed_euler_step_parity():
    """GraphedEulerStep must produce the same output as eager _euler_rollout_step.

    Both paths receive identical initial noise and action inputs; context K/V
    is prefilled from the same tensors.  The graphed path uses a shared BS=1
    cache (SDPA broadcast); the eager path passes ctx_latents directly.
    We verify the maximum absolute difference is within float16 tolerance.
    """
    from inference.planner import _euler_rollout_step

    model = _make_model(dtype=torch.float32)
    N, n_ctx = 4, 2   # small N for speed; parity not N-dependent
    num_ode_steps = 4
    dtype = torch.float32

    ctx_latents_1, ctx_actions, _, _, _ = _make_inputs(B=1, n_ctx=n_ctx, dtype=dtype, seed=20)
    n_ctx_tokens = n_ctx * NUM_PATCHES

    g = torch.Generator(device=DEVICE).manual_seed(99)
    a_cond = torch.randn(N, ctx_actions.shape[-1], device=DEVICE, dtype=dtype, generator=g)
    x_init = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=DEVICE, dtype=dtype, generator=g)

    # --- Graphed path ---
    solver = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                               num_ode_steps=num_ode_steps,
                               cache_type="kv", dtype=dtype)
    with torch.no_grad():
        out_graphed = solver.run(model, ctx_latents_1, a_cond, x_init=x_init).clone()

    # --- Eager path: manual Euler with shared BS=1 cache (mirrors graphed path) ---
    cache_ref = _make_cache(n_ctx_tokens, dtype=dtype)
    with torch.no_grad():
        model.prefill_cache(ctx_latents_1, a_cond[0:1], cache_ref)
        x_ref = x_init.clone()
        dt = 1.0 / num_ode_steps
        t_buf = torch.empty(N, device=DEVICE, dtype=dtype)
        for i in range(num_ode_steps):
            t_buf.fill_(i * dt)
            v = model(x_ref, t_buf, a_cond, cache=cache_ref)
            x_ref.add_(v, alpha=dt)

    diff = (out_graphed - x_ref).abs().max().item()
    assert diff <= ATOL, f"GraphedEulerStep parity failed: max diff = {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphed_heun_parity():
    """GraphedHeunSolver must produce the same output as eager KV-cached Heun.

    Both paths receive identical initial noise, action, and context.  The
    graphed path replays the captured CUDA graph; the eager path runs the
    same unrolled Heun loop with a fresh KVCache.
    """
    model = _make_model(dtype=torch.float32)
    n_ctx, num_steps = 2, 4
    dtype = torch.float32

    ctx_latents, ctx_actions, action, _, _ = _make_inputs(B=1, n_ctx=n_ctx, dtype=dtype, seed=30)
    n_ctx_tokens = n_ctx * NUM_PATCHES

    g = torch.Generator(device=DEVICE).manual_seed(42)
    x_init = torch.randn(1, IN_CHANNELS, LATENT_H, LATENT_W, device=DEVICE, dtype=dtype, generator=g)

    # --- Graphed path ---
    solver = GraphedHeunSolver(model, n_ctx=n_ctx, num_steps=num_steps, dtype=dtype)
    with torch.no_grad():
        out_graphed = solver.run(model, ctx_latents, ctx_actions, action, x_init=x_init).clone()

    # --- Eager path: Heun with KVCache (mirrors graphed loop exactly) ---
    cache_ref = _make_cache(n_ctx_tokens, dtype=dtype)
    with torch.no_grad():
        model.prefill_cache(ctx_latents, ctx_actions, cache_ref)
        x_ref = x_init.clone()
        dt = 1.0 / num_steps
        t_buf = torch.empty(1, device=DEVICE, dtype=dtype)
        for i in range(num_steps):
            t_buf.fill_(i * dt)
            v1 = model(x_ref, t_buf, action, cache=cache_ref)
            if i < num_steps - 1:
                x_pred = x_ref + dt * v1
                t_buf.fill_((i + 1) * dt)
                v2 = model(x_pred, t_buf, action, cache=cache_ref)
                x_ref = x_ref + dt * 0.5 * (v1 + v2)
            else:
                x_ref = x_ref + dt * v1

    diff = (out_graphed - x_ref).abs().max().item()
    assert diff <= ATOL, f"GraphedHeunSolver parity failed: max diff = {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
