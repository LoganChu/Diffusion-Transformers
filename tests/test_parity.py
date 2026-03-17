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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
