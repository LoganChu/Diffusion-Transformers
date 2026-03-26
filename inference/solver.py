"""Optimized Heun ODE solver with KV-cache reuse.

Heun's method (improved Euler) achieves 2nd-order accuracy:
    x_{i+1} = x_i + dt/2 * (v1 + v2)
where v1 = f(x_i, t_i) and v2 = f(x_i + dt*v1, t_i + dt).

The KV cache amortises the cost of context-frame attention: prefill once,
then reuse the static-prefix K/V across all ODE steps.  Only the denoise
region of the cache is overwritten per evaluation.

All allocations happen before the loop (zero-alloc inner loop per CLAUDE.md).
"""

from __future__ import annotations

import torch
from torch.profiler import record_function

from models.cache import KVCache
from models.dit import DEPTH, HEAD_DIM, IN_CHANNELS, LATENT_H, LATENT_W, NUM_HEADS, NUM_PATCHES


@torch.no_grad()
def sample_heun_cached(
    model,
    ctx_latents: torch.Tensor,
    ctx_actions: torch.Tensor,
    action: torch.Tensor,
    num_steps: int = 8,
) -> torch.Tensor:
    """Heun's method (2nd-order) with KV-cache context reuse.

    Args:
        model: DiTSmall instance (eval mode, float16).
        ctx_latents: [B, n_ctx_frames, 16, 8, 8] context frames.
        ctx_actions: [B, 7] most recent context cond vector [dx,dy,dz,gripper,ee_x,ee_y,ee_z].
        action: [B, 7] cond vector for the predicted frame.
        num_steps: number of ODE integration steps.
    Returns:
        [B, 16, 8, 8] predicted latent frame.
    """
    with record_function("sample_heun_cached"):
        B = ctx_latents.shape[0]
        n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
        device = action.device
        dtype = next(model.parameters()).dtype

        # --- One-time allocations ---
        cache = KVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES,
            device=device, dtype=dtype,
        )
        model.prefill_cache(ctx_latents, ctx_actions, cache)

        x = torch.randn(
            B, IN_CHANNELS, LATENT_H, LATENT_W,
            device=device, dtype=dtype,
        )
        dt = 1.0 / num_steps
        t_buf = torch.empty(B, device=device, dtype=dtype)
        # Pre-allocate scratch buffer for the Euler predictor
        x_euler = torch.empty_like(x)

        # --- ODE loop (zero-allocation) ---
        for i in range(num_steps):
            with record_function(f"heun_step_{i}"):
                t_buf.fill_(i * dt)
                v1 = model(x, t_buf, action, cache=cache)

                if i < num_steps - 1:
                    # Euler predictor: x_euler = x + dt * v1
                    torch.add(x, v1, alpha=dt, out=x_euler)

                    t_buf.fill_((i + 1) * dt)
                    v2 = model(x_euler, t_buf, action, cache=cache)

                    # Heun corrector: x += dt/2 * (v1 + v2)
                    v1.add_(v2)  # v1 is scratch after this
                    x.add_(v1, alpha=dt * 0.5)
                else:
                    # Last step: plain Euler (no lookahead available)
                    x.add_(v1, alpha=dt)

        return x
