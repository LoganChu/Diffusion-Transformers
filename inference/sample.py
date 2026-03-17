import torch
from torch.profiler import record_function

from models.cache import KVCache
from models.dit import DEPTH, HEAD_DIM, NUM_HEADS, NUM_PATCHES, IN_CHANNELS, LATENT_H, LATENT_W


@torch.no_grad()
def sample_ode_cached(
    model,
    ctx_latents: torch.Tensor,
    ctx_actions: torch.Tensor,
    action: torch.Tensor,
    num_steps: int = 8,
) -> torch.Tensor:
    """Cached ODE sampler with zero-allocation inner loop.

    Args:
        model: DiTSmall instance (should be in eval mode, float16)
        ctx_latents: [B, n_ctx_frames, 16, 8, 8] context frames
        ctx_actions: [B, 8] most recent context action
        action: [B, 8] action for the predicted frame
        num_steps: number of Euler ODE steps
    Returns:
        [B, 16, 8, 8] predicted latent frame
    """
    with record_function("sample_ode_cached"):
        B = ctx_latents.shape[0]
        n_ctx_tokens = ctx_latents.shape[1] * NUM_PATCHES
        device = action.device
        dtype = next(model.parameters()).dtype

        # 1. One-time cache allocation
        cache = KVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES,
            device=device, dtype=dtype,
        )

        # 2. Prefill context K/V (once, outside loop)
        model.prefill_cache(ctx_latents, ctx_actions, cache)

        # 3. ODE loop — zero allocation
        x = torch.randn(
            B, IN_CHANNELS, LATENT_H, LATENT_W,
            device=device, dtype=dtype,
        )
        dt = 1.0 / num_steps
        t_buf = torch.empty(B, device=device, dtype=dtype)

        for i in range(num_steps):
            t_buf.fill_(i * dt)
            v = model(x, t_buf, action, cache=cache)
            x.add_(v, alpha=dt)  # inplace: x += dt * v

        return x
