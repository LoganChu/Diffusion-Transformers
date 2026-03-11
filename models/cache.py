import torch
from torch import Tensor
from torch.profiler import record_function


class KVCache:
    """Pre-allocated KV cache for DiT inference.

    Separates "Static Prefix" (context frames, written once at prefill)
    from "Iterative Latents" (denoise tokens, updated each ODE step).
    All operations are zero-allocation: slice writes via .copy_() and
    views via indexing. CUDA Graph safe (static shapes, no .item()).
    """

    def __init__(
        self,
        depth: int,
        num_heads: int,
        head_dim: int,
        n_ctx_tokens: int,
        n_denoise_tokens: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        n_total = n_ctx_tokens + n_denoise_tokens
        self.n_ctx = n_ctx_tokens
        self.n_total = n_total
        # [depth, B=1, num_heads, N_total, head_dim]
        self._k = torch.zeros(
            depth, 1, num_heads, n_total, head_dim, device=device, dtype=dtype
        )
        self._v = torch.zeros(
            depth, 1, num_heads, n_total, head_dim, device=device, dtype=dtype
        )

    def prefill(self, layer_idx: int, k_ctx: Tensor, v_ctx: Tensor) -> None:
        """One-time write of context K/V into the static prefix region."""
        with record_function("KVCache.prefill"):
            self._k[layer_idx, :, :, : self.n_ctx, :].copy_(k_ctx)
            self._v[layer_idx, :, :, : self.n_ctx, :].copy_(v_ctx)

    def update(self, layer_idx: int, k_den: Tensor, v_den: Tensor) -> None:
        """Per-ODE-step write of denoise K/V into the iterative region."""
        with record_function("KVCache.update"):
            self._k[layer_idx, :, :, self.n_ctx :, :].copy_(k_den)
            self._v[layer_idx, :, :, self.n_ctx :, :].copy_(v_den)

    def get_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return views (not copies) of the full K/V for a layer."""
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        """Zero buffers for CUDA Graph replay reuse."""
        self._k.zero_()
        self._v.zero_()
