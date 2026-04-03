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

    def slide(self, layer_idx: int, k_new: Tensor, v_new: Tensor) -> None:
        """Evict the oldest context frame and append a new one. Zero-allocation.

        Shifts the context region left by one frame slot ([1:n_ctx] → [0:n_ctx-1]),
        then writes k_new/v_new into the last context slot. Keeps the total
        sequence length fixed so CUDA graph shapes stay static.

        Call this instead of prefill() after the initial warmup when the rolling
        context window has been fully filled and must evict the oldest frame.

        Args:
            layer_idx: Which transformer layer's cache to update.
            k_new: [B, num_heads, n_frame_tokens, head_dim] newest frame K.
            v_new: [B, num_heads, n_frame_tokens, head_dim] newest frame V.
        """
        with record_function("KVCache.slide"):
            n_frame = k_new.shape[-2]   # tokens per frame (NUM_PATCHES = 16)
            # Shift context left by one frame: drop slot 0, open slot at end
            self._k[layer_idx, :, :, :self.n_ctx - n_frame, :].copy_(
                self._k[layer_idx, :, :, n_frame:self.n_ctx, :]
            )
            self._v[layer_idx, :, :, :self.n_ctx - n_frame, :].copy_(
                self._v[layer_idx, :, :, n_frame:self.n_ctx, :]
            )
            # Write newest frame into the last context slot
            self._k[layer_idx, :, :, self.n_ctx - n_frame:self.n_ctx, :].copy_(k_new)
            self._v[layer_idx, :, :, self.n_ctx - n_frame:self.n_ctx, :].copy_(v_new)

    def reset(self) -> None:
        """Zero buffers for CUDA Graph replay reuse."""
        self._k.zero_()
        self._v.zero_()
