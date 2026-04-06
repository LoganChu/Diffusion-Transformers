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


class RingKVCache:
    """KV cache with O(1) context-frame eviction via a circular head pointer.

    Drop-in replacement for KVCache for prefill / update / get_kv calls.
    Adds slide_ring() which replaces the O(n_ctx) memory copy in KVCache.slide()
    with a single O(n_frame) write plus a pointer advance.

    Correctness note: DiT context attention uses full attention (no causal mask),
    so SDPA is permutation-invariant over K/V rows. The ring buffer's physical
    layout need not match logical frame order — get_kv() returns the raw buffer
    directly with no reordering required.

    Buffer layout mirrors KVCache:
        _k / _v: [depth, 1, num_heads, n_ctx + n_denoise, head_dim]
        positions [0 : n_ctx]       — ring buffer for context frames
        positions [n_ctx : n_total] — denoise tokens (unchanged per ODE step)
    """

    def __init__(
        self,
        depth: int,
        num_heads: int,
        head_dim: int,
        n_ctx_tokens: int,
        n_denoise_tokens: int,
        n_frame_tokens: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        n_total = n_ctx_tokens + n_denoise_tokens
        self.n_ctx    = n_ctx_tokens
        self.n_total  = n_total
        self.n_frame  = n_frame_tokens
        self.n_frames = n_ctx_tokens // n_frame_tokens  # context frame slots
        self.head     = 0                               # next write slot (frame index)
        # [depth, B=1, num_heads, N_total, head_dim]
        self._k = torch.zeros(
            depth, 1, num_heads, n_total, head_dim, device=device, dtype=dtype
        )
        self._v = torch.zeros(
            depth, 1, num_heads, n_total, head_dim, device=device, dtype=dtype
        )

    def prefill(self, layer_idx: int, k_ctx: Tensor, v_ctx: Tensor) -> None:
        """One-time write of context K/V into the static prefix region."""
        with record_function("RingKVCache.prefill"):
            self._k[layer_idx, :, :, :self.n_ctx, :].copy_(k_ctx)
            self._v[layer_idx, :, :, :self.n_ctx, :].copy_(v_ctx)

    def update(self, layer_idx: int, k_den: Tensor, v_den: Tensor) -> None:
        """Per-ODE-step write of denoise K/V into the iterative region."""
        with record_function("RingKVCache.update"):
            self._k[layer_idx, :, :, self.n_ctx:, :].copy_(k_den)
            self._v[layer_idx, :, :, self.n_ctx:, :].copy_(v_den)

    def get_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return views (not copies) of the full K/V for a layer.

        Context tokens are in ring-buffer physical order, not logical
        chronological order. This is correct for full (non-causal) attention.
        """
        return self._k[layer_idx], self._v[layer_idx]

    def slide_ring(self, layer_idx: int, k_new: Tensor, v_new: Tensor) -> None:
        """Write new frame K/V at the current head slot, evicting the oldest frame.

        O(n_frame) write — no memory shifting. After updating all layers,
        call advance_head() once to move the ring pointer forward.

        Args:
            layer_idx: Which transformer layer's cache to update.
            k_new: [B, num_heads, n_frame_tokens, head_dim] newest frame K.
            v_new: [B, num_heads, n_frame_tokens, head_dim] newest frame V.
        """
        with record_function("RingKVCache.slide_ring"):
            slot_start = self.head * self.n_frame
            slot_end   = slot_start + self.n_frame
            self._k[layer_idx, :, :, slot_start:slot_end, :].copy_(k_new)
            self._v[layer_idx, :, :, slot_start:slot_end, :].copy_(v_new)

    def advance_head(self) -> None:
        """Advance the ring pointer by one frame slot.

        Call once after all DEPTH layers have been updated via slide_ring().
        """
        self.head = (self.head + 1) % self.n_frames

    def reset(self) -> None:
        """Zero buffers and reset ring pointer for reuse."""
        self._k.zero_()
        self._v.zero_()
        self.head = 0
