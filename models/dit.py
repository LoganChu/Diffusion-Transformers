from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

if TYPE_CHECKING:
    from .cache import KVCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IN_CHANNELS = 16
PATCH_SIZE = 2
HIDDEN_DIM = 384
NUM_HEADS = 6
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
DEPTH = 12
MLP_RATIO = 4.0
ACTION_DIM = 8
LATENT_H = 8
LATENT_W = 8
NUM_PATCHES = (LATENT_H // PATCH_SIZE) * (LATENT_W // PATCH_SIZE)  # 16
PATCH_DIM = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64
MAX_CTX_FRAMES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(IN_CHANNELS, HIDDEN_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("PatchEmbed"):
            # [B, 16, 8, 8] -> [B, 384, 4, 4] -> [B, 16, 384]
            x = self.proj(x)
            B, C, H, W = x.shape
            return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, freq_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        # Pre-compute frequency table as buffer (compile-safe, no recomputation)
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)  # [128]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        with record_function("TimestepEmbedder"):
            # Keep sinusoidal computation in float32 for precision under autocast
            with torch.amp.autocast("cuda", enabled=False):
                t_f32 = t.float().unsqueeze(1)  # [B, 1]
                args = t_f32 * self.freqs.unsqueeze(0)  # [B, 128]
                emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 256]
            return self.mlp(emb)  # [B, 384]


# ---------------------------------------------------------------------------
# ActionEmbedder
# ---------------------------------------------------------------------------
class ActionEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ACTION_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        with record_function("ActionEmbedder"):
            return self.mlp(action)  # [B, 8] -> [B, 384]


# ---------------------------------------------------------------------------
# DiTBlock — adaLN-Zero transformer block
# ---------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mlp_hidden = int(HIDDEN_DIM * MLP_RATIO)

        self.norm1 = nn.LayerNorm(HIDDEN_DIM, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM, elementwise_affine=False)

        self.qkv = nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM)
        self.proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, HIDDEN_DIM),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 6 * HIDDEN_DIM),
        )

        # Zero-init: each block starts as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with record_function("DiTBlock"):
            B, N, D = x.shape

            # adaLN modulation
            mod = self.adaLN_modulation(c)  # [B, 6*384]
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

            # --- Attention branch ---
            x_norm = modulate(self.norm1(x), shift1, scale1)  # [B, N, D]
            qkv = self.qkv(x_norm)  # [B, N, 3*D]
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B, 6, N, 64]
            k = k.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q, k, v)  # [B, 6, N, 64]
            attn = attn.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
            attn = self.proj(attn)

            x = x + gate1.unsqueeze(1) * attn

            # --- MLP branch ---
            x_norm2 = modulate(self.norm2(x), shift2, scale2)
            x = x + gate2.unsqueeze(1) * self.mlp(x_norm2)

            return x

    def _adaln_qkv(self, x: torch.Tensor, c: torch.Tensor):
        """Shared adaLN + QKV projection for prefill/cached paths."""
        B, N, D = x.shape
        mod = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift1, scale1)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        return q, k, v, shift1, scale1, gate1, shift2, scale2, gate2

    def forward_prefill(
        self, x_ctx: torch.Tensor, c_ctx: torch.Tensor, cache: KVCache, layer_idx: int
    ) -> torch.Tensor:
        """Prefill: self-attention among context tokens only, caches K/V."""
        with record_function("DiTBlock.forward_prefill"):
            B, N, D = x_ctx.shape
            q, k, v, _, _, gate1, shift2, scale2, gate2 = self._adaln_qkv(x_ctx, c_ctx)

            cache.prefill(layer_idx, k, v)

            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, N, D)
            attn = self.proj(attn)
            x_ctx = x_ctx + gate1.unsqueeze(1) * attn

            x_norm2 = modulate(self.norm2(x_ctx), shift2, scale2)
            x_ctx = x_ctx + gate2.unsqueeze(1) * self.mlp(x_norm2)
            return x_ctx

    def forward_cached(
        self, x_den: torch.Tensor, c_den: torch.Tensor, cache: KVCache, layer_idx: int
    ) -> torch.Tensor:
        """Cached: Q from denoise tokens attends to full cached K/V."""
        with record_function("DiTBlock.forward_cached"):
            B, N, D = x_den.shape
            q, k, v, _, _, gate1, shift2, scale2, gate2 = self._adaln_qkv(x_den, c_den)

            cache.update(layer_idx, k, v)
            k_full, v_full = cache.get_kv(layer_idx)

            attn = F.scaled_dot_product_attention(q, k_full, v_full)
            attn = attn.transpose(1, 2).reshape(B, N, D)
            attn = self.proj(attn)
            x_den = x_den + gate1.unsqueeze(1) * attn

            x_norm2 = modulate(self.norm2(x_den), shift2, scale2)
            x_den = x_den + gate2.unsqueeze(1) * self.mlp(x_norm2)
            return x_den


# ---------------------------------------------------------------------------
# FinalLayer — adaLN + output projection
# ---------------------------------------------------------------------------
class FinalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_DIM, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 2 * HIDDEN_DIM),
        )
        self.linear = nn.Linear(HIDDEN_DIM, PATCH_DIM)

        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with record_function("FinalLayer"):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
            return self.linear(x)  # [B, 16, 64]


# ---------------------------------------------------------------------------
# DiTSmall — top-level model
# ---------------------------------------------------------------------------
class DiTSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, HIDDEN_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Per-frame temporal identity: [1, MAX_CTX_FRAMES+1, 1, HIDDEN_DIM]
        # Last slot is for the denoise frame.
        self.frame_pos_embed = nn.Parameter(
            torch.zeros(1, MAX_CTX_FRAMES + 1, 1, HIDDEN_DIM)
        )

        self.t_embed = TimestepEmbedder()
        self.action_embed = ActionEmbedder()

        self.blocks = nn.ModuleList([DiTBlock() for _ in range(DEPTH)])
        self.final_layer = FinalLayer()

    def forward(self, x: torch.Tensor, t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with record_function("DiTSmall"):
            B = x.shape[0]
            pH = LATENT_H // PATCH_SIZE  # 4
            pW = LATENT_W // PATCH_SIZE  # 4

            x = self.patch_embed(x) + self.pos_embed  # [B, 16, 384]
            c = self.t_embed(t) + self.action_embed(action)  # [B, 384]

            for block in self.blocks:
                x = block(x, c)  # [B, 16, 384]

            x = self.final_layer(x, c)  # [B, 16, 64]

            # Unpatchify: [B, 16, 64] -> [B, 16, 8, 8]
            x = x.view(B, pH, pW, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, IN_CHANNELS, LATENT_H, LATENT_W)

            return x  # predicted velocity v_pred

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        pH = LATENT_H // PATCH_SIZE
        pW = LATENT_W // PATCH_SIZE
        x = x.view(B, pH, pW, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(B, IN_CHANNELS, LATENT_H, LATENT_W)

    def prefill_cache(
        self,
        ctx_latents: torch.Tensor,
        ctx_actions: torch.Tensor,
        cache: KVCache,
    ) -> None:
        """Prefill KV cache with context frames. Call once before ODE loop.

        Args:
            ctx_latents: [B, n_ctx_frames, 16, 8, 8] historical latent frames
            ctx_actions: [B, 8] most recent context action
            cache: KVCache to fill
        """
        with record_function("DiTSmall.prefill_cache"), torch.amp.autocast("cuda", dtype=torch.float16):
            B, n_ctx, C, H, W = ctx_latents.shape

            # Patch-embed each context frame and add positional embeddings
            frame_tokens = []
            for i in range(n_ctx):
                tok = self.patch_embed(ctx_latents[:, i])  # [B, NUM_PATCHES, HIDDEN_DIM]
                tok = tok + self.pos_embed + self.frame_pos_embed[:, i, :, :]
                frame_tokens.append(tok)
            x_ctx = torch.cat(frame_tokens, dim=1)  # [B, N_ctx, HIDDEN_DIM]

            # Context conditioning: t=1.0 (fully denoised), fixed action
            t_ones = torch.ones(B, device=x_ctx.device, dtype=x_ctx.dtype)
            c_ctx = self.t_embed(t_ones) + self.action_embed(ctx_actions)

            for layer_idx, block in enumerate(self.blocks):
                x_ctx = block.forward_prefill(x_ctx, c_ctx, cache, layer_idx)
            # x_ctx discarded — we only needed K/V side-effects

    def forward_cached(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        """Forward pass for denoise tokens using cached context K/V.

        Args:
            x: [B, 16, 8, 8] noisy latent (denoise target)
            t: [B] timestep
            action: [B, 8] action conditioning
            cache: KVCache with prefilled context K/V
        Returns:
            [B, 16, 8, 8] predicted velocity
        """
        with record_function("DiTSmall.forward_cached"), torch.amp.autocast("cuda", dtype=torch.float16):
            # Denoise token = last frame slot
            x = self.patch_embed(x) + self.pos_embed + self.frame_pos_embed[:, -1, :, :]
            c = self.t_embed(t) + self.action_embed(action)

            for layer_idx, block in enumerate(self.blocks):
                x = block.forward_cached(x, c, cache, layer_idx)

            x = self.final_layer(x, c)
            return self._unpatchify(x)
