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
ACTION_DIM = 7  # [dx, dy, dz, gripper, ee_x, ee_y, ee_z]
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
# CubePosHead — auxiliary head for cube position prediction
# ---------------------------------------------------------------------------
class CubePosHead(nn.Module):
    """Predicts cube XYZ from mean-pooled DiT token representation.

    Used as an auxiliary loss during training to force the latent representation
    to geometrically ground object location. Not used at inference time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 3),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: [B, N, D] final DiT block output (before FinalLayer)
        Returns:
            [B, 3] predicted cube XYZ
        """
        with record_function("CubePosHead"):
            return self.mlp(tokens.mean(dim=1))


# ---------------------------------------------------------------------------
# Task heads — all share the same MLP structure as CubePosHead.
# They operate on denoise_tokens only (the predicted next-frame region),
# NOT the full context+denoise sequence, because they predict properties
# of the next state rather than using all historical context.
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Predicts scalar reward r_t from the next-frame token representation.

    Trained against env rewards stored in the replay buffer.
    Used by the CEM planner to score candidate trajectories.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: [B, N, D] denoise region tokens (x[:, -NUM_PATCHES:, :])
        Returns:
            [B, 1] predicted scalar reward
        """
        with record_function("RewardHead"):
            return self.mlp(tokens.mean(dim=1))


class DoneHead(nn.Module):
    """Predicts episode termination logit from the next-frame token representation.

    Returns raw logits (not sigmoid). Apply torch.sigmoid() for probability,
    or use F.binary_cross_entropy_with_logits() for training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        # Bias toward not-done at init: most steps are not terminal
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: [B, N, D] denoise region tokens
        Returns:
            [B, 1] termination logit (sigmoid → done probability)
        """
        with record_function("DoneHead"):
            return self.mlp(tokens.mean(dim=1))


class ValueHead(nn.Module):
    """Predicts state value V(z_t) — expected discounted return from z_t.

    Trained against Monte-Carlo or TD returns from the replay buffer.
    Used by the CEM planner to bootstrap long-horizon returns:
        score = Σ γ^k * r̂_{t+k}  +  γ^H * V̂(z_{t+H})
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: [B, N, D] denoise region tokens
        Returns:
            [B, 1] predicted state value V(z_t)
        """
        with record_function("ValueHead"):
            return self.mlp(tokens.mean(dim=1))


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

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Unified forward: standard self-attention when cache is None,
        cached cross-attention (denoise Q → full K/V) when cache is provided."""
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

            if cache is not None:
                cache.update(layer_idx, k, v)
                k, v = cache.get_kv(layer_idx)

            attn = F.scaled_dot_product_attention(q, k, v)  # [B, 6, N_kv, 64]
            attn = attn.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
            attn = self.proj(attn)

            x = x + gate1.unsqueeze(1) * attn

            # --- MLP branch ---
            x_norm2 = modulate(self.norm2(x), shift2, scale2)
            x = x + gate2.unsqueeze(1) * self.mlp(x_norm2)

            return x

    def forward_prefill(
        self, x_ctx: torch.Tensor, c_ctx: torch.Tensor, cache: KVCache, layer_idx: int
    ) -> torch.Tensor:
        """Prefill: self-attention among context tokens only, caches K/V."""
        with record_function("DiTBlock.forward_prefill"):
            B, N, D = x_ctx.shape

            mod = self.adaLN_modulation(c_ctx)
            shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

            x_norm = modulate(self.norm1(x_ctx), shift1, scale1)
            qkv = self.qkv(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)

            cache.prefill(layer_idx, k, v)

            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, N, D)
            attn = self.proj(attn)
            x_ctx = x_ctx + gate1.unsqueeze(1) * attn

            x_norm2 = modulate(self.norm2(x_ctx), shift2, scale2)
            x_ctx = x_ctx + gate2.unsqueeze(1) * self.mlp(x_norm2)
            return x_ctx


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
        self.cube_pos_head = CubePosHead()
        self.reward_head = RewardHead()
        self.done_head   = DoneHead()
        self.value_head  = ValueHead()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        cache: KVCache | None = None,
        ctx_latents: torch.Tensor | None = None,
        return_aux: bool = False,
        return_heads: bool = False,
    ) -> torch.Tensor:
        """Unified forward pass.

        Training (cache=None, ctx_latents provided):
            Context tokens are prepended to the denoise tokens and the full
            sequence is run through standard self-attention.  This matches the
            computation done by the KV cache at inference.

        Inference (cache provided):
            Context K/V has been prefilled once via prefill_cache().  Denoise
            tokens attend to the full [context | denoise] KV in the cache.

        No context (cache=None, ctx_latents=None):
            Standard self-attention over the 16 denoise tokens only.

        Args:
            ctx_latents:  [B, n_ctx, 16, 8, 8] context frames (training only).
            return_aux:   If True, returns (velocity, cube_pos_pred).
            return_heads: If True, returns (velocity, reward [B,1], done_logit [B,1],
                          value [B,1]).  Heads run on denoise_tokens only.
                          For head training pass a clean latent (t=1) so the
                          token representations reflect the actual next state.
        """
        with record_function("DiTSmall"):
            # --- Embed denoise frame ---
            x = self.patch_embed(x) + self.pos_embed  # [B, 16, D]

            if cache is not None:
                # Inference: denoise occupies the last temporal slot
                x = x + self.frame_pos_embed[:, -1, :, :]

            elif ctx_latents is not None:
                # Training: embed each context frame and prepend to denoise tokens
                B, n_ctx, C, H, W = ctx_latents.shape
                ctx_list = []
                for i in range(n_ctx):
                    tok = self.patch_embed(ctx_latents[:, i]) + self.pos_embed
                    tok = tok + self.frame_pos_embed[:, i, :, :]
                    ctx_list.append(tok)
                # Denoise frame uses the last temporal slot (matches inference)
                x = x + self.frame_pos_embed[:, -1, :, :]
                # [B, n_ctx*16 + 16, D]
                x = torch.cat(ctx_list + [x], dim=1)

            c = self.t_embed(t) + self.action_embed(action)  # [B, D]

            for i, block in enumerate(self.blocks):
                x = block(x, c, cache=cache, layer_idx=i)

            # Always use the last NUM_PATCHES tokens for output — these are the
            # denoise tokens regardless of how many context tokens were prepended
            denoise_tokens = x[:, -NUM_PATCHES:, :]   # [B, 16, D]
            v = self._unpatchify(self.final_layer(denoise_tokens, c))

            if return_aux:
                # cube_pos_head uses ALL tokens — context carries object location
                return v, self.cube_pos_head(x)

            if return_heads:
                # Task heads use denoise_tokens only — they predict properties of
                # the NEXT state, not properties aggregated over history
                reward = self.reward_head(denoise_tokens)   # [B, 1]
                done   = self.done_head(denoise_tokens)     # [B, 1] logit
                value  = self.value_head(denoise_tokens)    # [B, 1]
                return v, reward, done, value

            return v

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

    # forward_cached() is now unified into forward(cache=...).
    # Kept as a thin alias for backward compatibility.
    def forward_cached(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return self.forward(x, t, action, cache=cache)
