"""Circular replay buffer for online model-based RL.

Stores latent-space transitions collected by the CEM planner in the real
environment:

    (z_t, a_cond_t, r_t, z_{t+1}, terminated_t, truncated_t, success_t, ctx_t, mc_return_t)

Storage layout (all on CPU to avoid VRAM pressure):
    z, z_next, ctx          — float16   (VAE latents)
    a_cond, r, mc           — float32
    terminated, truncated,
    success                 — bool

Sampling copies a random batch to the target device (GPU) via non-blocking
transfers.  CPU storage for 50 K transitions with n_ctx=4 is roughly 360 MB.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.profiler import record_function


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for latent transitions.

    Args:
        capacity:     Maximum number of transitions stored before overwriting.
        n_ctx:        Context frames per transition (must match model training).
        latent_shape: (C, H, W) of a single encoded frame — (16, 8, 8) here.
        action_dim:   Dimensionality of the conditioning vector (7).
        gamma:        Discount factor used when computing MC returns.
    """

    def __init__(
        self,
        capacity:     int                    = 50_000,
        n_ctx:        int                    = 4,
        latent_shape: tuple[int, int, int]   = (16, 8, 8),
        action_dim:   int                    = 4,
        gamma:        float                  = 0.99,
    ) -> None:
        C, H, W         = latent_shape
        self.capacity   = capacity
        self.n_ctx      = n_ctx
        self.action_dim = action_dim
        self.gamma      = gamma
        self._head      = 0
        self._size      = 0

        # Pre-allocate all storage up front (zero-copy writes via .copy_())
        self.z         = torch.zeros(capacity, C, H, W,        dtype=torch.float16)
        self.z_next    = torch.zeros(capacity, C, H, W,        dtype=torch.float16)
        self.ctx       = torch.zeros(capacity, n_ctx, C, H, W, dtype=torch.float16)
        self.a_cond    = torch.zeros(capacity, action_dim,     dtype=torch.float32)
        self.r          = torch.zeros(capacity,                 dtype=torch.float32)
        self.mc_return  = torch.zeros(capacity,                 dtype=torch.float32)
        self.terminated = torch.zeros(capacity,                 dtype=torch.bool)
        self.truncated  = torch.zeros(capacity,                 dtype=torch.bool)
        self.success    = torch.zeros(capacity,                 dtype=torch.bool)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def push(
        self,
        z:          torch.Tensor,   # [C, H, W]
        a_cond:     torch.Tensor,   # [action_dim]
        r:          float,
        z_next:     torch.Tensor,   # [C, H, W]
        terminated: bool,
        truncated:  bool,
        success:    bool,
        ctx:        torch.Tensor,   # [n_ctx, C, H, W]
        mc_return:  float = 0.0,
    ) -> None:
        """Write one transition at the current head position."""
        i = self._head
        self.z[i].copy_(z.cpu().to(torch.float16))
        self.z_next[i].copy_(z_next.cpu().to(torch.float16))
        self.ctx[i].copy_(ctx.cpu().to(torch.float16))
        self.a_cond[i].copy_(a_cond.cpu().float())
        self.r[i]          = r
        self.mc_return[i]  = mc_return
        self.terminated[i] = terminated
        self.truncated[i]  = truncated
        self.success[i]    = success

        self._head = (self._head + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_episode(
        self,
        latents:    torch.Tensor,   # [T, C, H, W]  float16
        a_conds:    torch.Tensor,   # [T, action_dim] float32
        rewards:    torch.Tensor,   # [T] float32
        terminated: torch.Tensor,   # [T] bool
        truncated:  torch.Tensor,   # [T] bool
        success:    torch.Tensor,   # [T] bool
    ) -> None:
        """Push all transitions from one episode, computing MC returns first.

        Pads the context window with zeros at the start of the episode.
        MC returns treat terminated|truncated as the episode boundary.
        """
        T = latents.shape[0]
        if T < 2:
            return

        mc_returns = self._mc_returns(rewards, terminated)

        for t in range(T - 1):
            # Build [n_ctx, C, H, W] context ending at frame t
            ctx_start  = max(0, t - self.n_ctx + 1)
            ctx_frames = latents[ctx_start : t + 1]          # 1..n_ctx frames
            if ctx_frames.shape[0] < self.n_ctx:
                pad = torch.zeros(
                    self.n_ctx - ctx_frames.shape[0],
                    *latents.shape[1:],
                    dtype=latents.dtype,
                )
                ctx_frames = torch.cat([pad, ctx_frames], dim=0)

            self.push(
                z          = latents[t],
                a_cond     = a_conds[t],
                r          = float(rewards[t]),
                z_next     = latents[t + 1],
                terminated = bool(terminated[t]),
                truncated  = bool(truncated[t]),
                success    = bool(success[t]),
                ctx        = ctx_frames,
                mc_return  = float(mc_returns[t]),
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch and transfer to device.

        Returns a dict with keys:
            z, z_next, ctx, a_cond, r, mc_return, terminated, truncated, success
        All tensors have leading batch dimension batch_size.
        """
        with record_function("ReplayBuffer.sample"):
            idx = torch.randint(0, self._size, (batch_size,))
            return {
                "z":          self.z[idx].to(device, non_blocking=True),
                "z_next":     self.z_next[idx].to(device, non_blocking=True),
                "ctx":        self.ctx[idx].to(device, non_blocking=True),
                "a_cond":     self.a_cond[idx].to(device, non_blocking=True),
                "r":          self.r[idx].to(device, non_blocking=True),
                "mc_return":  self.mc_return[idx].to(device, non_blocking=True),
                "terminated": self.terminated[idx].to(device, non_blocking=True),
                "truncated":  self.truncated[idx].to(device, non_blocking=True),
                "success":    self.success[idx].to(device, non_blocking=True),
            }

    # ------------------------------------------------------------------
    # Seeding from offline HDF5
    # ------------------------------------------------------------------

    def seed_from_hdf5(self, hdf5_path: str) -> int:
        """Populate the buffer from an existing HDF5 trajectory file.

        Reads rewards, terminated, truncated, and success directly from the
        stored fields written by data/ingest.py.

        Returns the number of transitions added.
        """
        import h5py

        added = 0
        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f.keys()):
                if key == "metadata":
                    continue
                grp = f[key]
                if "latents" not in grp or "actions" not in grp:
                    continue

                T       = grp["latents"].shape[0]
                latents = torch.from_numpy(
                    grp["latents"][:].astype(np.float32)
                ).to(torch.float16)                               # [T, 16, 8, 8]
                actions = grp["actions"][:]                       # [T, action_dim]

                # Older HDF5 files (pre-ingest.py v2) may lack reward fields.
                # Fall back to zeros so seeding still populates z/ctx/a_cond.
                # WorldModelLoss reward/done/value heads will then rely solely
                # on online env.step() data once collection begins.
                if "rewards" in grp:
                    rewards    = torch.from_numpy(grp["rewards"][:].astype(np.float32))
                    terminated = torch.from_numpy(grp["terminated"][:].astype(bool))
                    truncated  = torch.from_numpy(grp["truncated"][:].astype(bool))
                    success    = torch.from_numpy(grp["success"][:].astype(bool))
                else:
                    rewards    = torch.zeros(T, dtype=torch.float32)
                    terminated = torch.zeros(T, dtype=torch.bool)
                    truncated  = torch.zeros(T, dtype=torch.bool)
                    success    = torch.zeros(T, dtype=torch.bool)

                # 4D conditioning: [dx, dy, dz, gripper]
                cnd = actions[:, :4].astype(np.float32)          # [T, 4]

                self.push_episode(
                    latents    = latents,
                    a_conds    = torch.from_numpy(cnd),
                    rewards    = rewards,
                    terminated = terminated,
                    truncated  = truncated,
                    success    = success,
                )
                added += max(0, T - 1)

        print(
            f"[ReplayBuffer] seeded {added:,} transitions from {hdf5_path} "
            f"({self._size:,}/{self.capacity:,} total)"
        )
        return added

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mc_returns(
        self, rewards: torch.Tensor, terminated: torch.Tensor
    ) -> torch.Tensor:
        """Compute MC returns, cutting only on terminated (not truncated).

        Truncated transitions are underestimated by gamma * V(s_next), which
        WorldModelLoss corrects at training time using the bootstrap mask.
        """
        T       = len(rewards)
        mc      = torch.zeros(T, dtype=torch.float32)
        running = 0.0
        for t in reversed(range(T)):
            if terminated[t]:
                running = 0.0
            running = float(rewards[t]) + self.gamma * running
            mc[t]   = running
        return mc

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size:,}/{self.capacity:,}, "
            f"n_ctx={self.n_ctx}, head={self._head})"
        )
