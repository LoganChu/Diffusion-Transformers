"""HDF5 Trajectory Dataset for DiT training.

Each sample is a (latent_frame, action) pair drawn from stored episodes.
Supports optional context window for multi-frame conditioning.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Reads latent trajectories from HDF5 produced by data/ingest.py.

    HDF5 schema (per episode group):
        latents: [T, 16, 8, 8] float16
        actions: [T, action_dim] float32

    Each __getitem__ returns:
        x_1:    [16, 8, 8]  target latent frame (bfloat16)
        action: [action_dim] action that produced this frame (bfloat16)
    """

    def __init__(self, hdf5_path: str, ctx_frames: int = 0) -> None:
        self.hdf5_path = hdf5_path
        self.ctx_frames = ctx_frames

        # Build an index: list of (episode_key, timestep) pairs
        # We skip the first ctx_frames steps so we always have full context.
        # Bad frames (NaN or abs > 1000) and any sample whose context window
        # overlaps a bad frame are excluded so corrupted latents never reach training.
        self._index: list[tuple[str, int]] = []
        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f.keys()):
                if key == "metadata":
                    continue
                grp = f[key]
                T = grp["latents"].shape[0]
                start = max(1, ctx_frames)  # need at least 1 prior frame for action

                # Compute bad[t] = True if frame t has NaN or any abs value > 1000.
                # Normal Cosmos latents are in ~[-5, 5]; values > 1000 are encoder
                # corruption from the CUDA stream race condition (now fixed in ingest.py).
                latents_all = grp["latents"][:].astype(np.float32)  # [T, 16, 8, 8]
                finite = np.isfinite(latents_all).all(axis=(1, 2, 3))        # [T] bool
                in_range = np.abs(latents_all).max(axis=(1, 2, 3)) < 50      # [T] bool
                bad = ~(finite & in_range)                                    # [T] bool

                for t in range(start, T):
                    # Exclude if target frame OR any context frame is bad.
                    window_start = t - ctx_frames
                    if not bad[window_start : t + 1].any():
                        self._index.append((key, t))

        self._file: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", rdcc_nbytes=32 * 1024 * 1024)
        return self._file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._open()
        ep_key, t = self._index[idx]
        grp = f[ep_key]

        # Target frame
        x_1 = torch.from_numpy(grp["latents"][t].astype(np.float32))
        # Action that led to this frame (action at t-1 produces frame at t)
        action = torch.from_numpy(grp["actions"][t - 1].astype(np.float32))

        # --- Enriched conditioning ---
        # EE position at t-1 is always available from the robot's proprioception.
        # Cube position at t is stored as an auxiliary supervision target only
        # (not available at inference without perception).
        if "ee_pos" in grp:
            ee_pos   = torch.from_numpy(grp["ee_pos"][t - 1].astype(np.float32))   # [3]
            cube_pos = torch.from_numpy(grp["cube_pos"][t].astype(np.float32))      # [3]
            phase    = int(grp["phase"][t - 1])
        else:
            # Backwards-compatible fallback for HDF5 files without state fields
            ee_pos   = torch.zeros(3)
            cube_pos = torch.zeros(3)
            phase    = -1

        # 4D conditioning vector: [dx, dy, dz, gripper]
        # ee_pos is stored separately in HDF5 and does not need to be duplicated here
        cond = action

        result = {
            "x_1":      x_1,
            "action":   action,   # kept for reference / compat
            "cond":     cond,     # 7D: what the model actually receives
            "cube_pos": cube_pos, # auxiliary supervision target
            "phase":    torch.tensor(phase, dtype=torch.long),
        }

        # Optional context frames for cached inference validation
        if self.ctx_frames > 0:
            ctx_start = t - self.ctx_frames
            ctx_latents = torch.from_numpy(
                grp["latents"][ctx_start:t].astype(np.float32)
            )
            result["ctx_latents"] = ctx_latents

        return result

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
