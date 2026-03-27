"""Heuristic reward function for retroactively labeling offline HDF5 data.

For online collection, use the reward returned directly by env.step() —
ManiSkill PickCube-v1 computes the real reward for you.  This module is only
needed when seeding the replay buffer from the offline HDF5 dataset, which was
collected without storing env.step() reward labels.

Reward components (dense, additive):
  reach:   -||ee_xy - cube_xy||         pull EE toward the cube in the XY plane
  height:  cube_z / LIFT_SUCCESS_Z      normalised lift progress in [0, 1]
  success: +5 bonus when cube is lifted above the threshold

Done criterion: cube_z > LIFT_SUCCESS_Z
"""

from __future__ import annotations

import numpy as np
import torch

LIFT_SUCCESS_Z = 0.15   # metres — cube must reach this height to count as success


def compute_reward(
    ee_pos:   torch.Tensor,   # [3] or [N, 3]
    cube_pos: torch.Tensor,   # [3] or [N, 3]
) -> torch.Tensor:
    """Dense per-step reward from stored state.

    Returns a scalar or [N] float32 tensor matching the input batch dimension.
    """
    ee_pos   = torch.as_tensor(ee_pos,   dtype=torch.float32)
    cube_pos = torch.as_tensor(cube_pos, dtype=torch.float32)

    batched = ee_pos.dim() > 1
    if not batched:
        ee_pos   = ee_pos.unsqueeze(0)
        cube_pos = cube_pos.unsqueeze(0)

    reach   = -torch.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=-1)          # XY only
    height  = (cube_pos[:, 2] / LIFT_SUCCESS_Z).clamp(0.0, 1.0)
    success = (cube_pos[:, 2] > LIFT_SUCCESS_Z).float() * 5.0

    reward = reach + height + success
    return reward if batched else reward.squeeze(0)


def compute_done(cube_pos: torch.Tensor) -> torch.Tensor:
    """True when the cube has been lifted above the success threshold."""
    cube_pos = torch.as_tensor(cube_pos, dtype=torch.float32)
    cube_z   = cube_pos[..., 2]
    return cube_z > LIFT_SUCCESS_Z


def label_episode(
    ee_pos:   np.ndarray,   # [T, 3]
    cube_pos: np.ndarray,   # [T, 3]
    gamma:    float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label a full HDF5 episode with rewards, dones, and Monte Carlo returns.

    Args:
        ee_pos:   End-effector positions stored in HDF5.
        cube_pos: Cube positions stored in HDF5.
        gamma:    Discount factor for MC return computation.

    Returns:
        rewards:    [T] float32
        dones:      [T] bool
        mc_returns: [T] float32
    """
    T       = len(ee_pos)
    rewards = np.zeros(T, dtype=np.float32)
    dones   = np.zeros(T, dtype=bool)

    for t in range(T):
        r = compute_reward(
            torch.from_numpy(ee_pos[t]),
            torch.from_numpy(cube_pos[t]),
        )
        rewards[t] = float(r)
        dones[t]   = bool(compute_done(torch.from_numpy(cube_pos[t])))

    # Backward pass for Monte Carlo returns
    mc_returns = np.zeros(T, dtype=np.float32)
    running    = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            running = 0.0
        running       = rewards[t] + gamma * running
        mc_returns[t] = running

    return rewards, dones, mc_returns
