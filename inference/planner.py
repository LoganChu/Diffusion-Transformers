"""MPC/CEM planner for the OptiWorld-FM DiT world model.

Algorithm (Cross-Entropy Method):
  1. Maintain a Gaussian distribution over H-step action sequences [H, 4].
  2. Sample N candidates and roll each through the world model via Euler ODE.
  3. Score each trajectory with a pluggable score_fn(model, z, t) -> [N].
  4. Refit the distribution to the top-K elite sequences.
  5. Return the first action of the fitted mean as the control output.

The inner ODE uses simple Euler (num_ode_steps=4) — fast enough for planning
while the full Heun solver (inference/solver.py) is used for deployment quality.

Rolling context is updated each horizon step with the predicted latent so that
temporal conditioning stays consistent with training.

Score functions
---------------
cube_height_score_fn  — default, uses the existing CubePosHead as a proxy reward.
                        Swap for a proper reward+value scorer once those heads are
                        added to DiTSmall.

Usage
-----
    from inference.planner import cem_plan, cube_height_score_fn

    action_4d = cem_plan(
        model,
        ctx_latents,    # [1, n_ctx, 16, 8, 8]
        ctx_actions,    # [1, 7]
        current_ee_pos, # [3]
    )
    # action_4d: [4] — [dx, dy, dz, gripper]
"""

from __future__ import annotations

import torch
from torch.profiler import record_function

from models.dit import ACTION_DIM, IN_CHANNELS, LATENT_H, LATENT_W

# Physical robot action dimensionality [dx, dy, dz, gripper]
ROBOT_ACTION_DIM = 4

# Default bounds for pd_ee_delta_pos control mode
_ACTION_LO = torch.tensor([-0.08, -0.08, -0.08, -1.0])
_ACTION_HI = torch.tensor([ 0.08,  0.08,  0.08,  1.0])


# ---------------------------------------------------------------------------
# Fast Euler rollout — no KV cache, runs inside CEM over a batch of N
# ---------------------------------------------------------------------------

@torch.no_grad()
def _euler_rollout_step(
    model,
    ctx_latents: torch.Tensor,   # [N, n_ctx, 16, 8, 8]
    action_cond: torch.Tensor,   # [N, 7]  [dx,dy,dz,gripper,ee_x,ee_y,ee_z]
    num_ode_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Predict the next latent frame for all N candidates in one batched pass.

    Starts from N independent noise samples and integrates the world model
    velocity field using simple Euler steps.

    Returns [N, 16, 8, 8] predicted next latent frames.
    """
    with record_function("planner._euler_rollout_step"):
        N      = action_cond.shape[0]
        device = action_cond.device

        # Independent noise per candidate — zero allocation after this
        x    = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
        dt   = 1.0 / num_ode_steps
        t_buf = torch.empty(N, device=device, dtype=dtype)

        for i in range(num_ode_steps):
            t_buf.fill_(i * dt)
            v = model(x, t_buf, action_cond, ctx_latents=ctx_latents)
            x.add_(v, alpha=dt)   # inplace per CLAUDE.md

        return x  # [N, 16, 8, 8]


# ---------------------------------------------------------------------------
# Default score function: cube height proxy (uses existing CubePosHead)
# ---------------------------------------------------------------------------

def cube_height_score_fn(
    model,
    z_batch: torch.Tensor,   # [N, 16, 8, 8]  predicted next latent
    t_batch: torch.Tensor,   # [N]  ones — treat z as a clean latent
) -> torch.Tensor:
    """Score candidates by the predicted cube z-coordinate.

    Uses the CubePosHead already attached to DiTSmall — no extra parameters
    required. cube_pos_z > 0 means the cube is off the table; higher is better.

    Replace this with a reward_head + gamma^H * value_head scorer once those
    heads are added to DiTSmall.

    Returns [N] float32 scores.
    """
    with record_function("planner.cube_height_score_fn"):
        N      = z_batch.shape[0]
        device = z_batch.device
        # Zero action: cube_pos_head is weakly conditioned on action
        dummy = torch.zeros(N, ACTION_DIM, device=device, dtype=z_batch.dtype)
        # return_aux returns (velocity [B,16,8,8], cube_pos [B,3])
        _, cube_pos = model(z_batch, t_batch, dummy, return_aux=True)
        return cube_pos[:, 2].float()   # cube z-coordinate [N]


# ---------------------------------------------------------------------------
# CEM planner
# ---------------------------------------------------------------------------

@torch.no_grad()
def cem_plan(
    model,
    ctx_latents: torch.Tensor,      # [1, n_ctx, 16, 8, 8]
    ctx_actions: torch.Tensor,      # [1, 7]
    current_ee_pos: torch.Tensor,   # [3] or [1, 3]
    score_fn=None,
    horizon: int = 6,
    n_candidates: int = 64,
    n_elites: int = 8,
    n_cem_iters: int = 3,
    gamma: float = 0.99,
    num_ode_steps: int = 4,
    action_lo: torch.Tensor | None = None,
    action_hi: torch.Tensor | None = None,
    noise_std_init: float = 0.05,
) -> torch.Tensor:
    """CEM-MPC: plan one control action via cross-entropy optimisation.

    Args:
        model:           DiTSmall in eval mode (float16 on CUDA).
        ctx_latents:     [1, n_ctx, 16, 8, 8] recent encoded observation frames.
        ctx_actions:     [1, 7] conditioning vector for the most recent context step.
        current_ee_pos:  [3] current end-effector position (from proprioception).
        score_fn:        Callable(model, z [N,C,H,W], t [N]) -> [N] float32.
                         Defaults to cube_height_score_fn.
                         To plug in reward+value heads later:
                             def my_scorer(model, z, t):
                                 _, r, _, v = model(z, t, dummy, return_heads=True)
                                 return r.squeeze(-1) + 0.99**H * v.squeeze(-1)
        horizon:         Planning horizon in world-model steps.
        n_candidates:    Number of parallel candidate sequences (N).
        n_elites:        Top-K sequences used to refit the CEM distribution.
        n_cem_iters:     CEM refinement iterations.
        gamma:           Discount factor for multi-step returns.
        num_ode_steps:   Euler steps per latent transition (4 is a good tradeoff).
        action_lo/hi:    [4] action bounds. Default: pd_ee_delta_pos bounds ±0.08m.
        noise_std_init:  Initial std of the CEM action distribution.

    Returns:
        [4] tensor — first action of the best planned sequence [dx, dy, dz, gripper].
    """
    with record_function("cem_plan"):
        if score_fn is None:
            score_fn = cube_height_score_fn

        device = ctx_latents.device
        dtype  = next(model.parameters()).dtype
        N, H   = n_candidates, horizon

        # Action bounds on device
        lo = (_ACTION_LO if action_lo is None else action_lo).to(device, dtype=dtype)
        hi = (_ACTION_HI if action_hi is None else action_hi).to(device, dtype=dtype)

        # Expand singleton inputs to batch of N candidates
        # ctx_base is NOT cloned here — each CEM iteration clones for the rollout
        ctx_base = ctx_latents.expand(N, -1, -1, -1, -1)   # [N, n_ctx, C, H, W]
        ctx_act  = ctx_actions.expand(N, -1)                # [N, 7]
        ee_base  = current_ee_pos.view(1, 3).expand(N, -1)  # [N, 3]

        # CEM distribution over [H, 4] action sequences
        mean = torch.zeros(H, ROBOT_ACTION_DIM, device=device, dtype=dtype)
        std  = torch.full(
            (H, ROBOT_ACTION_DIM), noise_std_init, device=device, dtype=dtype
        )

        # Reusable timestep buffer passed to score_fn (t=1 → treat z as clean)
        t_ones = torch.ones(N, device=device, dtype=dtype)

        for cem_iter in range(n_cem_iters):
            with record_function(f"cem_iter_{cem_iter}"):

                # --- Sample N action sequences [N, H, 4] ---
                eps     = torch.randn(N, H, ROBOT_ACTION_DIM, device=device, dtype=dtype)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(lo, hi)

                # --- Roll out H steps for all N candidates ---
                # Clone so each iteration gets a fresh rolling window
                ctx_roll = ctx_base.clone()       # [N, n_ctx, C, H, W]
                ee_pos   = ee_base.clone().float() # [N, 3]  — kept float32 for accuracy
                returns  = torch.zeros(N, device=device, dtype=torch.float32)

                for h in range(H):
                    with record_function(f"cem_horizon_{h}"):
                        a_4d   = actions[:, h, :]                   # [N, 4]
                        a_cond = torch.cat(
                            [a_4d, ee_pos.to(dtype)], dim=-1
                        )   # [N, 7]

                        z_next = _euler_rollout_step(
                            model, ctx_roll, a_cond, num_ode_steps, dtype
                        )   # [N, 16, 8, 8]

                        # Score this horizon step
                        step_score = score_fn(model, z_next, t_ones)  # [N] float32
                        returns.add_(step_score * float(gamma ** h))

                        # Advance rolling context window: drop t-n_ctx+1, append z_next
                        ctx_roll = torch.cat(
                            [ctx_roll[:, 1:], z_next.unsqueeze(1)], dim=1
                        )   # [N, n_ctx, C, H, W]

                        # Advance EE estimate (linear integration, no PD clamp)
                        ee_pos = ee_pos + a_4d[:, :3].float()

                # --- Refit to top-K elites ---
                _, elite_idx      = returns.topk(n_elites)
                elite_actions     = actions[elite_idx].float()  # [n_elites, H, 4]

                mean = elite_actions.mean(dim=0).to(dtype)
                std  = elite_actions.std(dim=0).clamp(min=0.01).to(dtype)

        # First action of the fitted mean is the control output
        return mean[0].float()  # [4] — return float32 for env compatibility
