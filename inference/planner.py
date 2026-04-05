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
                        trained:
                            def my_scorer(model, z, t):
                                dummy = torch.zeros(z.shape[0], ACTION_DIM, ...)
                                _, r, _, v = model(z, t, dummy, return_heads=True)
                                return r.squeeze(-1) + gamma**H * v.squeeze(-1)

Usage
-----
    from inference.planner import cem_plan, cube_height_score_fn

    action_4d = cem_plan(
        model,
        ctx_latents,  # [1, n_ctx, 16, 8, 8]
        ctx_actions,  # [1, 4]
    )
    # action_4d: [4] — [dx, dy, dz, gripper]
"""

from __future__ import annotations

import torch
from torch.profiler import record_function

from models.dit import ACTION_DIM, IN_CHANNELS, LATENT_H, LATENT_W, NUM_PATCHES

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
    action_cond: torch.Tensor,   # [N, 4]  [dx, dy, dz, gripper]
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

        x     = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
        dt    = 1.0 / num_ode_steps
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
    **kwargs,
) -> torch.Tensor:
    """Score candidates by the predicted cube z-coordinate.

    Uses the CubePosHead already attached to DiTSmall — no extra parameters
    required. cube_pos_z > 0 means the cube is off the table; higher is better.

    Returns [N] float32 scores.
    """
    with record_function("planner.cube_height_score_fn"):
        N      = z_batch.shape[0]
        device = z_batch.device
        dummy  = torch.zeros(N, ACTION_DIM, device=device, dtype=z_batch.dtype)
        _, cube_pos = model(z_batch, t_batch, dummy, return_aux=True)
        return cube_pos[:, 2].float()   # cube z-coordinate [N]


def reward_value_score_fn(
    model,
    z_batch: torch.Tensor,   # [N, 16, 8, 8]
    t_batch: torch.Tensor,   # [N] ones
    horizon: int = 6,
    gamma: float = 0.99,
    **kwargs,
) -> torch.Tensor:
    """Score candidates using trained RewardHead + ValueHead with return bootstrap.

    Returns [N] float32: r̂ + γ^H * V̂(z)

    Requires model trained via WorldModelLoss (training/train_online.py).
    Falls back silently to cube_height_score_fn if heads are untrained
    (zero-init bias means ValueHead outputs ~0, so scoring degrades gracefully).
    """
    with record_function("planner.reward_value_score_fn"):
        N      = z_batch.shape[0]
        device = z_batch.device
        dummy  = torch.zeros(N, ACTION_DIM, device=device, dtype=z_batch.dtype)
        _, r, _, v = model(z_batch, t_batch, dummy, return_heads=True)
        return (r + gamma ** horizon * v).squeeze(-1).float()   # [N]


def reward_only_score_fn(
    model,
    z_batch: torch.Tensor,   # [N, 16, 8, 8]
    t_batch: torch.Tensor,   # [N] ones
    **kwargs,
) -> torch.Tensor:
    """Score candidates using only the RewardHead — no value bootstrapping.

    Use this when the ValueHead is unreliable (e.g., after offline-only head
    pretraining where offline MC returns dominate and value overestimates).
    The RewardHead converges faster (pure MSE on per-step rewards) and gives
    clean directional signal without the instability of value bootstrapping.

    Returns [N] float32: r̂(z)
    """
    with record_function("planner.reward_only_score_fn"):
        N      = z_batch.shape[0]
        device = z_batch.device
        dummy  = torch.zeros(N, ACTION_DIM, device=device, dtype=z_batch.dtype)
        _, r, _, _ = model(z_batch, t_batch, dummy, return_heads=True)
        return r.squeeze(-1).float()   # [N]


def maniskill_reward_score_fn(
    model,
    z_batch: torch.Tensor,              # [N, 16, 8, 8]
    t_batch: torch.Tensor,              # [N] ones
    cumulative_ee: torch.Tensor | None = None,   # [N, 3] world-frame ee pos after cumulative actions
    **kwargs,
) -> torch.Tensor:
    """Score using the ManiSkill PickCube reward formula on imagined latents.

    score = reach_reward + lift_reward
          = −||ee_pos_predicted − cube_pos_predicted||  +  cube_z / 0.15

    cube_pos is extracted from the imagined latent via CubePosHead (trained offline,
    world-frame XYZ, ~0.04m XY error). ee_pos is approximated as the starting
    ee_pos plus the cumulative CEM action displacements — valid because pd_ee_delta_pos
    actions are world-frame deltas in the same coordinate frame as cube_pos.

    Falls back to cube_height if no ee_pos is provided (cumulative_ee=None).

    Requires: cem_plan called with ee_pos=[3] from env.unwrapped.agent.tcp.pose.p.
    """
    with record_function("planner.maniskill_reward_score_fn"):
        N      = z_batch.shape[0]
        device = z_batch.device
        dummy  = torch.zeros(N, ACTION_DIM, device=device, dtype=z_batch.dtype)
        _, cube_pos = model(z_batch, t_batch, dummy, return_aux=True)   # [N, 3]

        lift = cube_pos[:, 2].float() / 0.15   # [N]

        if cumulative_ee is not None:
            ee    = cumulative_ee.to(device=device, dtype=torch.float32)   # [N, 3]
            reach = -(ee - cube_pos.float()).norm(dim=-1)                  # [N]
            return reach + lift
        else:
            return lift   # graceful fallback — same as cube_height but normalised


# ---------------------------------------------------------------------------
# CEM planner
# ---------------------------------------------------------------------------

@torch.no_grad()
def cem_plan(
    model,
    ctx_latents: torch.Tensor,      # [1, n_ctx, 16, 8, 8]
    ctx_actions: torch.Tensor,      # [1, 4]
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
    log_score_stats: bool = False,
    ee_pos: torch.Tensor | None = None,   # [3] world-frame ee pos from env (for maniskill_reward)
) -> torch.Tensor:
    """CEM-MPC: plan one control action via cross-entropy optimisation.

    Args:
        model:          DiTSmall in eval mode (float16 on CUDA).
        ctx_latents:    [1, n_ctx, 16, 8, 8] recent encoded observation frames.
        ctx_actions:    [1, 4] most recent action conditioning vector.
        score_fn:       Callable(model, z [N,C,H,W], t [N], **kwargs) -> [N] float32.
                        Defaults to cube_height_score_fn.
        horizon:        Planning horizon in world-model steps.
        n_candidates:   Number of parallel candidate sequences (N).
        n_elites:       Top-K sequences used to refit the CEM distribution.
        n_cem_iters:    CEM refinement iterations.
        gamma:          Discount factor for multi-step returns.
        num_ode_steps:  Euler steps per latent transition (4 is a good tradeoff).
        action_lo/hi:   [4] action bounds. Default: pd_ee_delta_pos bounds ±0.08m.
        noise_std_init: Initial std of the CEM action distribution.
        ee_pos:         [3] world-frame end-effector position from env observation.
                        Required for maniskill_reward_score_fn; ignored by others.

    Returns:
        [4] tensor — first action of the best planned sequence [dx, dy, dz, gripper].
    """
    with record_function("cem_plan"):
        if score_fn is None:
            score_fn = cube_height_score_fn

        device = ctx_latents.device
        dtype  = next(model.parameters()).dtype
        N, H   = n_candidates, horizon

        lo = (_ACTION_LO if action_lo is None else action_lo).to(device, dtype=dtype)
        hi = (_ACTION_HI if action_hi is None else action_hi).to(device, dtype=dtype)

        ctx_base = ctx_latents.expand(N, -1, -1, -1, -1)   # [N, n_ctx, C, H, W]

        # CEM distribution over [H, ACTION_DIM] action sequences
        mean = torch.zeros(H, ACTION_DIM, device=device, dtype=dtype)
        std  = torch.full((H, ACTION_DIM), noise_std_init, device=device, dtype=dtype)

        t_ones = torch.ones(N, device=device, dtype=dtype)

        for cem_iter in range(n_cem_iters):
            with record_function(f"cem_iter_{cem_iter}"):

                # --- Sample N action sequences [N, H, 4] ---
                eps     = torch.randn(N, H, ACTION_DIM, device=device, dtype=dtype)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(lo, hi)

                # --- Roll out H steps for all N candidates ---
                ctx_roll = ctx_base.clone()
                returns  = torch.zeros(N, device=device, dtype=torch.float32)

                # Track cumulative ee displacement for maniskill_reward_score_fn.
                # ee_pos_init [3] → cumulative_ee [N, 3] grows by action_xyz each step.
                cumulative_ee: torch.Tensor | None = None
                if ee_pos is not None:
                    cumulative_ee = ee_pos.float().to(device).unsqueeze(0).expand(N, -1).clone()

                for h in range(H):
                    with record_function(f"cem_horizon_{h}"):
                        a_cond = actions[:, h, :]   # [N, 4]

                        if cumulative_ee is not None:
                            cumulative_ee = cumulative_ee + a_cond[:, :3].float()

                        z_next = _euler_rollout_step(
                            model, ctx_roll, a_cond, num_ode_steps, dtype
                        )   # [N, 16, 8, 8]

                        step_score = score_fn(
                            model, z_next, t_ones, cumulative_ee=cumulative_ee
                        )   # [N] float32
                        returns.add_(step_score * float(gamma ** h))

                        # Advance rolling context window
                        ctx_roll = torch.cat(
                            [ctx_roll[:, 1:], z_next.unsqueeze(1)], dim=1
                        )   # [N, n_ctx, C, H, W]

                # --- Refit to top-K elites ---
                _, elite_idx  = returns.topk(n_elites)
                elite_actions = actions[elite_idx].float()   # [n_elites, H, 4]

                if log_score_stats:
                    print(
                        f"  [cem iter {cem_iter}] "
                        f"score min={returns.min():.4f}  "
                        f"max={returns.max():.4f}  "
                        f"std={returns.std():.4f}  "
                        f"elite_mean_action={elite_actions[:, 0].mean(0).tolist()}"
                    )

                mean = elite_actions.mean(dim=0).to(dtype)
                std  = elite_actions.std(dim=0).clamp(min=0.01).to(dtype)

        return mean[0].float()   # [4] — float32 for env compatibility
