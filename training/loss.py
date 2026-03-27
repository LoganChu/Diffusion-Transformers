import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


class CFMLoss(nn.Module):
    """Conditional Flow Matching loss with auxiliary cube-position regression.

    Returns:
        (loss_total, loss_cfm, loss_aux) where
        loss_total = loss_cfm + 0.1 * loss_aux
    """

    AUX_WEIGHT = 0.1

    def forward(
        self,
        model: nn.Module,
        x_1: torch.Tensor,
        cond: torch.Tensor,       # [B, 4]: [dx,dy,dz,gripper]
        cube_pos: torch.Tensor,   # [B, 3]: auxiliary supervision target
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with record_function("CFMLoss"):
            B = x_1.shape[0]
            t = torch.rand(B, device=x_1.device, dtype=x_1.dtype)

            t_exp    = t.view(B, 1, 1, 1)
            x_0      = torch.randn_like(x_1)
            x_t      = (1 - t_exp) * x_0 + t_exp * x_1
            v_target = x_1 - x_0

            v_pred, cube_pos_pred = model(x_t, t, cond, return_aux=True)

            loss_cfm = F.mse_loss(v_pred, v_target)
            loss_aux = F.mse_loss(cube_pos_pred, cube_pos)
            loss     = loss_cfm + self.AUX_WEIGHT * loss_aux

            return loss, loss_cfm, loss_aux


class WorldModelLoss(nn.Module):
    """Combined loss for online training on replay-buffer batches.

    Two forward passes per step:

      Pass 1 — CFM (noisy input, random τ):
          Trains the world model to predict the velocity field that generates
          z_{t+1}.  Standard flow-matching objective.

      Pass 2 — Task heads (clean input, τ=1):
          Passes the clean z_{t+1} through the backbone so token representations
          reflect the actual next state.  Trains reward / done / value heads
          against env-labelled targets from the replay buffer.

    The cube_pos auxiliary loss is skipped during online training because the
    replay buffer does not store cube_pos labels (those exist only in the
    offline HDF5).  The backbone was already pretrained with that signal.
    """

    CFM_WEIGHT    = 1.0
    REWARD_WEIGHT = 1.0
    DONE_WEIGHT   = 0.5
    VALUE_WEIGHT  = 0.5
    GAMMA         = 0.99

    def forward(
        self,
        model:  nn.Module,
        batch:  dict,           # keys: z_next, ctx, a_cond, r, terminated, truncated, success, mc_return
    ) -> tuple[torch.Tensor, dict]:
        with record_function("WorldModelLoss"):
            z_next  = batch["z_next"]                    # [B, 16, 8, 8]
            ctx     = batch["ctx"]                       # [B, n_ctx, 16, 8, 8]
            a_cond  = batch["a_cond"]                    # [B, 4]
            r_gt    = batch["r"]                                              # [B]
            done_gt = (batch["terminated"] | batch["truncated"]).float()  # [B]
            mc_ret  = batch["mc_return"]                                   # [B]
            B       = z_next.shape[0]
            device  = z_next.device

            # ---- Pass 1: CFM ----
            t        = torch.rand(B, device=device, dtype=z_next.dtype)
            x_0      = torch.randn_like(z_next)
            t_exp    = t.view(B, 1, 1, 1)
            x_t      = (1 - t_exp) * x_0 + t_exp * z_next
            v_target = z_next - x_0

            v_pred   = model(x_t, t, a_cond, ctx_latents=ctx)
            loss_cfm = F.mse_loss(v_pred, v_target)

            # ---- Pass 2: task heads (clean input) ----
            t_ones               = torch.ones(B, device=device, dtype=z_next.dtype)
            _, r_pred, d_pred, v_pred_head = model(
                z_next, t_ones, a_cond, ctx_latents=ctx, return_heads=True
            )
            loss_reward = F.mse_loss(r_pred.squeeze(-1), r_gt)
            loss_done   = F.binary_cross_entropy_with_logits(
                d_pred.squeeze(-1), done_gt
            )

            # Bootstrap correction for truncated transitions:
            # MC returns cut off at the time limit, so add gamma * V(s_next).detach()
            # to recover the missing future. terminated transitions are already correct.
            trunc          = batch["truncated"].float()           # [B]
            value_target   = mc_ret + trunc * self.GAMMA * v_pred_head.squeeze(-1).detach()
            loss_value     = F.mse_loss(v_pred_head.squeeze(-1), value_target)

            loss = (
                self.CFM_WEIGHT    * loss_cfm    +
                self.REWARD_WEIGHT * loss_reward +
                self.DONE_WEIGHT   * loss_done   +
                self.VALUE_WEIGHT  * loss_value
            )

            return loss, {
                "loss":        loss.item(),
                "loss_cfm":    loss_cfm.item(),
                "loss_reward": loss_reward.item(),
                "loss_done":   loss_done.item(),
                "loss_value":  loss_value.item(),
            }


@torch.no_grad()
def sample_ode(
    model: nn.Module,
    cond: torch.Tensor,       # [B, 7]: [dx,dy,dz,gripper,ee_x,ee_y,ee_z]
    num_steps: int = 50,
    shape: tuple = (16, 8, 8),
) -> torch.Tensor:
    with record_function("sample_ode"):
        B = cond.shape[0]
        x = torch.randn(B, *shape, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
            v = model(x, t, cond)
            x.add_(dt * v)  # inplace per CLAUDE.md

        return x
