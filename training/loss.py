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
        cond: torch.Tensor,       # [B, 7]: [dx,dy,dz,gripper,ee_x,ee_y,ee_z]
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
