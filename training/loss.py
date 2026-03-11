import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


class CFMLoss(nn.Module):
    def forward(self, model: nn.Module, x_1: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with record_function("CFMLoss"):
            B = x_1.shape[0]
            t = torch.rand(B, device=x_1.device, dtype=x_1.dtype)  # [B]
            x_0 = torch.randn_like(x_1)  # [B, 16, 8, 8]

            t_exp = t.view(B, 1, 1, 1)  # broadcast
            x_t = (1 - t_exp) * x_0 + t_exp * x_1  # OT interpolant

            v_target = x_1 - x_0  # constant velocity field
            v_pred = model(x_t, t, action)  # [B, 16, 8, 8]

            return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_ode(
    model: nn.Module,
    action: torch.Tensor,
    num_steps: int = 50,
    shape: tuple = (16, 8, 8),
) -> torch.Tensor:
    with record_function("sample_ode"):
        B = action.shape[0]
        x = torch.randn(B, *shape, device=action.device, dtype=action.dtype)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
            v = model(x, t, action)
            x.add_(dt * v)  # inplace per CLAUDE.md

        return x
