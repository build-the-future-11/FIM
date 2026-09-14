from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------
# FAST OPERATORS (CONV-BASED)
# ---------------------------------
def laplacian_2d(x: torch.Tensor) -> torch.Tensor:
    # depthwise conv kernel
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    C = x.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)

    return F.conv2d(x, kernel, padding=1, groups=C)


def gradient_2d(x: torch.Tensor):
    kx = torch.tensor(
        [[-1.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 1, 2)

    ky = torch.tensor(
        [[-1.0],
         [1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 2, 1)

    C = x.shape[1]
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=(0, 1), groups=C)
    gy = F.conv2d(x, ky, padding=(1, 0), groups=C)

    return gx, gy


# ---------------------------------
# FAST PROPAGATION OPERATOR
# ---------------------------------
class PropagationOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        learn_diffusion: bool = True,
        learn_velocity: bool = True,
    ):
        super().__init__()

        # diffusion coefficient per channel
        if learn_diffusion:
            self.D = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_buffer("D", torch.ones(1, channels, 1, 1))

        # velocity vector
        if learn_velocity:
            self.v = nn.Parameter(torch.zeros(1, 2))
        else:
            self.register_buffer("v", torch.zeros(1, 2))

        # nonlinear block (channels-first now → no permute)
        self.nonlinear = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W]  ← IMPORTANT CHANGE
        dt: float = 1.0,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ---------------------------------
        # DIFFUSION (Laplacian)
        # ---------------------------------
        lap = laplacian_2d(x)
        diffusion = self.D * lap

        # ---------------------------------
        # ADVECTION (grad-based)
        # ---------------------------------
        gx, gy = gradient_2d(x)

        vx = self.v[..., 0].view(1, 1, 1, 1)
        vy = self.v[..., 1].view(1, 1, 1, 1)

        advection = -(vx * gx + vy * gy)

        # ---------------------------------
        # NONLINEAR
        # ---------------------------------
        nonlinear = self.nonlinear(x)

        if control is not None:
            nonlinear = nonlinear + control

        # ---------------------------------
        # UPDATE (FUSED)
        # ---------------------------------
        update = diffusion + advection + nonlinear

        scale = torch.clamp(self.scale, 0.0, 1.0)

        return x + dt * scale * update