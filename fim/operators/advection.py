from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _grad_x(x: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    if periodic:
        return 0.5 * (torch.roll(x, -1, dims=3) - torch.roll(x, 1, dims=3))
    out = torch.zeros_like(x)
    out[:, :, :, 1:-1] = 0.5 * (x[:, :, :, 2:] - x[:, :, :, :-2])
    return out


def _grad_y(x: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    if periodic:
        return 0.5 * (torch.roll(x, -1, dims=2) - torch.roll(x, 1, dims=2))
    out = torch.zeros_like(x)
    out[:, :, 1:-1, :] = 0.5 * (x[:, :, 2:, :] - x[:, :, :-2, :])
    return out


class AdvectionOperator(nn.Module):
    """
    Advection: dF/dt = -v·∇F
    """

    def __init__(
        self,
        channels: int,
        velocity: Optional[Tuple[float, float]] = None,
        learnable_velocity: bool = False,
        periodic: bool = True,
    ) -> None:
        super().__init__()
        self.periodic = periodic

        if velocity is None:
            velocity = (0.0, 0.0)

        if learnable_velocity:
            self.velocity = nn.Parameter(torch.tensor([velocity[0], velocity[1]], dtype=torch.float32))
        else:
            self.register_buffer("velocity", torch.tensor([velocity[0], velocity[1]], dtype=torch.float32))

        self.channel_mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, F_: torch.Tensor) -> torch.Tensor:
        vx, vy = self.velocity[0], self.velocity[1]
        gx = _grad_x(F_, periodic=self.periodic)
        gy = _grad_y(F_, periodic=self.periodic)
        adv = -(vx * gx + vy * gy)
        return self.channel_mix(adv)