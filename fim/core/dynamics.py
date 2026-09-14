from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fim.geometry.grid import laplacian_kernel2d
from .stability import clamp_norm_


class DepthwiseLaplacian(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernel = laplacian_kernel2d().view(1, 1, 3, 3)
        kernel = kernel / kernel.abs().sum()
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1))
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)


class FabricDynamics(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        diffusion_strength: float = 0.1,
        reaction_hidden: int | None = None,
        damping: float = 0.01,
        dt: float = 1.0,
        max_norm: float | None = None,
    ):
        super().__init__()

        self.channels = int(channels)
        hidden = reaction_hidden or max(channels, 16)

        self.diffusion = DepthwiseLaplacian(channels)

        self.reaction = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

        self.diffusion_strength = nn.Parameter(torch.tensor(diffusion_strength))
        self.damping = nn.Parameter(torch.full((channels,), damping))

        self.dt = float(dt)
        self.max_norm = max_norm

        self.stability_scale = nn.Parameter(torch.tensor(0.98))
        self.reaction_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, fabric: torch.Tensor) -> torch.Tensor:
        D = torch.clamp(self.diffusion_strength, 0.0, 1.0)

        lam = F.softplus(self.damping).view(1, -1, 1, 1)

        prop = D * self.diffusion(fabric)

        nonlin = self.reaction(fabric)
        nonlin = self.reaction_scale * nonlin

        dF = prop + nonlin - lam * fabric

        out = fabric + self.dt * dF

        s = torch.clamp(self.stability_scale, 0.0, 1.0)
        out = s * out

        if self.max_norm is not None:
            clamp_norm_(out, self.max_norm)

        return out


class SalienceGate(nn.Module):
    def __init__(self, in_channels: int, hidden: int | None = None):
        super().__init__()

        hidden = hidden or max(in_channels, 16)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)

        t = torch.clamp(self.temperature, 0.1, 10.0)
        s = torch.clamp(self.scale, 0.1, 5.0)

        logits = s * logits + self.bias

        return torch.sigmoid(logits / t)