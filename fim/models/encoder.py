from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_groups(channels: int, max_groups: int = 8) -> int:
    g = max(1, min(max_groups, channels))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class SpectralMix2d(nn.Module):
    def __init__(self, channels: int, modes: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = max(1, int(modes))
        self.weight = nn.Parameter(torch.randn(channels, self.modes, self.modes, 2) * 0.02)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W]")

        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        if mh == 0 or mw == 0:
            return x

        out_ft = torch.zeros_like(x_ft)
        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())
        out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        mixed = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        scale = torch.clamp(self.scale, 0.0, 2.0)

        return x + scale * mixed + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()

        g = _make_groups(channels, groups)

        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.act(self.norm2(h)))
        return x + self.scale * h


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv7 = nn.Conv2d(channels, channels, 7, padding=3)

        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        fused = self.fuse(torch.cat([x3, x5, x7], dim=1))
        gate = torch.sigmoid(self.gate(x))

        return x + gate * fused


class FabricEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hidden_channels: int = 128,
        control_channels: int = 0,
        num_blocks: int = 4,
        spectral_modes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.control_channels = int(control_channels)

        stem_in = in_channels + self.control_channels
        g_hidden = _make_groups(hidden_channels, 8)
        g_latent = _make_groups(latent_channels, 8)

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, hidden_channels, 3, padding=1),
            nn.GroupNorm(g_hidden, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels, dropout=dropout) for _ in range(num_blocks)]
        )

        self.spectral = SpectralMix2d(hidden_channels, modes=spectral_modes)
        self.multiscale = MultiScaleFusion(hidden_channels)

        self.latent_proj = nn.Conv2d(hidden_channels, latent_channels, 1)

        self.write_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, latent_channels, 1),
            nn.Sigmoid(),
        )

        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, latent_channels, 1),
            nn.Tanh(),
        )

        self.energy_norm = nn.GroupNorm(g_latent, latent_channels)

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.noise_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input [B, C, H, W]")

        if control is not None:
            if control.ndim != 4:
                raise ValueError("Control must be [B, C, H, W]")
            if control.shape[0] != x.shape[0] or control.shape[2:] != x.shape[2:]:
                raise ValueError("Control must match input batch and spatial dims")
            x = torch.cat([x, control], dim=1)

        h = self.stem(x)
        h = self.blocks(h)

        spectral_out = self.spectral(h)
        h = h + self.residual_scale * spectral_out

        h = self.multiscale(h)

        z = self.latent_proj(h)

        gate = self.write_gate(h)
        mod = self.modulation(h)

        z = z * gate + mod

        z = self.energy_norm(z)

        if self.training:
            noise = torch.randn_like(z) * torch.clamp(self.noise_scale, 0.0, 1.0)
            z = z + noise

        scale = torch.clamp(self.scale, 0.0, 2.0)
        z = scale * z

        return z