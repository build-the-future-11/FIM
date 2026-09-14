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

        weight = torch.randn(self.channels, self.modes, self.modes, 2) * 0.02
        self.weight = nn.Parameter(weight)

        self.scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(self.channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W]")

        b, c, h, w = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")

        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        if mh == 0 or mw == 0:
            return x

        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())

        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        mixed = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

        scale = torch.clamp(self.scale, 0.0, 2.0)
        return x + scale * mixed + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()

        g = _make_groups(channels, groups)

        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.act = nn.GELU()

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + self.scale * h + self.shift


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(self.pool(x))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x)
        return x * w


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv7 = nn.Conv2d(channels, channels, 7, padding=3)

        self.mix = nn.Conv2d(channels * 3, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h3 = self.conv3(x)
        h5 = self.conv5(x)
        h7 = self.conv7(x)

        fused = self.mix(torch.cat([h3, h5, h7], dim=1))
        g = self.gate(x)

        return x + g * fused


class FabricDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        spectral_modes: int = 12,
    ) -> None:
        super().__init__()

        g_hidden = _make_groups(hidden_channels, 8)

        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(g_hidden, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        self.multi_scale = MultiScaleFusion(hidden_channels)

        self.channel_attn = ChannelAttention(hidden_channels)
        self.spatial_attn = SpatialAttention(hidden_channels)

        self.spectral = SpectralMix2d(hidden_channels, modes=spectral_modes)

        self.readout_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )

        self.readout_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.energy_norm = nn.GroupNorm(g_hidden, hidden_channels)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.attn_scale = nn.Parameter(torch.tensor(0.5))
        self.spectral_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError("Expected [B, C, H, W]")

        h = self.stem(z)

        for block in self.blocks:
            h = block(h)

        h = self.multi_scale(h)

        attn = self.channel_attn(h)
        attn = self.spatial_attn(attn)

        h = h + torch.clamp(self.attn_scale, 0.0, 2.0) * attn

        spectral_out = self.spectral(h)
        h = h + torch.clamp(self.spectral_scale, 0.0, 2.0) * spectral_out

        gate = self.readout_gate(h)
        proj = self.readout_proj(h)

        h = proj * gate + torch.clamp(self.residual_scale, 0.0, 2.0) * h

        h = self.energy_norm(h)

        out = self.head(h)

        scale = torch.clamp(self.output_scale, 0.0, 5.0)
        return scale * out + self.output_bias