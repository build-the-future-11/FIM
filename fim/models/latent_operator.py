from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_groups(channels: int, max_groups: int = 8) -> int:
    g = max(1, min(max_groups, channels))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class SpectralPropagation(nn.Module):
    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        self.channels = int(channels)
        self.modes = max(1, int(modes))

        self.weight = nn.Parameter(
            torch.randn(self.channels, self.modes, self.modes, 2) * 0.02
        )

        self.scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected [B,C,H,W]")

        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")

        mh = min(self.modes, H)
        mw = min(self.modes, W // 2 + 1)

        if mh == 0 or mw == 0:
            return x

        out_ft = torch.zeros_like(x_ft)

        weight = torch.view_as_complex(
            self.weight[:, :mh, :mw].contiguous()
        )

        out_ft[:, :, :mh, :mw] = (
            x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)
        )

        mixed = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        scale = torch.clamp(self.scale, 0.0, 2.0)

        return x + scale * mixed + self.bias


class LatentOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        propagation: Optional[nn.Module] = None,
        dt: float = 1.0,
        damping: float = 0.1,
        spectral_modes: int = 8,
        use_residual_mix: bool = False,
        use_adaptive_dt: bool = True,
        stochastic: bool = False,
    ):
        super().__init__()

        self.channels = int(channels)
        self.dt = float(dt)
        self.use_residual_mix = bool(use_residual_mix)
        self.use_adaptive_dt = bool(use_adaptive_dt)
        self.stochastic = bool(stochastic)

        g = _make_groups(self.channels)

        self.norm = nn.GroupNorm(g, self.channels)
        self.drive_norm = nn.GroupNorm(g, self.channels)

        self.write_proj = nn.Conv2d(self.channels, self.channels, 1)
        self.write_gate = nn.Conv2d(self.channels, self.channels, 1)

        if propagation is None:
            self.propagation_local = nn.Sequential(
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    groups=self.channels,
                    bias=False,
                ),
                nn.Conv2d(self.channels, self.channels, 1),
            )
        else:
            self.propagation_local = propagation

        self.propagation_spectral = SpectralPropagation(
            self.channels, modes=spectral_modes
        )
        self.spectral_scale = nn.Parameter(torch.tensor(0.0))

        self.nonlinear = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 1),
        )

        self.cross_channel = nn.Conv2d(self.channels, self.channels, 1)

        self.velocity = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(self.channels // 2, 2, 1),
        )

        self.damping = nn.Parameter(torch.full((self.channels,), damping))

        self.retrieval_proj = nn.Conv2d(self.channels, self.channels, 1)
        self.retrieval_gate = nn.Conv2d(self.channels, self.channels, 1)
        self.retrieval_scale = nn.Parameter(torch.tensor(0.1))

        if self.use_residual_mix:
            self.mix = nn.Conv2d(self.channels * 4, self.channels, 1)

        self.stability_scale = nn.Parameter(torch.tensor(0.99))

        if self.use_adaptive_dt:
            self.dt_modulator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels, self.channels, 1),
                nn.Sigmoid(),
            )

        self.noise_log_scale = nn.Parameter(torch.tensor(-4.0))

    def write(self, state: torch.Tensor, z: torch.Tensor):
        z = self.drive_norm(z)

        delta = torch.tanh(self.write_proj(z))
        gate = torch.sigmoid(self.write_gate(z))

        return state + gate * delta

    def _advect(self, x: torch.Tensor):
        vel = torch.tanh(self.velocity(x))
        vx = vel[:, 0:1]
        vy = vel[:, 1:2]

        dx = 0.5 * (torch.roll(x, -1, -1) - torch.roll(x, 1, -1))
        dy = 0.5 * (torch.roll(x, -1, -2) - torch.roll(x, 1, -2))

        return -(vx * dx + vy * dy)

    def evolve(self, state: torch.Tensor):
        s = self.norm(state)

        P_local = self.propagation_local(s)

        spectral_gain = torch.clamp(self.spectral_scale, 0.0, 2.0)
        P_spectral = spectral_gain * self.propagation_spectral(s)

        N = self.nonlinear(s)

        cross = self.cross_channel(s)

        adv = self._advect(s)

        damping = F.softplus(self.damping).view(1, -1, 1, 1)

        dF = (
            P_local
            + 0.5 * P_spectral
            + N
            + 0.3 * cross
            + 0.2 * adv
            - damping * s
        )

        dt = self.dt

        if self.use_adaptive_dt:
            dt_mod = self.dt_modulator(s)
            dt = dt * (0.5 + dt_mod)

        state_next = state + dt * dF

        if self.stochastic:
            sigma = F.softplus(self.noise_log_scale)
            state_next = state_next + sigma * torch.randn_like(state)

        stability = torch.clamp(self.stability_scale, 0.0, 1.0)

        return stability * state_next

    def inject_retrieval(
        self,
        state: torch.Tensor,
        retrieval: Optional[torch.Tensor],
    ):
        if retrieval is None or retrieval.numel() == 0:
            return state

        R_map = retrieval[:, :, None, None]

        injection = self.retrieval_proj(R_map)

        gate = torch.sigmoid(self.retrieval_gate(state))

        scale = torch.clamp(self.retrieval_scale, 0.0, 1.0)

        return state + scale * gate * injection

    def forward(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        retrieval: Optional[torch.Tensor] = None,
    ):
        if state.ndim != 4:
            raise ValueError(f"Expected state [B,C,H,W], got {state.shape}")

        state_written = self.write(state, z)

        state_evolved = self.evolve(state_written)

        state_reinforced = self.inject_retrieval(
            state_evolved, retrieval
        )

        if self.use_residual_mix:
            state_out = self.mix(
                torch.cat(
                    [
                        state_written,
                        state_evolved,
                        state_reinforced,
                        state,
                    ],
                    dim=1,
                )
            )
        else:
            state_out = state_reinforced

        return state_out