from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DampingOperator(nn.Module):
    def __init__(self, channels: int, damping: float = 0.1) -> None:
        super().__init__()
        self.channels = int(channels)
        self.damping = nn.Parameter(torch.tensor(float(damping)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        return -torch.clamp(self.damping, 0.0, 10.0) * x


class PhysicsOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        diffusion: float = 0.05,
        damping: float = 0.1,
        advection: bool = True,
        stochastic: bool = False,
        spectral: bool = True,
        per_channel: bool = True,
        nonlinear: bool = True,
        anisotropic: bool = True,
        adaptive_damping: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.stochastic = bool(stochastic)
        self.spectral_enabled = bool(spectral)
        self.advection_enabled = bool(advection)
        self.nonlinear_enabled = bool(nonlinear)
        self.anisotropic = bool(anisotropic)
        self.adaptive_damping = bool(adaptive_damping)
        self.eps = float(eps)

        self.damping = DampingOperator(channels, damping=damping)
        self.diffusion = nn.Parameter(torch.full((channels,), float(diffusion)))

        if per_channel:
            self.view_shape = (1, channels, 1, 1)
        else:
            self.view_shape = (1, 1, 1, 1)

        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        if self.advection_enabled:
            self.velocity = nn.Sequential(
                nn.Conv2d(channels, max(1, channels // 2), 1),
                nn.GELU(),
                nn.Conv2d(max(1, channels // 2), 2, 1),
            )
        else:
            self.velocity = None

        if self.stochastic:
            self.noise_log_scale = nn.Parameter(torch.tensor(-4.0))

        if self.spectral_enabled:
            self.spectral_gain = nn.Parameter(torch.tensor(0.1))

        if self.nonlinear_enabled:
            self.reaction = nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )

        if self.anisotropic:
            self.diffusion_tensor = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.Tanh(),
            )

        if self.adaptive_damping:
            self.damping_mod = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )

        self.energy_proj = nn.Conv2d(channels, channels, 1)

        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def get_damping(self, x: torch.Tensor):
        d = F.softplus(self.damping.damping) + self.eps
        d = d.view(self.view_shape)

        if self.adaptive_damping:
            mod = self.damping_mod(x)
            d = d * (0.5 + mod)

        return d

    def get_diffusion(self):
        d = F.softplus(self.diffusion) + self.eps
        return d.view(self.view_shape)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return (-4.0 * x + torch.roll(x, 1, dims=-2) + torch.roll(x, -1, dims=-2) + torch.roll(x, 1, dims=-1) + torch.roll(x, -1, dims=-1))

    def gradient(self, x: torch.Tensor):
        dx = 0.5 * (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))
        dy = 0.5 * (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))
        return dx, dy

    def anisotropic_diffusion(self, x: torch.Tensor):
        dx, dy = self.gradient(x)
        tensor = self.diffusion_tensor(x)
        gx, gy = torch.chunk(tensor, 2, dim=1)
        return gx * dx + gy * dy

    def spectral_mix(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        gain = torch.clamp(self.spectral_gain, 0.0, 1.0)
        xf = xf * (1.0 - gain)
        return torch.fft.irfft2(xf, s=(h, w), norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = self.get_damping(x)
        nu = self.get_diffusion()

        damping_term = -lam * x
        diffusion_term = nu * self.laplacian(x)
        prop_term = self.depthwise(x)

        adv_term = 0.0
        if self.velocity is not None:
            v = self.velocity(x)
            dx, dy = self.gradient(x)
            adv_term = -(v[:, :1] * dx + v[:, 1:] * dy)

        anisotropic_term = 0.0
        if self.anisotropic:
            anisotropic_term = self.anisotropic_diffusion(x)

        reaction_term = 0.0
        if self.nonlinear_enabled:
            reaction_term = self.reaction(x)

        spectral_term = 0.0
        if self.spectral_enabled:
            spectral_term = self.spectral_mix(x) - x

        noise_term = 0.0
        if self.stochastic:
            sigma = F.softplus(self.noise_log_scale)
            noise_term = sigma * torch.randn_like(x)

        energy = torch.tanh(self.energy_proj(x))
        gate = self.global_gate(x)

        total = damping_term + diffusion_term + prop_term + adv_term + 0.5 * anisotropic_term + 0.5 * reaction_term + spectral_term + noise_term
        total = total * (1.0 + energy) * gate

        scale = torch.clamp(self.output_scale, 0.1, 5.0)
        return scale * total
