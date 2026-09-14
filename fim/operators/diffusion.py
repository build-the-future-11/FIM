from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inverse(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.expm1(x) + eps)


class DiffusionOperator(nn.Module):
    def __init__(
        self,
        channels: int,
        diffusivity: float = 0.1,
        dt: float = 1.0,
        mode: str = "spectral",
        fractional: bool = False,
        alpha: float = 1.5,
        anisotropic: bool = False,
        learnable: bool = True,
        max_diffusivity: float = 2.0,
        eps: float = 1e-6,
        use_residual: bool = True,
        clamp_output: bool = False,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.dt = float(dt)
        self.mode = mode
        self.fractional = bool(fractional)
        self.alpha = float(alpha)
        self.anisotropic = bool(anisotropic)
        self.max_diffusivity = float(max_diffusivity)
        self.eps = float(eps)
        self.use_residual = bool(use_residual)
        self.clamp_output = bool(clamp_output)

        base = torch.full((self.channels,), diffusivity)

        if learnable:
            self.raw_diff = nn.Parameter(softplus_inverse(base))
        else:
            self.register_buffer("diff", base)

        if self.anisotropic:
            self.tensor_net = nn.Sequential(
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=1),
            )

        self.stability = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def get_diff(self) -> torch.Tensor:
        if hasattr(self, "raw_diff"):
            d = F.softplus(self.raw_diff) + self.eps
        else:
            d = self.diff

        d = torch.clamp(d, 0.0, self.max_diffusivity)
        return d.view(1, self.channels, 1, 1)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return (
            -4.0 * x
            + torch.roll(x, 1, -2)
            + torch.roll(x, -1, -2)
            + torch.roll(x, 1, -1)
            + torch.roll(x, -1, -1)
        )

    def spectral_step(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        xf = torch.fft.rfft2(x, norm="ortho")

        ky = torch.fft.fftfreq(h, device=x.device).view(h, 1)
        kx = torch.fft.rfftfreq(w, device=x.device).view(1, -1)

        k2 = kx**2 + ky**2
        k2 = k2.clamp_min(self.eps)

        if self.fractional:
            kpow = k2 ** (self.alpha / 2.0)
        else:
            kpow = k2

        kpow = kpow.unsqueeze(0).unsqueeze(0)

        d = d.view(1, c, 1, 1)

        decay = torch.exp(-self.dt * d * kpow)

        xf = xf * decay

        out = torch.fft.irfft2(xf, s=(h, w), norm="ortho")
        return out

    def anisotropic_step(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        v = self.tensor_net(x)
        vx, vy = torch.chunk(v, 2, dim=1)

        dx = 0.5 * (torch.roll(x, -1, -1) - torch.roll(x, 1, -1))
        dy = 0.5 * (torch.roll(x, -1, -2) - torch.roll(x, 1, -2))

        return x + self.dt * d * (vx * dx + vy * dy)

    def explicit_step(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return x + self.dt * d * self.laplacian(x)

    def implicit_step(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (x + self.dt * d * self.laplacian(x)) / (1.0 + self.dt * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.get_diff()

        if self.mode == "spectral":
            out = self.spectral_step(x, d)
        elif self.mode == "implicit":
            out = self.implicit_step(x, d)
        else:
            out = self.explicit_step(x, d)

        if self.anisotropic:
            out = self.anisotropic_step(out, d)

        if self.use_residual:
            out = self.residual_scale * (out - x)

        s = torch.clamp(self.stability, 0.0, 1.5)
        out = s * out

        if self.clamp_output:
            out = torch.clamp(out, -10.0, 10.0)

        return out