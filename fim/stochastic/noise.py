from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 1.0, clip: float | None = None) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        if self.clip is not None:
            noise = torch.clamp(noise, -self.clip, self.clip)
        return x + noise


class ColoredNoise(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-6,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            return x + torch.randn_like(x)

        dims = tuple(range(1, x.ndim))

        X = torch.fft.fftn(x, dim=dims, norm="ortho")

        shape = x.shape[1:]
        freqs = [
            torch.fft.fftfreq(n, device=x.device, dtype=x.dtype)
            for n in shape
        ]

        grid = torch.meshgrid(*freqs, indexing="ij")
        r2 = sum(g * g for g in grid)
        r = torch.sqrt(r2 + self.eps)

        filt = torch.pow(r, -self.beta / 2.0)

        while filt.ndim < X.ndim:
            filt = filt.unsqueeze(0)

        Y = X * filt.to(dtype=X.dtype)
        y = torch.fft.ifftn(Y, dim=dims, norm="ortho").real

        if self.normalize:
            std = y.std(dim=dims, keepdim=True).clamp_min(1e-6)
            y = y / std

        return x + y


class PoissonNoise(nn.Module):
    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(self.rate * x.abs(), min=1e-6)
        noise = torch.poisson(lam) - lam
        return x + noise


class HeteroscedasticNoise(nn.Module):
    def __init__(
        self,
        base_sigma: float = 0.1,
        scale: float = 1.0,
        clip: float | None = None,
    ) -> None:
        super().__init__()
        self.base_sigma = float(base_sigma)
        self.scale = float(scale)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.base_sigma + self.scale * x.abs()
        noise = sigma * torch.randn_like(x)

        if self.clip is not None:
            noise = torch.clamp(noise, -self.clip, self.clip)

        return x + noise


class CompositeNoise(nn.Module):
    def __init__(self, *noises: nn.Module) -> None:
        super().__init__()
        self.noises = nn.ModuleList(noises)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for noise in self.noises:
            x = noise(x)
        return x