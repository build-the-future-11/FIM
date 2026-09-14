from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch


@dataclass
class LevyStableConfig:
    alpha: float = 1.5
    beta: float = 0.0
    scale: float = 1.0
    loc: float = 0.0
    dim: int = 1
    dt: float = 1.0

    stability_clip: float | None = 50.0
    tempering: float = 0.0
    correlated: bool = False
    normalize: bool = False


class LevyStableProcess:
    def __init__(self, config: LevyStableConfig) -> None:
        if not (0.0 < config.alpha <= 2.0):
            raise ValueError("alpha must be in (0, 2]")
        if not (-1.0 <= config.beta <= 1.0):
            raise ValueError("beta must be in [-1, 1]")
        if config.scale <= 0.0:
            raise ValueError("scale must be positive")

        self.config = config

        self.cov = torch.eye(config.dim)
        self._chol: Optional[torch.Tensor] = None

    def set_covariance(self, cov: torch.Tensor):
        if cov.shape[-1] != self.config.dim:
            raise ValueError("Covariance dimension mismatch")
        self.cov = cov
        self._chol = None

    def _get_cholesky(self, device, dtype):
        if self._chol is None or self._chol.device != device or self._chol.dtype != dtype:
            self._chol = torch.linalg.cholesky(
                self.cov.to(device=device, dtype=dtype)
            )
        return self._chol

    def _cms_sample(self, shape, device, dtype):
        a = self.config.alpha
        b = self.config.beta
        eps = 1e-8

        if abs(a - 2.0) < 1e-6:
            return torch.randn(*shape, device=device, dtype=dtype) * math.sqrt(2.0)

        U = torch.empty(*shape, device=device, dtype=dtype).uniform_(
            -math.pi / 2, math.pi / 2
        )
        W = torch.empty(*shape, device=device, dtype=dtype).exponential_()

        if abs(a - 1.0) > 1e-6:
            tan_term = math.tan(math.pi * a / 2)
            B = math.atan(b * tan_term)
            phi = B / a
            S = (1 + (b * tan_term) ** 2) ** (1 / (2 * a))

            cosU = torch.clamp(torch.cos(U), min=eps)
            part1 = S * torch.sin(a * (U + phi)) / (cosU ** (1 / a))

            cos_term = torch.clamp(torch.cos(U - a * (U + phi)), min=eps)
            part2 = (cos_term / torch.clamp(W, min=eps)) ** ((1 - a) / a)

            X = part1 * part2
        else:
            Uc = torch.clamp(U, -math.pi / 2 + eps, math.pi / 2 - eps)

            num = (math.pi / 2 + b * Uc) * torch.tan(Uc)
            denom = torch.clamp(math.pi / 2 + b * Uc, min=eps)

            log_term = torch.log(
                torch.clamp((math.pi / 2 * W * torch.cos(Uc)) / denom, min=eps)
            )

            X = (2 / math.pi) * (num - b * log_term)

        return X

    def _apply_correlation(self, X: torch.Tensor) -> torch.Tensor:
        if not self.config.correlated:
            return X
        L = self._get_cholesky(X.device, X.dtype)
        return X @ L.T

    def _apply_tempering(self, X: torch.Tensor) -> torch.Tensor:
        lam = self.config.tempering
        if lam <= 0.0:
            return X
        return X * torch.exp(-lam * X.abs())

    def sample_increment(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = torch.device(device)

        X = self._cms_sample((batch_size, self.config.dim), device, dtype)

        X = self._apply_correlation(X)

        dt_scale = self.config.dt ** (1.0 / self.config.alpha)
        X = X * dt_scale

        X = self._apply_tempering(X)

        X = self.config.scale * X + self.config.loc

        if self.config.stability_clip is not None:
            X = torch.clamp(X, -self.config.stability_clip, self.config.stability_clip)

        if self.config.normalize:
            X = X / (X.abs().mean(dim=-1, keepdim=True) + 1e-6)

        return X

    def sample_path(
        self,
        steps: int,
        batch_size: int = 1,
        x0: Optional[torch.Tensor] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = torch.device(device)

        if x0 is None:
            x0 = torch.zeros(batch_size, self.config.dim, device=device, dtype=dtype)
        else:
            x0 = x0.to(device=device, dtype=dtype)

        traj = torch.zeros(
            batch_size,
            steps + 1,
            self.config.dim,
            device=device,
            dtype=dtype,
        )

        traj[:, 0] = x0

        x = x0
        for t in range(steps):
            inc = self.sample_increment(batch_size, device=device, dtype=dtype)
            x = x + inc
            traj[:, t + 1] = x

        return traj

    def generate_batch(
        self,
        batch_size: int,
        steps: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.sample_path(
            steps=steps,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return traj[:, :-1], traj[:, 1:]

    def characteristic_function(self, t: torch.Tensor) -> torch.Tensor:
        a = self.config.alpha
        b = self.config.beta
        c = self.config.scale
        mu = self.config.loc

        t = t.to(dtype=torch.float32)

        if abs(a - 1.0) > 1e-6:
            exponent = (
                1j * mu * t
                - (c ** a)
                * torch.abs(t) ** a
                * (1 - 1j * b * torch.sign(t) * math.tan(math.pi * a / 2))
            )
        else:
            exponent = (
                1j * mu * t
                - c
                * torch.abs(t)
                * (
                    1
                    + 1j
                    * b
                    * (2 / math.pi)
                    * torch.sign(t)
                    * torch.log(torch.abs(t) + 1e-8)
                )
            )

        return torch.exp(exponent)

    def log_characteristic(self, t: torch.Tensor) -> torch.Tensor:
        a = self.config.alpha
        b = self.config.beta
        c = self.config.scale
        mu = self.config.loc

        t = t.to(dtype=torch.float32)

        if abs(a - 1.0) > 1e-6:
            return (
                1j * mu * t
                - (c ** a)
                * torch.abs(t) ** a
                * (1 - 1j * b * torch.sign(t) * math.tan(math.pi * a / 2))
            )
        else:
            return (
                1j * mu * t
                - c
                * torch.abs(t)
                * (
                    1
                    + 1j
                    * b
                    * (2 / math.pi)
                    * torch.sign(t)
                    * torch.log(torch.abs(t) + 1e-8)
                )
            )


__all__ = [
    "LevyStableConfig",
    "LevyStableProcess",
]