from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch

from physics.base import BenchmarkBase


@dataclass
class LevyStableConfig:
    alpha: float = 1.5
    beta: float = 0.0
    scale: float = 1.0
    loc: float = 0.0
    dim: int = 1
    steps: int = 100
    dt: float = 1.0
    stability_clip: float | None = 50.0
    normalize: bool = False
    tempering: float | None = None


class LevyStableProcess(BenchmarkBase):
    def __init__(self, config: LevyStableConfig) -> None:
        super().__init__()
        self.config = config
        if not (0.0 < config.alpha <= 2.0):
            raise ValueError("alpha must be in (0, 2]")
        if not (-1.0 <= config.beta <= 1.0):
            raise ValueError("beta must be in [-1, 1]")
        if config.scale <= 0:
            raise ValueError("scale must be > 0")
        self._dimension = int(config.dim)

    @property
    def default_steps(self) -> int:
        return int(self.config.steps)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (int(self.config.dim),)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dt(self) -> float:
        return float(self.config.dt)

    @property
    def stochastic(self) -> bool:
        return True

    def _sample_cms(self, shape, device, dtype) -> torch.Tensor:
        a = self.config.alpha
        b = self.config.beta
        eps = 1e-8

        if abs(a - 2.0) < 1e-6:
            return torch.randn(shape, device=device, dtype=dtype) * math.sqrt(2.0)

        U = torch.empty(shape, device=device, dtype=dtype).uniform_(-math.pi / 2, math.pi / 2)
        W = torch.empty(shape, device=device, dtype=dtype).exponential_()

        if abs(a - 1.0) > eps:
            tan_term = math.tan(math.pi * a / 2)
            B = math.atan(b * tan_term) / a
            S = (1 + (b * tan_term) ** 2) ** (1 / (2 * a))
            num = torch.sin(a * (U + B))
            den = torch.clamp(torch.cos(U), min=eps) ** (1 / a)
            frac = num / den
            inner = torch.clamp(
                torch.cos(U - a * (U + B)) / torch.clamp(W, min=eps),
                min=eps,
            )
            X = S * frac * (inner ** ((1 - a) / a))
        else:
            Uc = torch.clamp(U, -math.pi / 2 + eps, math.pi / 2 - eps)
            num = (math.pi / 2 + b * Uc) * torch.tan(Uc)
            den = torch.clamp(math.pi / 2 + b * Uc, min=eps)
            log_term = torch.log(torch.clamp((math.pi / 2 * W * torch.cos(Uc)) / den, min=eps))
            X = (2 / math.pi) * (num - b * log_term)

        return X

    def sample_increment(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = torch.device(device)
        shape = (batch_size, self.config.dim)
        X = self._sample_cms(shape, device, dtype)

        dt_scale = self.config.dt ** (1.0 / self.config.alpha)
        X = dt_scale * X
        X = self.config.scale * X + self.config.loc

        if self.config.tempering is not None:
            X = X * torch.exp(-self.config.tempering * X.abs())

        if self.config.stability_clip is not None:
            X = torch.clamp(X, -self.config.stability_clip, self.config.stability_clip)

        if self.config.normalize:
            X = X / (X.abs().mean(dim=-1, keepdim=True) + 1e-6)

        return X

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.sample_increment(state.shape[0], device=state.device, dtype=state.dtype)

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = torch.device(device)
        return torch.zeros(batch_size, self.config.dim, device=device)

    def sample_path(
        self,
        steps: Optional[int] = None,
        batch_size: int = 1,
        x0: Optional[torch.Tensor] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        device = torch.device(device)

        if x0 is None:
            x0 = torch.zeros(batch_size, self.config.dim, device=device, dtype=dtype)
        else:
            x0 = x0.to(device=device, dtype=dtype)

        increments = self.sample_increment(batch_size * steps, device=device, dtype=dtype)
        increments = increments.view(batch_size, steps, self.config.dim)

        traj = torch.zeros(batch_size, steps + 1, self.config.dim, device=device, dtype=dtype)
        traj[:, 0] = x0
        traj[:, 1:] = torch.cumsum(increments, dim=1) + x0.unsqueeze(1)
        return traj

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        return self.sample_path(steps=steps, batch_size=x0.shape[0], x0=x0, device=x0.device, dtype=x0.dtype)

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.sample_path(steps=steps, batch_size=batch_size, device=device)
        return traj[:, :-1], traj[:, 1:]

    def sample_moments(
        self,
        num_samples: int = 10000,
        device: str | torch.device = "cpu",
    ) -> dict:
        samples = self.sample_increment(num_samples, device=device)
        median = samples.median(dim=0).values
        mad = (samples - median).abs().median(dim=0).values
        return {
            "mean": samples.mean(dim=0),
            "std": samples.std(dim=0),
            "median": median,
            "mad": mad,
            "max_abs": samples.abs().max(dim=0).values,
        }

    def tail_index_estimate(
        self,
        num_samples: int = 20000,
        k_frac: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        samples = self.sample_increment(num_samples, device=device).abs().flatten()
        samples = samples[samples > 1e-8]
        samples, _ = torch.sort(samples)
        k = max(2, int(len(samples) * k_frac))
        tail = samples[-k:]
        x_k = tail[0]
        hill = torch.mean(torch.log(tail / (x_k + 1e-8)))
        return 1.0 / (hill + 1e-8)

    def characteristic_function(self, t: torch.Tensor) -> torch.Tensor:
        a = self.config.alpha
        b = self.config.beta
        c = self.config.scale
        mu = self.config.loc
        t = t.to(dtype=torch.float32)

        if abs(a - 1.0) > 1e-6:
            exponent = (
                mu * t
                - (c**a)
                * torch.abs(t) ** a
                * (1 - 1j * b * torch.sign(t) * math.tan(math.pi * a / 2))
            )
        else:
            exponent = (
                mu * t
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

        return torch.exp(1j * exponent)

    def metrics(self) -> dict:
        return {
            "name": "LevyStableProcess",
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": False,
            "stochastic": True,
            "dt": self.dt,
            "dimension": self.dimension,
            "alpha": float(self.config.alpha),
            "beta": float(self.config.beta),
            "scale": float(self.config.scale),
            "tempering": None if self.config.tempering is None else float(self.config.tempering),
            "normalize": bool(self.config.normalize),
        }