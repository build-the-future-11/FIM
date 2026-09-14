from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BenchmarkBase


@dataclass
class Lorenz96Config:
    dimension: int = 32
    forcing: float = 8.0
    dt: float = 0.01
    steps: int = 100
    burn_in: int = 500
    process_noise: float = 0.01
    observation_noise: float = 0.02
    stability_clip: float | None = 20.0
    prediction_horizon: int = 5
    multi_horizon: bool = False
    mask_prob: float = 0.3
    dropout_prob: float = 0.1
    normalize: bool = True


class Lorenz96Benchmark(BenchmarkBase):
    def __init__(self, config: Lorenz96Config) -> None:
        super().__init__()
        if config.dimension < 4:
            raise ValueError("Lorenz-96 dimension must be at least 4.")
        self.config = config
        self.forcing = nn.Parameter(torch.tensor(float(config.forcing)))
        self.register_buffer("running_mean", torch.zeros(config.dimension))
        self.register_buffer("running_std", torch.ones(config.dimension))
        self.register_buffer("norm_initialized", torch.tensor(0, dtype=torch.long))
        self._dimension = int(config.dimension)

    @property
    def default_steps(self) -> int:
        return int(self.config.steps)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (int(self.config.dimension),)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dt(self) -> float:
        return float(self.config.dt)

    @property
    def stochastic(self) -> bool:
        return float(self.config.process_noise) > 0.0

    def rhs(self, x: torch.Tensor) -> torch.Tensor:
        xp1 = torch.roll(x, -1, dims=-1)
        xm1 = torch.roll(x, 1, dims=-1)
        xm2 = torch.roll(x, 2, dims=-1)
        return (xp1 - xm2) * xm1 - x + self.forcing

    def step(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt

        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * dt * k1)
        k3 = self.rhs(x + 0.5 * dt * k2)
        k4 = self.rhs(x + dt * k3)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if self.config.process_noise > 0.0:
            x_next = x_next + self.config.process_noise * torch.randn_like(x_next)

        if self.config.stability_clip is not None:
            x_next = torch.clamp(x_next, -self.config.stability_clip, self.config.stability_clip)

        return x_next

    def burn_in(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.config.burn_in):
            x = self.step(x)
        return x

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        x = torch.randn(batch_size, self.config.dimension, device=device)
        return self.burn_in(x)

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        traj = torch.empty(
            x0.shape[0],
            steps + 1,
            x0.shape[-1],
            device=x0.device,
            dtype=x0.dtype,
        )
        traj[:, 0] = x0
        x = x0
        for t in range(steps):
            x = self.step(x)
            traj[:, t + 1] = x

        if self.config.observation_noise > 0.0:
            traj = traj + self.config.observation_noise * torch.randn_like(traj)

        return traj

    def _update_normalization(self, traj: torch.Tensor) -> None:
        if self.config.normalize and self.norm_initialized.item() == 0:
            mean = traj.mean(dim=(0, 1))
            std = traj.std(dim=(0, 1)).clamp_min(1e-6)
            self.running_mean.copy_(mean)
            self.running_std.copy_(std)
            self.norm_initialized.fill_(1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize:
            return x
        return (x - self.running_mean) / self.running_std

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x0 = self.sample_initial_state(batch_size, device=device)
            traj = self.rollout(x0, steps)

        self._update_normalization(traj)

        horizon = max(1, int(self.config.prediction_horizon))

        if self.config.multi_horizon:
            x = traj[:, :-horizon]
            y = torch.stack(
                [traj[:, i : i + x.shape[1]] for i in range(1, horizon + 1)],
                dim=2,
            )
        else:
            x = traj[:, :-horizon]
            y = traj[:, horizon:]

        x = self._normalize(x)
        y = self._normalize(y)

        if self.config.mask_prob > 0.0:
            mask = (torch.rand_like(x) > self.config.mask_prob).to(x.dtype)
            x = x * mask

        if self.config.dropout_prob > 0.0:
            drop_mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.config.dropout_prob).to(x.dtype)
            x = x * drop_mask

        return x.contiguous(), y.contiguous()

    def metrics(self) -> dict:
        return {
            "name": "Lorenz96Benchmark",
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": False,
            "stochastic": self.stochastic,
            "dt": self.dt,
            "dimension": self.dimension,
            "forcing": float(self.forcing.item()),
            "burn_in": int(self.config.burn_in),
            "prediction_horizon": int(self.config.prediction_horizon),
            "multi_horizon": bool(self.config.multi_horizon),
        }