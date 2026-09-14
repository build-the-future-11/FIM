from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BenchmarkBase


def laplacian_periodic(u: torch.Tensor) -> torch.Tensor:
    return (
        -4.0 * u
        + torch.roll(u, 1, dims=-2)
        + torch.roll(u, -1, dims=-2)
        + torch.roll(u, 1, dims=-1)
        + torch.roll(u, -1, dims=-1)
    )


def laplacian_dirichlet(u: torch.Tensor) -> torch.Tensor:
    up = torch.zeros_like(u)
    up[..., 1:-1, 1:-1] = (
        -4.0 * u[..., 1:-1, 1:-1]
        + u[..., :-2, 1:-1]
        + u[..., 2:, 1:-1]
        + u[..., 1:-1, :-2]
        + u[..., 1:-1, 2:]
    )
    return up


@dataclass
class DiffusionConfig:
    diffusivity: float = 0.1
    dt: float = 0.01
    steps: int = 100
    periodic: bool = True
    stability_scale: float = 0.99
    state_clip: float | None = 10.0
    init_scale: float = 1.0
    height: int = 32
    width: int = 32
    channels: int = 1


class DiffusionBenchmark(BenchmarkBase):
    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_diffusivity = nn.Parameter(torch.tensor(float(config.diffusivity)))
        self._dimension = int(config.height * config.width * config.channels)

    @property
    def default_steps(self) -> int:
        return int(self.config.steps)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (int(self.config.channels), int(self.config.height), int(self.config.width))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dt(self) -> float:
        return float(self.config.dt)

    @property
    def stochastic(self) -> bool:
        return False

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        if self.config.periodic:
            return laplacian_periodic(u)
        return laplacian_dirichlet(u)

    def effective_diffusivity(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        diff = F.softplus(self.raw_diffusivity.to(device=device, dtype=dtype))
        cfl_max = self.config.stability_scale / max(4.0 * self.config.dt, 1e-8)
        return torch.clamp(diff, 0.0, cfl_max)

    def step(self, u: torch.Tensor) -> torch.Tensor:
        diff = self.effective_diffusivity(u.device, u.dtype)
        du = diff * self.laplacian(u)
        u_next = u + self.config.dt * du

        if self.config.state_clip is not None:
            u_next = torch.clamp(u_next, -self.config.state_clip, self.config.state_clip)

        return self.config.stability_scale * u_next

    def rollout(self, u0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        traj = torch.zeros(
            u0.shape[0],
            steps + 1,
            *u0.shape[1:],
            device=u0.device,
            dtype=u0.dtype,
        )
        traj[:, 0] = u0
        u = u0
        for t in range(steps):
            u = self.step(u)
            traj[:, t + 1] = u
        return traj

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = torch.device(device)
        return self.config.init_scale * torch.randn(
            batch_size,
            self.config.channels,
            self.config.height,
            self.config.width,
            device=device,
        )

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> dict:
        return {
            "name": "DiffusionBenchmark",
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": False,
            "stochastic": False,
            "dt": self.dt,
            "dimension": self.dimension,
            "diffusivity": float(F.softplus(self.raw_diffusivity).item()),
            "periodic": bool(self.config.periodic),
            "stability_scale": float(self.config.stability_scale),
        }