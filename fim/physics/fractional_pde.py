from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BenchmarkBase


def fractional_weights(
    alpha: float,
    steps: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    normalize: bool = True,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32

    w = torch.zeros(steps, device=device, dtype=dtype)
    w[0] = 1.0
    for k in range(1, steps):
        w[k] = -w[k - 1] * (alpha - (k - 1)) / k

    if normalize:
        w = w / (w.abs().sum() + 1e-8)

    return w


@dataclass
class FractionalConfig:
    alpha: float = 0.8
    diffusivity: float = 0.05
    dt: float = 0.01
    history: int = 64
    steps: int = 100
    stability_clip: float = 10.0
    spectral: bool = True
    normalize_history_weights: bool = True
    init_scale: float = 0.1
    height: int = 32
    width: int = 32
    channels: int = 1


class FractionalDiffusionBenchmark(BenchmarkBase):
    def __init__(self, config: FractionalConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_diffusivity = nn.Parameter(torch.tensor(float(config.diffusivity)))
        self.frac_scale = nn.Parameter(torch.tensor(1.0))
        self._dimension = int(config.channels * config.height * config.width)

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

    def spatial_term(self, u: torch.Tensor) -> torch.Tensor:
        return (
            -4.0 * u
            + torch.roll(u, 1, dims=-2)
            + torch.roll(u, -1, dims=-2)
            + torch.roll(u, 1, dims=-1)
            + torch.roll(u, -1, dims=-1)
        )

    def effective_diffusivity(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        diff = F.softplus(self.raw_diffusivity.to(device=device, dtype=dtype))
        cfl_max = 0.5 / max(self.config.dt, 1e-8)
        return torch.clamp(diff, 0.0, cfl_max)

    def fractional_term(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim < 4:
            raise ValueError("history must have shape [T, B, C, H, W] or similar")
        t = history.shape[0]
        weights = fractional_weights(
            self.config.alpha,
            t,
            device=history.device,
            dtype=history.dtype,
            normalize=self.config.normalize_history_weights,
        )
        weights = weights * (self.config.dt ** (-self.config.alpha))
        weights = weights / (weights.abs().sum() + 1e-8)
        weights = weights.view(t, *([1] * (history.ndim - 1)))
        hist = torch.flip(history, dims=[0])
        return self.frac_scale * torch.sum(hist * weights, dim=0)

    def step(self, history: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(history):
            hist = history
            if hist.ndim != 5:
                raise ValueError("stacked history must have shape [T, B, C, H, W]")
        else:
            if len(history) == 0:
                raise ValueError("history must contain at least one tensor")
            hlen = min(len(history), self.config.history)
            hist = torch.stack(list(history)[-hlen:], dim=0)

        frac_term = self.fractional_term(hist)
        u_t = hist[-1]
        diff = self.effective_diffusivity(u_t.device, u_t.dtype)
        diffusion = diff * self.spatial_term(u_t)
        u_next = u_t + self.config.dt * (frac_term + diffusion)

        if self.config.stability_clip is not None:
            u_next = torch.clamp(u_next, -self.config.stability_clip, self.config.stability_clip)

        return u_next

    def rollout(self, u0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        history = [u0]
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
            hist = torch.stack(history[-self.config.history :], dim=0)
            u = self.step(hist)
            history.append(u)
            if len(history) > self.config.history:
                history.pop(0)
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
            "name": "FractionalDiffusionBenchmark",
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": True,
            "stochastic": False,
            "dt": self.dt,
            "dimension": self.dimension,
            "alpha": float(self.config.alpha),
            "diffusivity": float(F.softplus(self.raw_diffusivity).item()),
            "history": int(self.config.history),
            "spectral": bool(self.config.spectral),
            "stability_clip": float(self.config.stability_clip),
        }