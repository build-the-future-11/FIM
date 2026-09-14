from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


DriftFn = Callable[[torch.Tensor, float], torch.Tensor]
DiffusionFn = Callable[[torch.Tensor, float], torch.Tensor]
DiffusionGradFn = Callable[[torch.Tensor, float], torch.Tensor]
NoiseFn = Callable[[torch.Tensor, float], torch.Tensor]


@dataclass
class SDEConfig:
    dt: float = 0.01
    steps: int = 100
    stability_clip: float | None = 20.0
    noise_scale: float = 1.0


class SDEIntegrator:
    def __init__(
        self,
        config: SDEConfig,
        noise_fn: Optional[NoiseFn] = None,
    ) -> None:
        self.config = config
        self.noise_fn = noise_fn

    def _sample_noise(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        if self.noise_fn is not None:
            return self.noise_fn(x, dt)

        return torch.randn_like(x) * (dt ** 0.5) * self.config.noise_scale

    def euler_maruyama(
        self,
        x0: torch.Tensor,
        drift: DriftFn,
        diffusion: DiffusionFn,
        steps: Optional[int] = None,
        t0: float = 0.0,
    ) -> torch.Tensor:
        steps = steps or self.config.steps
        dt = self.config.dt

        traj = torch.zeros(
            *x0.shape[:1],
            steps + 1,
            *x0.shape[1:],
            device=x0.device,
            dtype=x0.dtype,
        )

        traj[:, 0] = x0

        x = x0
        t = t0

        for i in range(steps):
            f = drift(x, t)
            g = diffusion(x, t)

            dw = self._sample_noise(x, dt)

            x = x + f * dt + g * dw

            if self.config.stability_clip is not None:
                x = torch.clamp(
                    x,
                    -self.config.stability_clip,
                    self.config.stability_clip,
                )

            traj[:, i + 1] = x
            t += dt

        return traj

    def milstein(
        self,
        x0: torch.Tensor,
        drift: DriftFn,
        diffusion: DiffusionFn,
        diffusion_grad: Optional[DiffusionGradFn] = None,
        steps: Optional[int] = None,
        t0: float = 0.0,
    ) -> torch.Tensor:
        steps = steps or self.config.steps
        dt = self.config.dt

        traj = torch.zeros(
            *x0.shape[:1],
            steps + 1,
            *x0.shape[1:],
            device=x0.device,
            dtype=x0.dtype,
        )

        traj[:, 0] = x0

        x = x0
        t = t0

        for i in range(steps):
            f = drift(x, t)
            g = diffusion(x, t)

            dw = self._sample_noise(x, dt)

            correction = 0.0
            if diffusion_grad is not None:
                gp = diffusion_grad(x, t)
                correction = 0.5 * g * gp * (dw**2 - dt)

            x = x + f * dt + g * dw + correction

            if self.config.stability_clip is not None:
                x = torch.clamp(
                    x,
                    -self.config.stability_clip,
                    self.config.stability_clip,
                )

            traj[:, i + 1] = x
            t += dt

        return traj

    def integrate(
        self,
        x0: torch.Tensor,
        drift: DriftFn,
        diffusion: DiffusionFn,
        method: str = "euler",
        diffusion_grad: Optional[DiffusionGradFn] = None,
        steps: Optional[int] = None,
        t0: float = 0.0,
    ) -> torch.Tensor:
        if method == "euler":
            return self.euler_maruyama(x0, drift, diffusion, steps, t0)

        if method == "milstein":
            return self.milstein(x0, drift, diffusion, diffusion_grad, steps, t0)

        raise ValueError(f"Unknown method: {method}")