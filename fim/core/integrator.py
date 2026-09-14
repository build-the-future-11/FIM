from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class VectorField(Protocol):
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


@dataclass
class ExplicitEulerIntegrator:
    dt: float = 1.0

    def step(self, x: torch.Tensor, field: VectorField, *args, **kwargs) -> torch.Tensor:
        return x + self.dt * field(x, *args, **kwargs)


@dataclass
class RungeKutta2Integrator:
    dt: float = 1.0

    def step(self, x: torch.Tensor, field: VectorField, *args, **kwargs) -> torch.Tensor:
        k1 = field(x, *args, **kwargs)
        k2 = field(x + 0.5 * self.dt * k1, *args, **kwargs)
        return x + self.dt * k2
