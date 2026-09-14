from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BenchmarkMetrics:
    name: str
    state_shape: Tuple[int, ...]
    horizon: int
    is_non_markovian: bool
    stochastic: bool
    dt: float
    dimension: int


class BenchmarkBase(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def default_steps(self) -> int:
        """Number of integration steps used by the default protocol."""

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of one simulated state, excluding the batch dimension."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of modeled state dimensions."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Integration time step."""

    @property
    def stochastic(self) -> bool:
        return False

    @abstractmethod
    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Sample a protocol-valid initial state."""

    @abstractmethod
    def step(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Advance a batch of states by one integration step."""

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        traj = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            traj.append(x)
        return torch.stack(traj, dim=1)

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        return {
            "name": type(self).__name__,
            "state_shape": self.state_shape,
            "horizon": int(self.default_steps),
            "is_non_markovian": False,
            "stochastic": bool(self.stochastic),
            "dt": float(self.dt),
            "dimension": int(self.dimension),
        }
