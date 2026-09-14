from __future__ import annotations

from typing import Optional

import torch


def _gl_weights(alpha: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.empty(n, device=device, dtype=dtype)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (-(alpha - (k - 1)) / k)
    return w


def grunwald_letnikov_weights(
    alpha: float,
    order: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if order < 1:
        raise ValueError("order must be positive")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    return _gl_weights(alpha, order, device, dtype)


def fractional_difference(
    history: torch.Tensor,
    alpha: float,
    dt: float = 1.0,
) -> torch.Tensor:
    if history.ndim < 2:
        raise ValueError("history must include a time dimension")

    T = history.shape[-2]

    weights = grunwald_letnikov_weights(
        alpha,
        T,
        device=history.device,
        dtype=history.dtype,
    )

    scale = (dt ** (-alpha))
    weights = weights * scale

    flipped = torch.flip(history, dims=[-2])

    view_shape = [1] * history.ndim
    view_shape[-2] = T
    weights = weights.view(*view_shape)

    return torch.sum(flipped * weights, dim=-2)


class FractionalMemory:
    def __init__(
        self,
        alpha: float,
        max_history: int = 128,
        dt: float = 1.0,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = float(alpha)
        self.max_history = int(max_history)
        self.dt = float(dt)

        self._history: Optional[torch.Tensor] = None

    def push(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()

        if self._history is None:
            self._history = x.unsqueeze(-2)
        else:
            self._history = torch.cat([self._history, x.unsqueeze(-2)], dim=-2)

        if self._history.shape[-2] > self.max_history:
            self._history = self._history[:, :, :, :, -self.max_history:]

        return fractional_difference(self._history, self.alpha, self.dt)

    def reset(self) -> None:
        self._history = None

    @property
    def history_length(self) -> int:
        if self._history is None:
            return 0
        return self._history.shape[-2]