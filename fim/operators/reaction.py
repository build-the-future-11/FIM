from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ReactionOperator(nn.Module):
    """
    Reaction term: pointwise nonlinear source/sink dynamics.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, F_: torch.Tensor) -> torch.Tensor:
        return self.net(F_)