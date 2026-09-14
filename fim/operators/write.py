from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_groups(channels: int, max_groups: int = 8) -> int:
    g = max(1, min(max_groups, channels))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class LocalWrite(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        hidden_channels: Optional[int] = None,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()

        hidden = hidden_channels or channels
        padding = kernel_size // 2

        self.use_residual_gate = use_residual_gate

        g = _make_groups(channels)

        self.norm_z = nn.GroupNorm(g, channels)
        self.norm_f = nn.GroupNorm(g, channels)

        self.write_map = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

        self.delta_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        z: torch.Tensor,
        F: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        z_n = self.norm_z(z)
        f_n = self.norm_f(F)

        inp = torch.cat([z_n, f_n], dim=1)

        delta = self.write_map(inp)
        delta = torch.tanh(delta) * torch.clamp(self.delta_scale, 0.1, 10.0)

        gate_logits = self.gate(inp)
        temp = torch.clamp(self.gate_temperature, 0.1, 5.0)
        gate = torch.sigmoid(gate_logits / temp)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            gate = gate * mask.to(dtype=gate.dtype, device=gate.device)

        if self.use_residual_gate:
            scale = torch.clamp(self.residual_scale, 0.0, 2.0)
            out = F + scale * gate * delta
        else:
            out = F + gate * delta

        return out