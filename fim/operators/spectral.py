from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SpectralOperator(nn.Module):
    """
    Learnable spectral filter in Fourier space.

    Input:
        [B, C, H, W]
    Output:
        [B, C, H, W]
    """

    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        freq_w = width // 2 + 1
        self.real = nn.Parameter(torch.ones(channels, height, freq_w))
        self.imag = nn.Parameter(torch.zeros(channels, height, freq_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W] tensor")

        X = torch.fft.rfft2(x, norm="ortho")
        gain = torch.complex(self.real, self.imag).to(dtype=X.dtype, device=X.device)
        Y = X * gain.unsqueeze(0)
        y = torch.fft.irfft2(Y, s=(x.shape[-2], x.shape[-1]), norm="ortho")
        return y