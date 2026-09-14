from __future__ import annotations

import torch
import torch.nn.functional as F


def laplacian_kernel2d(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    return kernel.view(1, 1, 3, 3)


def apply_laplacian(
    x: torch.Tensor,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Input must be [B, C, H, W]")

    if kernel is None:
        kernel = laplacian_kernel2d(device=x.device, dtype=x.dtype)

    C = x.shape[1]

    kernel = kernel.to(device=x.device, dtype=x.dtype)
    kernel = kernel.expand(C, 1, 3, 3).contiguous()

    return F.conv2d(
        x,
        kernel,
        padding=1,
        groups=C,
    )