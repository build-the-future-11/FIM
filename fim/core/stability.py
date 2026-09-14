from __future__ import annotations

from typing import Optional

import torch


def fabric_energy(tensor: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius energy."""
    return tensor.pow(2).sum()


def is_bounded(
    tensor: torch.Tensor,
    *,
    threshold: float,
) -> bool:
    return bool(torch.linalg.norm(tensor).item() <= float(threshold))


def clamp_norm_(tensor: torch.Tensor, max_norm: float, eps: float = 1e-12) -> torch.Tensor:
    """In-place norm clamp."""
    norm = torch.linalg.norm(tensor)
    if norm > max_norm:
        tensor.mul_(max_norm / (norm + eps))
    return tensor


@torch.no_grad()
def estimate_growth_rate(
    sequence: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    eps: float = 1e-12,
) -> float:
    """Estimate average log growth from a sequence of tensors."""
    if len(sequence) < 2:
        return 0.0
    norms = [torch.linalg.norm(x).item() for x in sequence]
    ratios = []
    for a, b in zip(norms[:-1], norms[1:]):
        ratios.append(torch.log(torch.tensor((b + eps) / (a + eps))).item())
    return float(sum(ratios) / len(ratios))
