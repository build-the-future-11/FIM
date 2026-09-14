from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def estimate_lyapunov_exponent(
    trajectory_a: List[torch.Tensor],
    trajectory_b: List[torch.Tensor],
    eps: float = 1e-8,
    reduction: str = "mean",
) -> float:
    if len(trajectory_a) != len(trajectory_b):
        raise ValueError("trajectory lengths must match")
    if len(trajectory_a) < 2:
        return 0.0

    dists = []

    for a, b in zip(trajectory_a, trajectory_b):
        if a.shape != b.shape:
            raise RuntimeError("trajectory tensors must have same shape")

        d = torch.linalg.norm(a - b)
        d = torch.clamp(d, min=eps)
        dists.append(d)

    dists = torch.stack(dists)

    log_d = torch.log(dists)

    growth = log_d[1:] - log_d[:-1]

    if reduction == "mean":
        lyap = growth.mean()
    elif reduction == "median":
        lyap = growth.median()
    else:
        raise ValueError("reduction must be 'mean' or 'median'")

    return float(lyap.item())