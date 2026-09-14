from __future__ import annotations

import math
from typing import Optional

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        step = max(step, 0)

        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)

        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = cosine

        elif schedule == "linear":
            factor = 1.0 - progress

        elif schedule == "constant":
            factor = 1.0

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch=last_epoch,
    )