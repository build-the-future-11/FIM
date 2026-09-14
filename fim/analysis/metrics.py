from __future__ import annotations

import math
from typing import Iterable

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.mean(target ** 2).clamp_min(eps)
    return mse(pred, target) / denom


def rare_event_recall(pred_events: torch.Tensor, true_events: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Recall on binary event masks."""
    tp = torch.sum((pred_events > 0) & (true_events > 0)).float()
    fn = torch.sum((pred_events <= 0) & (true_events > 0)).float()
    return tp / (tp + fn).clamp_min(eps)


def rare_event_precision(pred_events: torch.Tensor, true_events: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tp = torch.sum((pred_events > 0) & (true_events > 0)).float()
    fp = torch.sum((pred_events > 0) & (true_events <= 0)).float()
    return tp / (tp + fp).clamp_min(eps)


def memory_efficiency_score(recall: float, stored_traces: int) -> float:
    """A compactness-aware score in [0, 1+] depending on recall and storage."""
    return float(recall) / (1.0 + math.log1p(max(int(stored_traces), 0)))
