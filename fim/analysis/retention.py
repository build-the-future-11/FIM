from __future__ import annotations

import math
from typing import Iterable

import torch


def decay_curve(response: torch.Tensor) -> torch.Tensor:
    """Normalize a response curve so the first value is 1."""
    if response.ndim != 1:
        raise ValueError("response must be a 1D tensor")
    return response / response[0].clamp_min(1e-8)


def estimate_exponential_decay(times: torch.Tensor, response: torch.Tensor, eps: float = 1e-8) -> float:
    """Fit log(response) ≈ a - lambda * t; returns lambda."""
    if times.shape != response.shape:
        raise ValueError("times and response must have same shape")
    y = torch.log(response.clamp_min(eps))
    x = times.float()
    x_mean = x.mean()
    y_mean = y.mean()
    slope = torch.sum((x - x_mean) * (y - y_mean)) / torch.sum((x - x_mean) ** 2).clamp_min(eps)
    return float(-slope.item())


def estimate_power_law_decay(times: torch.Tensor, response: torch.Tensor, eps: float = 1e-8) -> float:
    """Fit log(response) ≈ a - alpha * log(t); returns alpha."""
    if times.shape != response.shape:
        raise ValueError("times and response must have same shape")
    x = torch.log(times.clamp_min(eps))
    y = torch.log(response.clamp_min(eps))
    x_mean = x.mean()
    y_mean = y.mean()
    slope = torch.sum((x - x_mean) * (y - y_mean)) / torch.sum((x - x_mean) ** 2).clamp_min(eps)
    return float(-slope.item())


def retention_auc(times: torch.Tensor, response: torch.Tensor) -> float:
    """Area under a retention curve using the trapezoid rule."""
    if times.shape != response.shape:
        raise ValueError("times and response must have same shape")
    return float(torch.trapz(response.float(), times.float()).item())
