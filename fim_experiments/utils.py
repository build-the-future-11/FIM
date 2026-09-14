from __future__ import annotations

import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5