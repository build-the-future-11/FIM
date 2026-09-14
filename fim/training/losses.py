from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    prediction: torch.Tensor
    memory: torch.Tensor
    stability: torch.Tensor
    sparsity: torch.Tensor
    retrieval: torch.Tensor
    consistency: torch.Tensor


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=x.device, dtype=x.dtype)


def prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown prediction loss kind: {kind}")


def sparsity_loss(mask: torch.Tensor) -> torch.Tensor:
    if mask is None or mask.numel() == 0:
        return torch.zeros((), device=mask.device if mask is not None else "cpu")

    p = mask.float().mean().clamp(1e-6, 1 - 1e-6)
    return p * torch.log(p) + (1 - p) * torch.log(1 - p)


def memory_usage_loss(
    mask: torch.Tensor,
    target_usage: float = 0.1,
) -> torch.Tensor:
    if mask is None or mask.numel() == 0:
        return torch.zeros((), device=mask.device if mask is not None else "cpu")

    usage = mask.float().mean()
    return (usage - target_usage).abs()


def stability_loss(state: torch.Tensor) -> torch.Tensor:
    if state is None or state.numel() == 0:
        return torch.zeros((), device=state.device if state is not None else "cpu")

    mean = state.mean()
    std = state.std(unbiased=False)
    return mean.pow(2) + (std - 1.0).pow(2)


def retrieval_loss(
    retrieved: Optional[torch.Tensor],
    target_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if retrieved is None or retrieved.numel() == 0:
        return torch.zeros((), device=target_context.device if target_context is not None else "cpu")

    if target_context is None:
        return -retrieved.norm(dim=-1).mean()

    pos = F.mse_loss(retrieved, target_context)

    perm = torch.randperm(target_context.size(0), device=target_context.device)
    shuffled = target_context[perm]

    neg = -F.cosine_similarity(retrieved, shuffled, dim=-1).mean()

    return pos + 0.1 * neg


def consistency_loss(
    pre_state: torch.Tensor,
    post_state: torch.Tensor,
) -> torch.Tensor:
    if pre_state is None or post_state is None:
        return torch.zeros((), device=pre_state.device if pre_state is not None else "cpu")

    return F.smooth_l1_loss(pre_state, post_state)


def fim_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    stored_mask: Optional[torch.Tensor],
    retrieval_context: Optional[torch.Tensor],
    pre_retrieval_state: Optional[torch.Tensor],
    post_retrieval_state: Optional[torch.Tensor],
    *,
    w_pred: float = 1.0,
    w_mem: float = 0.05,
    w_stab: float = 0.05,
    w_sparse: float = 0.05,
    w_retr: float = 0.1,
    w_cons: float = 0.05,
    pred_kind: str = "mse",
) -> LossOutput:

    lp = prediction_loss(pred, target, kind=pred_kind)

    l_sparse = sparsity_loss(stored_mask)
    l_mem = memory_usage_loss(stored_mask)

    l_stab = stability_loss(state)
    l_retr = retrieval_loss(retrieval_context)
    l_cons = consistency_loss(pre_retrieval_state, post_retrieval_state)

    total = (
        w_pred * lp
        + w_mem * l_mem
        + w_sparse * l_sparse
        + w_stab * l_stab
        + w_retr * l_retr
        + w_cons * l_cons
    )

    return LossOutput(
        total=total,
        prediction=lp,
        memory=l_mem,
        stability=l_stab,
        sparsity=l_sparse,
        retrieval=l_retr,
        consistency=l_cons,
    )