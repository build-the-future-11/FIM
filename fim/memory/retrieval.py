from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trace_bank import TraceBank


@dataclass
class RetrievalOutput:
    context: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    similarities: torch.Tensor
    regularization: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __iter__(self):
        yield self.context
        yield self.weights
        yield self.indices
        yield self.similarities


class MemoryRetrieval(nn.Module):
    def __init__(
        self,
        query_dim: int,
        value_dim: int,
        temperature: float = 0.2,
        top_k: Optional[int] = None,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
        entropy_reg_weight: float = 0.0,
        sparsity_reg_weight: float = 0.0,
        coverage_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.query_dim = int(query_dim)
        self.value_dim = int(value_dim)
        self.top_k = top_k

        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.salience_scale = nn.Parameter(torch.tensor(float(salience_scale)))
        self.recency_scale = nn.Parameter(torch.tensor(float(recency_scale)))

        self.entropy_reg_weight = float(entropy_reg_weight)
        self.sparsity_reg_weight = float(sparsity_reg_weight)
        self.coverage_reg_weight = float(coverage_reg_weight)

        self.query_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.key_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.value_proj = nn.Linear(value_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, value_dim, bias=False)

    def forward(self, query: torch.Tensor, bank: TraceBank) -> RetrievalOutput:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        query = query.reshape(query.shape[0], -1)

        keys, values, saliences, timestamps = bank.as_tensors()

        batch = query.shape[0]
        device = query.device
        dtype = query.dtype

        if keys.numel() == 0:
            zero = torch.zeros(batch, self.value_dim, device=device, dtype=dtype)
            empty_2d = torch.zeros(batch, 0, device=device, dtype=dtype)
            empty_idx = torch.empty(batch, 0, device=device, dtype=torch.long)
            return RetrievalOutput(
                context=zero,
                weights=empty_2d,
                indices=empty_idx,
                similarities=empty_2d,
                regularization={
                    "entropy": torch.tensor(0.0, device=device, dtype=dtype),
                    "sparsity": torch.tensor(0.0, device=device, dtype=dtype),
                    "coverage": torch.tensor(0.0, device=device, dtype=dtype),
                },
            )

        keys = keys.to(device=device, dtype=dtype)
        values = values.to(device=device, dtype=dtype)
        saliences = saliences.to(device=device, dtype=dtype).view(-1)
        timestamps = timestamps.to(device=device, dtype=dtype).view(-1)

        q = self.query_proj(query)
        k = self.key_proj(keys)
        v = self.value_proj(values)

        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        logits = torch.matmul(q, k.t())

        sal_scale = torch.clamp(self.salience_scale, 0.0, 10.0)
        rec_scale = torch.clamp(self.recency_scale, 0.0, 10.0)

        if saliences.numel() > 0:
            logits = logits + sal_scale * saliences.unsqueeze(0)

        if timestamps.numel() > 0:
            age = timestamps.max() - timestamps
            logits = logits - rec_scale * age.unsqueeze(0)

        temp = torch.clamp(self.temperature, 0.05, 5.0)

        if self.top_k is not None and 0 < self.top_k < logits.shape[-1]:
            top_logits, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
            weights = torch.softmax(top_logits / temp, dim=-1)

            gathered = v.unsqueeze(0).expand(batch, -1, -1)
            gathered = torch.gather(
                gathered,
                dim=1,
                index=top_idx.unsqueeze(-1).expand(-1, -1, self.value_dim),
            )

            context = torch.sum(weights.unsqueeze(-1) * gathered, dim=1)
            context = self.out_proj(context)

            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            sparsity = (weights**2).sum(dim=-1).mean()
            coverage = weights.amax(dim=-1).mean()

            return RetrievalOutput(
                context=context,
                weights=weights,
                indices=top_idx,
                similarities=top_logits,
                regularization={
                    "entropy": entropy * self.entropy_reg_weight,
                    "sparsity": sparsity * self.sparsity_reg_weight,
                    "coverage": coverage * self.coverage_reg_weight,
                },
            )

        weights = torch.softmax(logits / temp, dim=-1)
        context = torch.matmul(weights, v)
        context = self.out_proj(context)
        indices = torch.arange(v.shape[0], device=device).unsqueeze(0).expand(batch, -1)

        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        sparsity = (weights**2).sum(dim=-1).mean()
        coverage = weights.amax(dim=-1).mean()

        return RetrievalOutput(
            context=context,
            weights=weights,
            indices=indices,
            similarities=logits,
            regularization={
                "entropy": entropy * self.entropy_reg_weight,
                "sparsity": sparsity * self.sparsity_reg_weight,
                "coverage": coverage * self.coverage_reg_weight,
            },
        )