from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .trace_bank import TraceBank


class MemoryConsolidator(nn.Module):
    def __init__(
        self,
        salience_threshold: float = 0.5,
        min_store_probability: float = 0.0,
        decay: float = 0.0,
        novelty_bias: float = 0.0,
        layer_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.salience_threshold = nn.Parameter(torch.tensor(float(salience_threshold)))
        self.min_store_probability = nn.Parameter(torch.tensor(float(min_store_probability)))
        self.decay = nn.Parameter(torch.tensor(float(decay)))
        self.novelty_bias = nn.Parameter(torch.tensor(float(novelty_bias)))
        self.layer_decay = nn.Parameter(torch.tensor(float(layer_decay)))

    def _score_to_probability(
        self,
        salience: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
        write_gate: Optional[torch.Tensor] = None,
        layer: int = 0,
    ) -> torch.Tensor:
        thr = torch.clamp(self.salience_threshold, 0.0, 1.0)
        temp = torch.clamp(self.decay.abs() + 0.25, 0.05, 5.0)
        score = self._adjusted_score(salience, novelty, write_gate, layer)
        return torch.sigmoid((score - thr) / temp)

    def _adjusted_score(
        self,
        salience: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
        write_gate: Optional[torch.Tensor] = None,
        layer: int = 0,
    ) -> torch.Tensor:
        score = salience

        if novelty is not None:
            score = score + torch.clamp(self.novelty_bias, -10.0, 10.0) * novelty

        if write_gate is not None:
            score = score * write_gate

        if layer > 0:
            score = score * torch.exp(-torch.clamp(self.layer_decay, 0.0, 10.0) * float(layer))

        return score

    def forward(
        self,
        bank: TraceBank,
        key: torch.Tensor,
        value: torch.Tensor,
        salience: torch.Tensor,
        timestamp: int,
        layer: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        novelty: Optional[torch.Tensor] = None,
        write_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if salience.ndim == 2:
            salience = salience.squeeze(-1)
        if salience.ndim == 0:
            salience = salience.view(1)
        if key.ndim == 1:
            key = key.unsqueeze(0)
        if value.ndim == 1:
            value = value.unsqueeze(0)

        if key.shape[0] != value.shape[0]:
            raise ValueError("Key/value batch mismatch")

        if salience.shape[0] != key.shape[0]:
            raise ValueError("Salience batch mismatch")

        salience = salience.detach().reshape(-1)
        novelty = novelty.detach().reshape(-1) if novelty is not None else None
        write_gate = write_gate.detach().reshape(-1) if write_gate is not None else None

        # The threshold is the deterministic admission boundary.  The
        # probability setting is an explicit exploration rate below that
        # boundary; zero must disable random writes.  Sampling the sigmoid
        # score directly made low-salience writes possible at the default.
        floor = torch.clamp(self.min_store_probability, 0.0, 1.0)
        adjusted_score = self._adjusted_score(salience, novelty, write_gate, layer)
        eligible = (
            torch.ones_like(salience, dtype=torch.bool)
            if write_gate is None
            else write_gate > 0
        )
        deterministic_mask = eligible & (
            adjusted_score >= torch.clamp(self.salience_threshold, 0.0, 1.0)
        )

        if self.training:
            random_mask = eligible & (~deterministic_mask) & (torch.rand_like(salience) < floor)
            store_mask = deterministic_mask | random_mask
        else:
            store_mask = deterministic_mask

        if not store_mask.any():
            return torch.zeros_like(store_mask, dtype=torch.bool)

        key_sel = key[store_mask]
        value_sel = value[store_mask]
        sal_sel = salience[store_mask]

        if self.decay.item() > 0.0:
            decay_factor = torch.exp(-torch.clamp(self.decay, 0.0, 10.0) * float(timestamp))
            value_sel = value_sel * decay_factor

        meta = metadata or {}
        for i in range(key_sel.shape[0]):
            bank.add(
                key=key_sel[i],
                value=value_sel[i],
                salience=sal_sel[i],
                timestamp=timestamp,
                layer=layer,
                metadata=meta,
            )

        return store_mask.to(dtype=torch.bool)

    def analyze(
        self,
        salience: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
        write_gate: Optional[torch.Tensor] = None,
        layer: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if salience.ndim == 0:
            salience = salience.view(1)
        score_probability = self._score_to_probability(
            salience=salience.reshape(-1),
            novelty=novelty.reshape(-1) if novelty is not None else None,
            write_gate=write_gate.reshape(-1) if write_gate is not None else None,
            layer=layer,
        )
        threshold = torch.clamp(self.salience_threshold, 0.0, 1.0).detach()
        floor = torch.clamp(self.min_store_probability, 0.0, 1.0).detach()
        adjusted_score = self._adjusted_score(
            salience=salience.reshape(-1),
            novelty=novelty.reshape(-1) if novelty is not None else None,
            write_gate=write_gate.reshape(-1) if write_gate is not None else None,
            layer=layer,
        )
        eligible = (
            torch.ones_like(adjusted_score, dtype=torch.bool)
            if write_gate is None
            else write_gate.reshape(-1) > 0
        )
        deterministic = eligible & (adjusted_score >= threshold)
        write_probability = torch.where(
            deterministic,
            torch.ones_like(score_probability),
            floor.expand_as(score_probability),
        )
        write_probability = torch.where(
            eligible,
            write_probability,
            torch.zeros_like(write_probability),
        )
        return {
            "write_probability": write_probability,
            "score_probability": score_probability,
            "threshold": threshold,
            "floor": floor,
        }
