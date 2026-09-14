from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryTrace:
    key: torch.Tensor
    value: torch.Tensor
    salience: float
    timestamp: int
    layer: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceBank:
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        max_traces: int = 4096,
        merge_threshold: float = 0.98,
        decay_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.max_traces = int(max_traces)
        self.merge_threshold = float(merge_threshold)
        self.decay_rate = float(decay_rate)
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self._traces: List[MemoryTrace] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self._traces)

    def clear(self) -> None:
        self._traces.clear()
        self._clock = 0

    def _ensure_device(self, x: torch.Tensor) -> torch.Tensor:
        device = self.device or x.device
        if self.device is None:
            self.device = device
        return x.detach().to(device=device, dtype=self.dtype).reshape(-1)

    def _validate_trace(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_1d = self._ensure_device(key)
        value_1d = self._ensure_device(value)
        if key_1d.numel() != self.key_dim:
            raise ValueError(f"Expected key_dim={self.key_dim}, got {key_1d.numel()}")
        if value_1d.numel() != self.value_dim:
            raise ValueError(f"Expected value_dim={self.value_dim}, got {value_1d.numel()}")
        return key_1d, value_1d

    def _effective_score(self, trace: MemoryTrace, now: Optional[int] = None) -> float:
        current = self._clock if now is None else int(now)
        age = max(0, current - int(trace.timestamp))
        decay = math_exp_safe(-self.decay_rate * age)
        return float(trace.salience * decay)

    def _prune_if_needed(self) -> None:
        if len(self._traces) <= self.max_traces:
            return
        self._traces.sort(key=lambda t: (self._effective_score(t), t.timestamp), reverse=True)
        self._traces = self._traces[: self.max_traces]

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        salience: float | torch.Tensor,
        timestamp: int,
        layer: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        key_1d, value_1d = self._validate_trace(key, value)
        sal = float(salience.item() if torch.is_tensor(salience) else salience)
        meta = metadata or {}
        ts = int(timestamp)
        self._clock = max(self._clock, ts)

        if self._traces:
            keys, _, _, _ = self.as_tensors()
            query = F.normalize(key_1d.unsqueeze(0), dim=-1, eps=1e-6)
            bank_keys = F.normalize(keys, dim=-1, eps=1e-6)
            sim = torch.matmul(query, bank_keys.t()).squeeze(0)
            best_idx = int(torch.argmax(sim).item())
            if float(sim[best_idx].item()) >= self.merge_threshold:
                existing = self._traces[best_idx]
                w_new = max(sal, 1e-6)
                w_old = max(existing.salience, 1e-6)
                alpha = w_new / (w_new + w_old)
                merged_key = (1.0 - alpha) * existing.key + alpha * key_1d
                merged_value = (1.0 - alpha) * existing.value + alpha * value_1d
                self._traces[best_idx] = MemoryTrace(
                    key=merged_key,
                    value=merged_value,
                    salience=max(existing.salience, sal),
                    timestamp=max(existing.timestamp, ts),
                    layer=layer,
                    metadata={**existing.metadata, **meta},
                )
                return

        self._traces.append(
            MemoryTrace(
                key=key_1d,
                value=value_1d,
                salience=sal,
                timestamp=ts,
                layer=layer,
                metadata=meta,
            )
        )
        self._prune_if_needed()

    def batch_add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        saliences: torch.Tensor,
        timestamp: int,
        layer: int = 0,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if saliences.ndim > 1:
            saliences = saliences.reshape(saliences.shape[0], -1).mean(dim=-1)
        if keys.shape[0] != values.shape[0] or keys.shape[0] != saliences.shape[0]:
            raise ValueError("Batch add requires matching batch sizes")

        meta_list = metadata or [{} for _ in range(keys.shape[0])]
        for i in range(keys.shape[0]):
            self.add(
                key=keys[i],
                value=values[i],
                salience=saliences[i],
                timestamp=timestamp,
                layer=layer,
                metadata=meta_list[i],
            )

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self._traces) == 0:
            device = self.device or torch.device("cpu")
            return (
                torch.empty(0, self.key_dim, device=device, dtype=self.dtype),
                torch.empty(0, self.value_dim, device=device, dtype=self.dtype),
                torch.empty(0, 1, device=device, dtype=self.dtype),
                torch.empty(0, 1, device=device, dtype=self.dtype),
            )

        keys = torch.stack([t.key for t in self._traces], dim=0)
        values = torch.stack([t.value for t in self._traces], dim=0)
        saliences = torch.tensor([[t.salience] for t in self._traces], device=keys.device, dtype=self.dtype)
        timestamps = torch.tensor([[t.timestamp] for t in self._traces], device=keys.device, dtype=self.dtype)
        return keys, values, saliences, timestamps

    def statistics(self) -> Dict[str, float]:
        if len(self._traces) == 0:
            return {
                "size": 0.0,
                "mean_salience": 0.0,
                "max_salience": 0.0,
                "mean_age": 0.0,
                "max_age": 0.0,
            }

        now = self._clock
        ages = [max(0, now - t.timestamp) for t in self._traces]
        saliences = [t.salience for t in self._traces]
        return {
            "size": float(len(self._traces)),
            "mean_salience": float(sum(saliences) / len(saliences)),
            "max_salience": float(max(saliences)),
            "mean_age": float(sum(ages) / len(ages)),
            "max_age": float(max(ages)),
        }

    def recent(self, k: int) -> List[MemoryTrace]:
        if k <= 0:
            return []
        return self._traces[-k:]

    def iter_traces(self) -> Iterable[MemoryTrace]:
        return iter(self._traces)

    def _retrieve_impl(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        query = query.reshape(query.shape[0], -1)

        keys, values, saliences, timestamps = self.as_tensors()
        if keys.numel() == 0:
            batch = query.shape[0]
            device = query.device
            dtype = query.dtype
            return (
                torch.zeros(batch, self.value_dim, device=device, dtype=dtype),
                torch.zeros(batch, 0, device=device, dtype=dtype),
                torch.empty(batch, 0, device=device, dtype=torch.long),
                torch.empty(batch, 0, device=device, dtype=dtype),
            )

        device = query.device
        dtype = query.dtype
        keys = keys.to(device=device, dtype=dtype)
        values = values.to(device=device, dtype=dtype)
        saliences = saliences.to(device=device, dtype=dtype).view(-1)
        timestamps = timestamps.to(device=device, dtype=dtype).view(-1)

        q = F.normalize(query, dim=-1, eps=1e-6)
        k = F.normalize(keys, dim=-1, eps=1e-6)

        sim = torch.matmul(q, k.t())
        if saliences.numel() > 0:
            sim = sim + float(salience_scale) * saliences.unsqueeze(0)
        if timestamps.numel() > 0:
            age = timestamps.max() - timestamps
            sim = sim - float(recency_scale) * age.unsqueeze(0)

        temp = max(float(temperature), 1e-6)
        if topk is not None and topk > 0 and topk < sim.shape[-1]:
            top_sim, top_idx = torch.topk(sim, k=topk, dim=-1)
            weights = torch.softmax(top_sim / temp, dim=-1)
            gathered = values.unsqueeze(0).expand(query.shape[0], -1, -1)
            gathered = torch.gather(
                gathered,
                dim=1,
                index=top_idx.unsqueeze(-1).expand(-1, -1, self.value_dim),
            )
            context = torch.sum(weights.unsqueeze(-1) * gathered, dim=1)
            return context, weights, top_idx, top_sim

        weights = torch.softmax(sim / temp, dim=-1)
        context = torch.matmul(weights, values)
        indices = torch.arange(values.shape[0], device=device).unsqueeze(0).expand(query.shape[0], -1)
        return context, weights, indices, sim

    def retrieve(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> torch.Tensor:
        context, _, _, _ = self._retrieve_impl(
            query=query,
            topk=topk,
            temperature=temperature,
            salience_scale=salience_scale,
            recency_scale=recency_scale,
        )
        return context

    def retrieve_detailed(
        self,
        query: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        salience_scale: float = 0.05,
        recency_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._retrieve_impl(
            query=query,
            topk=topk,
            temperature=temperature,
            salience_scale=salience_scale,
            recency_scale=recency_scale,
        )


def math_exp_safe(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return float("inf")
    import math

    return math.exp(x)