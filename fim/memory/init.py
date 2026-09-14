from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .compression import CompressedTrace, TraceCompressor
from .consolidation import MemoryConsolidator
from .retrieval import MemoryRetrieval, RetrievalOutput
from .salience import SalienceOutput, SalienceScorer
from .trace_bank import MemoryTrace, TraceBank


@dataclass
class MemoryStepOutput:
    salience: SalienceOutput
    trace: CompressedTrace
    retrieval: RetrievalOutput
    stored_mask: torch.Tensor
    bank_size: int
    regularization: Dict[str, torch.Tensor] = field(default_factory=dict)


class PersistentMemorySystem(nn.Module):
    def __init__(
        self,
        obs_channels: int,
        latent_channels: int,
        trace_dim: int,
        hidden_channels: int = 128,
        max_traces: int = 4096,
        top_k: int = 32,
    ) -> None:
        super().__init__()
        self.bank = TraceBank(
            key_dim=trace_dim,
            value_dim=trace_dim,
            max_traces=max_traces,
        )
        self.scorer = SalienceScorer(
            obs_channels=obs_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        self.compressor = TraceCompressor(
            latent_channels=latent_channels,
            trace_dim=trace_dim,
            hidden_channels=hidden_channels,
        )
        self.consolidator = MemoryConsolidator()
        self.retriever = MemoryRetrieval(
            query_dim=trace_dim,
            value_dim=trace_dim,
            top_k=top_k,
        )

    def reset(self) -> None:
        self.bank.clear()

    def forward(
        self,
        observation: torch.Tensor,
        prediction: Optional[torch.Tensor],
        fabric_before: torch.Tensor,
        fabric_after: torch.Tensor,
        timestamp: int = 0,
        layer: int = 0,
        uncertainty: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_full: bool = True,
    ):
        salience = self.scorer.analyze(
            observation=observation,
            prediction=prediction,
            fabric_before=fabric_before,
            fabric_after=fabric_after,
            uncertainty=uncertainty,
        )
        trace = self.compressor(fabric_after, temporal_index=torch.tensor([[float(timestamp)]], device=fabric_after.device, dtype=fabric_after.dtype))
        write_gate = salience.gate.mean(dim=(1, 2, 3)).reshape(-1)
        stored_mask = self.consolidator(
            bank=self.bank,
            key=trace.key,
            value=trace.value,
            salience=salience.score,
            timestamp=timestamp,
            layer=layer,
            metadata=metadata,
            write_gate=write_gate,
        )
        retrieval = self.retriever(trace.key, self.bank)
        regularization = {}
        regularization.update(trace.regularization)
        regularization.update(retrieval.regularization)
        if return_full:
            return MemoryStepOutput(
                salience=salience,
                trace=trace,
                retrieval=retrieval,
                stored_mask=stored_mask,
                bank_size=len(self.bank),
                regularization=regularization,
            )
        return retrieval.context


MEMORY_REGISTRY = {
    "salience": SalienceScorer,
    "compressor": TraceCompressor,
    "trace_bank": TraceBank,
    "consolidator": MemoryConsolidator,
    "retrieval": MemoryRetrieval,
    "persistent_memory": PersistentMemorySystem,
}


def build_memory_component(name: str, *args, **kwargs):
    if name not in MEMORY_REGISTRY:
        raise ValueError(f"Unknown memory component: {name}")
    return MEMORY_REGISTRY[name](*args, **kwargs)


__all__ = [
    "CompressedTrace",
    "TraceCompressor",
    "MemoryConsolidator",
    "MemoryRetrieval",
    "RetrievalOutput",
    "SalienceOutput",
    "SalienceScorer",
    "MemoryTrace",
    "TraceBank",
    "PersistentMemorySystem",
    "MemoryStepOutput",
    "MEMORY_REGISTRY",
    "build_memory_component",
]