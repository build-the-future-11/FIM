from __future__ import annotations

from typing import Optional

import torch

from systems import FIMOutput, FIMSystem, _to_field


class AblatedFIMSystem(FIMSystem):
    """FIMSystem with explicit, auditable current-code component switches.

    These switches are intentionally limited to mechanisms that exist in the
    maintained FIMSystem implementation. They are not aliases for historical
    paper labels whose semantics are no longer represented in current code.
    """

    def __init__(
        self,
        *args,
        memory_enabled: bool = True,
        retrieval_enabled: bool = True,
        salience_gating_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_enabled = bool(memory_enabled)
        self.retrieval_enabled = bool(retrieval_enabled)
        self.salience_gating_enabled = bool(salience_gating_enabled)

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        z = self.encoder(x4)
        z = self.dynamics(z)

        sal = self.salience(z)
        key, value = self.compressor(z)
        owners = torch.arange(key.shape[0], device=key.device, dtype=torch.long)

        retrieved: Optional[torch.Tensor] = None
        if self.memory_enabled and self.retrieval_enabled and retrieve and len(self.bank) > 0:
            retrieved = self.bank.retrieve(
                key,
                topk=self.retrieval_topk,
                temperature=float(torch.clamp(self.retrieval_temperature, 0.05, 1.0).item()),
                owner=owners,
            )
            if retrieved is not None:
                proj = self.retrieval_proj(retrieved)
                gate = torch.sigmoid(self.retrieval_gate(retrieved))
                z = z + self._match_latent(torch.tanh(self.feedback_scale) * proj, z) * self._match_latent(gate, z)

        pred = self.decoder(z)

        if self.memory_enabled and store_traces:
            with torch.no_grad():
                sal_score = sal.reshape(sal.shape[0], -1).mean(dim=-1)
                if self.salience_gating_enabled:
                    mask = sal_score > self.salience_threshold
                else:
                    mask = torch.ones_like(sal_score, dtype=torch.bool)
                if mask.any():
                    self.bank.add(
                        key[mask],
                        value[mask],
                        score=sal_score[mask],
                        owner=owners[mask],
                    )
                self.bank.advance()

        return self._pack(x4, pred, z, sal, retrieved, key, value)


def variant_switches(name: str) -> dict[str, bool]:
    normalized = name.strip().lower().replace('-', '_')
    variants = {
        'full': {
            'memory_enabled': True,
            'retrieval_enabled': True,
            'salience_gating_enabled': True,
        },
        'no_memory': {
            'memory_enabled': False,
            'retrieval_enabled': False,
            'salience_gating_enabled': False,
        },
        'no_retrieval': {
            'memory_enabled': True,
            'retrieval_enabled': False,
            'salience_gating_enabled': True,
        },
        'no_salience_gating': {
            'memory_enabled': True,
            'retrieval_enabled': True,
            'salience_gating_enabled': False,
        },
    }
    if normalized not in variants:
        raise ValueError(f'Unsupported current-code FIM ablation variant: {name}')
    return variants[normalized].copy()
