from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import FabricDynamics, SalienceGate
from .state import FabricState


class LocalWrite(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.kernel = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, write_signal: torch.Tensor, fabric: torch.Tensor):
        if write_signal.shape != fabric.shape:
            raise ValueError("write_signal and fabric must have the same shape")

        delta = self.kernel(write_signal)
        g = self.gate(torch.cat([write_signal, fabric], dim=1))
        updated = fabric + g * delta
        return updated, g


class TraceBank(nn.Module):
    def __init__(self, channels: int, capacity: int = 256, top_k: int = 32):
        super().__init__()

        self.channels = int(channels)
        self.capacity = int(capacity)
        self.top_k = int(top_k)

        self.register_buffer("keys", torch.zeros(self.capacity, self.channels))
        self.register_buffer("values", torch.zeros(self.capacity, self.channels))
        self.register_buffer("weights", torch.zeros(self.capacity))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("size", torch.zeros((), dtype=torch.long))

    def clear(self):
        self.keys.zero_()
        self.values.zero_()
        self.weights.zero_()
        self.ptr.zero_()
        self.size.zero_()

    @torch.no_grad()
    def add(self, trace: torch.Tensor, weight: float):
        if trace.ndim != 4:
            raise ValueError("trace must have shape (B, C, H, W)")
        if trace.shape[1] != self.channels:
            raise ValueError("trace channel dimension does not match TraceBank channels")

        pooled = F.adaptive_avg_pool2d(trace, 1).flatten(1)
        pooled = F.normalize(pooled, dim=-1, eps=1e-6)

        n = int(pooled.shape[0])
        if n == 0:
            return

        cap = self.capacity
        ptr = int(self.ptr.item())
        weight_tensor = torch.full(
            (n,),
            float(weight),
            device=pooled.device,
            dtype=self.weights.dtype,
        )

        if n >= cap:
            self.keys.copy_(pooled[-cap:])
            self.values.copy_(pooled[-cap:])
            self.weights.copy_(weight_tensor[-cap:])
            self.ptr.zero_()
            self.size.fill_(cap)
            return

        end = ptr + n
        if end <= cap:
            self.keys[ptr:end].copy_(pooled)
            self.values[ptr:end].copy_(pooled)
            self.weights[ptr:end].copy_(weight_tensor)
        else:
            first = cap - ptr
            self.keys[ptr:].copy_(pooled[:first])
            self.values[ptr:].copy_(pooled[:first])
            self.weights[ptr:].copy_(weight_tensor[:first])

            remain = n - first
            self.keys[:remain].copy_(pooled[first:])
            self.values[:remain].copy_(pooled[first:])
            self.weights[:remain].copy_(weight_tensor[first:])

        self.ptr.fill_((ptr + n) % cap)
        self.size.fill_(min(int(self.size.item()) + n, cap))

    def retrieve(self, query: torch.Tensor, temperature: float = 1.0):
        if query.ndim != 4:
            raise ValueError("query must have shape (B, C, H, W)")
        if query.shape[1] != self.channels:
            raise ValueError("query channel dimension does not match TraceBank channels")

        size = int(self.size.item())
        if size == 0:
            return torch.zeros_like(query)

        b, c, h, w = query.shape

        q = F.adaptive_avg_pool2d(query, 1).flatten(1)
        q = F.normalize(q, dim=-1, eps=1e-6)

        keys = self.keys[:size]
        values = self.values[:size]
        weights = self.weights[:size].clamp_min(0.0)

        sim = torch.matmul(q, keys.t())
        logits = sim / max(float(temperature), 1e-6)

        k = min(self.top_k, size)
        if k < size:
            top_logits, top_idx = torch.topk(logits, k=k, dim=-1)
            attn = torch.softmax(top_logits, dim=-1)
            attn = attn * weights[top_idx]
            context = torch.sum(attn.unsqueeze(-1) * values[top_idx], dim=1)
        else:
            attn = torch.softmax(logits, dim=-1)
            attn = attn * weights.unsqueeze(0)
            context = torch.matmul(attn, values)

        context = context.unsqueeze(-1).unsqueeze(-1)
        return context.expand(-1, -1, h, w)


@dataclass
class FabricStepOutput:
    state: FabricState
    write_gate: Optional[torch.Tensor] = None
    salience: Optional[torch.Tensor] = None
    retrieved: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None


class Fabric(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fabric_channels: int,
        *,
        height: int,
        width: int,
        trace_capacity: int = 256,
        max_norm: float | None = 1000.0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.fabric_channels = int(fabric_channels)
        self.height = int(height)
        self.width = int(width)

        self.encoder = nn.Conv2d(self.in_channels, self.fabric_channels, 3, padding=1)
        self.writer = LocalWrite(self.fabric_channels)
        self.dynamics = FabricDynamics(self.fabric_channels, max_norm=max_norm)
        self.salience = SalienceGate(self.fabric_channels)
        self.readout = nn.Conv2d(self.fabric_channels, self.in_channels, 3, padding=1)
        self.trace_bank = TraceBank(self.fabric_channels, capacity=trace_capacity)

        self.register_buffer(
            "state_tensor",
            torch.zeros(1, self.fabric_channels, self.height, self.width),
        )

        self.step_count = 0
        self.retrieval_strength = nn.Parameter(torch.tensor(0.5))
        self.salience_threshold = 0.5

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None):
        dtype = dtype or self.state_tensor.dtype
        self.state_tensor = torch.zeros(
            batch_size,
            self.fabric_channels,
            self.height,
            self.width,
            device=device,
            dtype=dtype,
        )
        self.step_count = 0

    def step(
        self,
        x: torch.Tensor,
        *,
        store_traces: bool = True,
        retrieve: bool = True,
        temperature: float = 1.0,
    ) -> FabricStepOutput:
        if x.ndim != 4:
            raise ValueError("x must have shape (B, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError("x channel dimension does not match Fabric input channels")
        if x.shape[2] != self.height or x.shape[3] != self.width:
            raise ValueError("x spatial dimensions do not match Fabric height/width")

        batch_size = int(x.shape[0])

        if self.state_tensor.shape[0] != batch_size or self.state_tensor.device != x.device:
            self.reset_state(batch_size, x.device, dtype=x.dtype)

        z = self.encoder(x)
        written, write_gate = self.writer(z, self.state_tensor)
        evolved = self.dynamics(written)
        sal = self.salience(evolved)

        if store_traces:
            sal_score = float(sal.mean().item())
            if sal_score >= self.salience_threshold:
                self.trace_bank.add(evolved * sal, weight=sal_score)

        retrieved = None
        if retrieve:
            retrieved = self.trace_bank.retrieve(evolved, temperature=temperature)
            strength = torch.clamp(self.retrieval_strength, 0.0, 1.0)
            evolved = evolved + strength * retrieved

        pred = self.readout(evolved)
        self.state_tensor = evolved.detach()
        self.step_count += 1

        state = FabricState(
            tensor=evolved,
            step=self.step_count,
            metadata={
                "salience_mean": float(sal.mean().item()),
                "retrieval_strength": float(torch.clamp(self.retrieval_strength, 0.0, 1.0).item()),
            },
        )

        return FabricStepOutput(
            state=state,
            write_gate=write_gate,
            salience=sal,
            retrieved=retrieved,
            prediction=pred,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x).prediction