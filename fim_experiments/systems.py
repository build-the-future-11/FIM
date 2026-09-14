from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Decoder, Encoder, LatentDynamics, Salience, TraceCompressor


def _field_shape(x: torch.Tensor) -> Tuple[int, int, int, int]:
    if x.ndim == 4:
        return x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    if x.ndim == 3:
        return x.shape[0], x.shape[1], 1, x.shape[2]
    if x.ndim == 2:
        return x.shape[0], 1, 1, x.shape[1]
    raise ValueError(f"Expected tensor with 2, 3, or 4 dims, got {tuple(x.shape)}")


def _to_field(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.contiguous()
    if x.ndim == 3:
        return x.unsqueeze(2).contiguous()
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(2).contiguous()
    raise ValueError(f"Expected tensor with 2, 3, or 4 dims, got {tuple(x.shape)}")


def _like_input(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim == 4:
        return pred
    if ref.ndim == 3:
        return pred.squeeze(2)
    if ref.ndim == 2:
        return pred.squeeze(1).squeeze(1)
    return pred


@dataclass
class FabricState:
    tensor: torch.Tensor
    pre_retrieval: Optional[torch.Tensor] = None
    layers: Optional[List[torch.Tensor]] = None


@dataclass
class FIMOutput:
    prediction: torch.Tensor
    state: FabricState
    latent: torch.Tensor
    salience: torch.Tensor
    retrieved: Optional[torch.Tensor]
    memory_key: Optional[torch.Tensor] = None
    memory_value: Optional[torch.Tensor] = None


class AdaptiveMemoryBank(nn.Module):
    def __init__(self, dim: int, capacity: int, age_decay: float = 0.01) -> None:
        super().__init__()
        self.capacity = int(max(1, capacity))
        self.dim = int(dim)
        self.age_decay = float(age_decay)
        self.register_buffer('keys', torch.zeros(self.capacity, self.dim))
        self.register_buffer('values', torch.zeros(self.capacity, self.dim))
        self.register_buffer('scores', torch.zeros(self.capacity))
        self.register_buffer('ages', torch.zeros(self.capacity))
        # Entries are scoped to a trajectory position within the current
        # minibatch.  Without this field, sample i can retrieve a trace written
        # by sample j, which is target-independent leakage across evaluation
        # examples and an unintended communication channel during training.
        self.register_buffer('owners', torch.full((self.capacity,), -1, dtype=torch.long))
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def clear(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.scores.zero_()
        self.ages.zero_()
        self.owners.fill_(-1)
        self.ptr = 0
        self.size = 0

    def advance(self) -> None:
        if self.size > 0:
            self.ages[: self.size] += 1.0

    def _write(
        self,
        idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        score: torch.Tensor,
        owner: torch.Tensor,
    ) -> None:
        self.keys[idx].copy_(k)
        self.values[idx].copy_(v)
        self.scores[idx].copy_(score)
        self.ages[idx].zero_()
        self.owners[idx].copy_(owner)

    def add(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        score: Optional[torch.Tensor] = None,
        owner: Optional[torch.Tensor] = None,
    ) -> None:
        if k is None or v is None or k.numel() == 0 or v.numel() == 0:
            return
        if k.ndim == 1:
            k = k.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if k.shape != v.shape:
            raise ValueError(f'Key/value shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}')
        if k.shape[-1] != self.dim:
            raise ValueError(f'Expected trace dim {self.dim}, got {k.shape[-1]}')

        k = k.detach()
        v = v.detach()
        if score is None:
            score = torch.ones(k.shape[0], device=k.device, dtype=k.dtype)
        if score.ndim == 0:
            score = score.unsqueeze(0)
        score = score.detach().flatten()
        if score.shape[0] != k.shape[0]:
            score = score.expand(k.shape[0])
        if owner is None:
            owner = torch.arange(k.shape[0], device=k.device, dtype=torch.long)
        owner = owner.detach().flatten().to(device=k.device, dtype=torch.long)
        if owner.shape[0] != k.shape[0]:
            raise ValueError("owner must contain one trajectory id per key/value")

        if k.shape[0] >= self.capacity:
            keep_k = k[-self.capacity :]
            keep_v = v[-self.capacity :]
            keep_s = score[-self.capacity :]
            self.keys.copy_(keep_k)
            self.values.copy_(keep_v)
            self.scores.copy_(keep_s)
            self.owners.copy_(owner[-self.capacity :])
            self.ages.zero_()
            self.ptr = 0
            self.size = self.capacity
            return

        for i in range(k.shape[0]):
            if self.size < self.capacity:
                idx = self.ptr
                self.ptr = (self.ptr + 1) % self.capacity
                self.size += 1
            else:
                priority = self.scores[: self.size] - self.age_decay * torch.log1p(self.ages[: self.size])
                idx = int(torch.argmin(priority).item())
            self._write(idx, k[i], v[i], score[i], owner[i])

    def retrieve(
        self,
        q: torch.Tensor,
        topk: int = 32,
        temperature: float = 0.2,
        owner: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.size == 0:
            return None
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=-1, eps=1e-8)
        keys = F.normalize(self.keys[: self.size], dim=-1, eps=1e-8)
        sim = torch.matmul(q, keys.T)
        age_bias = self.age_decay * torch.log1p(self.ages[: self.size]).unsqueeze(0)
        score = sim - age_bias + 0.05 * self.scores[: self.size].unsqueeze(0)
        if owner is None:
            owner = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
        owner = owner.flatten().to(device=q.device, dtype=torch.long)
        if owner.shape[0] != q.shape[0]:
            raise ValueError("owner must contain one trajectory id per query")
        eligible = owner.unsqueeze(1) == self.owners[: self.size].unsqueeze(0)
        if not bool(eligible.any(dim=1).all().item()):
            return None
        score = score.masked_fill(~eligible, float('-inf'))
        k = max(1, min(int(topk), self.size))
        vals, idx = torch.topk(score, k=k, dim=-1)
        gathered = self.values[: self.size][idx]
        temp = float(max(0.05, min(1.0, temperature)))
        weights = torch.softmax(vals / temp, dim=-1).unsqueeze(-1)
        return (weights * gathered).sum(dim=1)


class _StepOutputMixin:
    def _pack(self, x: torch.Tensor, pred: torch.Tensor, latent: torch.Tensor, salience: torch.Tensor, retrieved: Optional[torch.Tensor], key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> FIMOutput:
        return FIMOutput(
            prediction=_like_input(pred, x),
            state=FabricState(tensor=latent),
            latent=latent,
            salience=salience,
            retrieved=retrieved,
            memory_key=key,
            memory_value=value,
        )


class FIMSystem(nn.Module, _StepOutputMixin):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        trace_dim: int,
        memory_capacity: int = 2048,
        retrieval_topk: int = 32,
        memory_decay: float = 0.02,
        salience_threshold: float = 0.45,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hidden)
        self.dynamics = LatentDynamics(hidden)
        self.decoder = Decoder(hidden, in_channels)
        self.salience = Salience(hidden)
        self.compressor = TraceCompressor(hidden, trace_dim)
        self.bank = AdaptiveMemoryBank(trace_dim, capacity=memory_capacity, age_decay=memory_decay)
        self.trace_bank = self.bank
        self.memory_bank = self.bank
        self.retrieval_proj = nn.Linear(trace_dim, hidden)
        self.retrieval_gate = nn.Linear(trace_dim, hidden)
        self.retrieval_topk = int(retrieval_topk)
        self.salience_threshold = float(salience_threshold)
        self.retrieval_temperature = nn.Parameter(torch.tensor(0.2))
        self.feedback_scale = nn.Parameter(torch.tensor(0.5))
        self.hidden = int(hidden)
        self.trace_dim = int(trace_dim)

    def clear_memory(self) -> None:
        self.bank.clear()

    def reset_state(self) -> None:
        self.clear_memory()

    def detach_state(self) -> None:
        self.clear_memory()

    def _match_latent(self, update: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        while update.ndim < ref.ndim:
            update = update.unsqueeze(-1)
        return update.expand_as(ref)

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        z = self.encoder(x4)
        z = self.dynamics(z)

        sal = self.salience(z)
        key, value = self.compressor(z)
        owners = torch.arange(key.shape[0], device=key.device, dtype=torch.long)

        retrieved = None
        if retrieve and len(self.bank) > 0:
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

        if store_traces:
            with torch.no_grad():
                sal_score = sal.reshape(sal.shape[0], -1).mean(dim=-1)
                mask = sal_score > self.salience_threshold
                if mask.any():
                    self.bank.add(
                        key[mask],
                        value[mask],
                        score=sal_score[mask],
                        owner=owners[mask],
                    )
                self.bank.advance()

        return self._pack(x4, pred, z, sal, retrieved, key, value)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        groups = max(1, min(8, channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g = torch.sigmoid(self.gate(h))
        h = F.gelu(self.depthwise(h))
        h = self.pointwise(h)
        return x + torch.tanh(self.scale) * g * h


class SelectiveSSMSystem(nn.Module, _StepOutputMixin):
    def __init__(self, in_channels: int, hidden: int, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(max(2, depth))])
        self.norm = nn.GroupNorm(max(1, min(8, hidden)), hidden)
        self.head = nn.Conv2d(hidden, in_channels, 3, padding=1)
        self.hidden = hidden

    def clear_memory(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def detach_state(self) -> None:
        return None

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        z = self.stem(x4)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        pred = self.head(z)
        sal = torch.sigmoid(z.mean(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True))
        return self._pack(x4, pred, z, sal.flatten(1), None)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


class SpectralFieldSystem(nn.Module, _StepOutputMixin):
    def __init__(self, in_channels: int, hidden: int, modes: int = 8, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([LatentDynamics(hidden) for _ in range(max(2, depth))])
        self.head = nn.Conv2d(hidden, in_channels, 3, padding=1)
        self.hidden = hidden
        self.modes = modes

    def clear_memory(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def detach_state(self) -> None:
        return None

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        z = self.stem(x4)
        for block in self.blocks:
            z = block(z)
        pred = self.head(z)
        sal = torch.sigmoid(z.mean(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True))
        return self._pack(x4, pred, z, sal.flatten(1), None)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


class TransformerFieldSystem(nn.Module, _StepOutputMixin):
    def __init__(self, in_channels: int, hidden: int, heads: int = 4, depth: int = 3, ff_mult: int = 4, max_tokens: int = 4096) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden, 3, padding=1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, heads),
            dim_feedforward=hidden * ff_mult,
            batch_first=True,
            norm_first=True,
            dropout=0.05,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(2, depth))
        self.head = nn.Conv2d(hidden, in_channels, 1)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.hidden = hidden
        self.max_tokens = max_tokens

    def clear_memory(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def detach_state(self) -> None:
        return None

    def _positional(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if n <= self.pos.shape[1]:
            return self.pos[:, :n].to(device=device, dtype=dtype)
        idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(-1)
        dim = torch.arange(self.hidden, device=device, dtype=dtype).unsqueeze(0)
        angle = idx / torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2 * (dim // 2)) / max(1, self.hidden))
        pe = torch.zeros(1, n, self.hidden, device=device, dtype=dtype)
        pe[..., 0::2] = torch.sin(angle[..., 0::2])
        pe[..., 1::2] = torch.cos(angle[..., 1::2])
        return pe

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        z = self.stem(x4)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        pos = self._positional(tokens.shape[1], z.device, z.dtype)
        tokens = tokens + pos
        tokens = self.encoder(tokens)
        z = tokens.transpose(1, 2).reshape(b, c, h, w)
        pred = self.head(z)
        sal = torch.sigmoid(z.mean(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True))
        return self._pack(x4, pred, z, sal.flatten(1), None)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


class DeepONetFieldSystem(nn.Module, _StepOutputMixin):
    def __init__(self, in_channels: int, hidden: int, basis_dim: int = 64) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, basis_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, basis_dim),
        )
        self.project = nn.Conv2d(basis_dim, hidden, 1)
        self.head = nn.Conv2d(hidden, in_channels, 3, padding=1)
        self.hidden = hidden
        self.basis_dim = basis_dim

    def clear_memory(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def detach_state(self) -> None:
        return None

    def _grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([yy, xx], dim=-1).reshape(h * w, 2)

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        b, c, h, w = x4.shape
        coeff = self.branch(x4)
        basis = self.trunk(self._grid(h, w, x4.device, x4.dtype)).view(h * w, self.basis_dim)
        field = torch.einsum('bd,nd->bnd', coeff, basis).transpose(1, 2).reshape(b, self.basis_dim, h, w)
        z = self.project(field)
        pred = self.head(z)
        sal = torch.sigmoid(z.mean(dim=(2, 3), keepdim=True).mean(dim=1, keepdim=True))
        return self._pack(x4, pred, z, sal.flatten(1), None)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


class FieldMLPSystem(nn.Module, _StepOutputMixin):
    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.LazyLinear(in_channels)
        self.hidden = hidden

    def clear_memory(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def detach_state(self) -> None:
        return None

    def step(self, x: torch.Tensor, store_traces: bool = True, retrieve: bool = True) -> FIMOutput:
        x4 = _to_field(x)
        b, c, h, w = x4.shape
        flat = x4.reshape(b, -1)
        h1 = self.net(flat)
        out = self.out(h1).view(b, c, 1, 1).expand(-1, -1, h, w)
        sal = torch.sigmoid(h1.mean(dim=-1, keepdim=True))
        return self._pack(x4, out, out, sal, None)

    def forward(self, x: torch.Tensor) -> FIMOutput:
        return self.step(x)


MODEL_REGISTRY = {
    'fim': FIMSystem,
    'fim_plus': FIMSystem,
    'transformer': TransformerFieldSystem,
    'ssm': SelectiveSSMSystem,
    'spectral': SpectralFieldSystem,
    'fno': SpectralFieldSystem,
    'deeponet': DeepONetFieldSystem,
    'mlp': FieldMLPSystem,
}
