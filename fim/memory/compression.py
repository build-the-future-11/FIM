from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompressedTrace:
    key: torch.Tensor
    value: torch.Tensor
    context: torch.Tensor
    pooled: torch.Tensor
    spatial: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None
    write_gate: Optional[torch.Tensor] = None
    temporal_index: Optional[torch.Tensor] = None
    regularization: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.key
        yield self.value


class TraceCompressor(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        trace_dim: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        normalize: bool = True,
        use_attention: bool = True,
        spatial_dim: Optional[int] = None,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()

        self.latent_channels = int(latent_channels)
        self.trace_dim = int(trace_dim or latent_channels)
        self.hidden_channels = int(hidden_channels or max(latent_channels, self.trace_dim))
        self.normalize = bool(normalize)
        self.use_attention = bool(use_attention)
        self.spatial_dim = int(spatial_dim or self.trace_dim)
        self.use_temporal = bool(use_temporal)

        self.backbone = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.key_head = nn.Linear(self.hidden_channels, self.trace_dim)
        self.value_head = nn.Linear(self.hidden_channels, self.trace_dim)
        self.context_head = nn.Linear(self.hidden_channels, self.trace_dim)
        self.spatial_head = nn.Conv2d(self.hidden_channels, self.spatial_dim, kernel_size=1)
        self.write_head = nn.Linear(self.hidden_channels, 1)

        if self.use_attention:
            self.attn_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=1)

        if self.use_temporal:
            self.temporal_proj = nn.Sequential(
                nn.Linear(1, self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, self.hidden_channels),
            )

    def forward(self, fabric: torch.Tensor, temporal_index: Optional[torch.Tensor] = None) -> CompressedTrace:
        if fabric.ndim != 4:
            raise ValueError("fabric must have shape (B, C, H, W)")
        if fabric.shape[1] != self.latent_channels:
            raise ValueError("fabric channel mismatch")

        x = self.backbone(fabric)

        temporal_embed = None
        if self.use_temporal and temporal_index is not None:
            if temporal_index.ndim == 0:
                temporal_index = temporal_index.view(1, 1)
            elif temporal_index.ndim == 1:
                temporal_index = temporal_index.unsqueeze(-1)
            elif temporal_index.ndim > 2:
                temporal_index = temporal_index.reshape(temporal_index.shape[0], -1)[:, :1]
            temporal_index = temporal_index.to(device=x.device, dtype=x.dtype)
            temporal_embed = self.temporal_proj(temporal_index)
            x = x + temporal_embed.unsqueeze(-1).unsqueeze(-1)

        attention = None
        if self.use_attention:
            logits = self.attn_conv(x)
            attention = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
            pooled = (x * attention).flatten(2).sum(dim=-1)
        else:
            pooled = self.pool(x).flatten(1)

        key = self.key_head(pooled)
        value = self.value_head(pooled)
        context = self.context_head(pooled)
        spatial = self.spatial_head(x)
        write_gate = torch.sigmoid(self.write_head(pooled))

        if self.normalize:
            key = F.normalize(key, dim=-1, eps=1e-6)
            value = F.normalize(value, dim=-1, eps=1e-6)
            context = F.normalize(context, dim=-1, eps=1e-6)

        regularization = {}
        if attention is not None:
            attn_flat = attention.flatten(2).clamp_min(1e-8)
            entropy = -(attn_flat * attn_flat.log()).sum(dim=-1).mean()
            regularization["attention_entropy"] = entropy
            regularization["attention_peak"] = attention.amax(dim=(1, 2, 3)).mean()
        regularization["write_gate_mean"] = write_gate.mean()

        return CompressedTrace(
            key=key,
            value=value,
            context=context,
            pooled=pooled,
            spatial=spatial,
            attention=attention,
            write_gate=write_gate,
            temporal_index=temporal_index,
            regularization=regularization,
        )

    def similarity(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if query.ndim != 2 or keys.ndim != 2:
            raise ValueError("query and keys must be (B, D) and (N, D)")

        q = F.normalize(query, dim=-1, eps=1e-6)
        k = F.normalize(keys, dim=-1, eps=1e-6)
        sim = torch.matmul(q, k.t())
        logits = sim / max(float(temperature), 1e-6)
        return torch.softmax(logits, dim=-1)

    def reconstruct(
        self,
        weights: torch.Tensor,
        values: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        if weights.ndim != 2:
            raise ValueError("weights must be (B, N)")
        if values.ndim != 2:
            raise ValueError("values must be (N, D)")

        context = torch.matmul(weights, values)
        context = context.unsqueeze(-1).unsqueeze(-1)
        return context.expand(-1, -1, spatial_size[0], spatial_size[1])