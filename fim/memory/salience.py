from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SalienceOutput:
    score_map: torch.Tensor
    score: torch.Tensor
    gate: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)


class SalienceScorer(nn.Module):
    def __init__(
        self,
        obs_channels: int,
        latent_channels: int,
        hidden_channels: int = 128,
        threshold: float = 0.5,
        temperature: float = 0.25,
        use_attention: bool = True,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()

        self.obs_channels = int(obs_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)

        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.use_attention = bool(use_attention)
        self.use_temporal = bool(use_temporal)

        self.obs_proj = nn.Sequential(
            nn.Conv2d(self.obs_channels, self.hidden_channels, 1),
            nn.GELU(),
        )
        self.pred_proj = nn.Sequential(
            nn.Conv2d(self.obs_channels, self.hidden_channels, 1),
            nn.GELU(),
        )
        self.fabric_proj = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.hidden_channels, 1),
            nn.GELU(),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.hidden_channels, 1),
            nn.GELU(),
        )

        self.norm = nn.GroupNorm(max(1, min(8, self.hidden_channels)), self.hidden_channels)

        in_channels = self.hidden_channels * 4 + 1
        if self.use_temporal:
            in_channels += 1
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, 1, 1),
        )

        if self.use_attention:
            self.attn_head = nn.Conv2d(in_channels, 1, 1)

    def _align_map(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return x.view(1, 1, 1, 1).expand(target.shape[0], 1, target.shape[2], target.shape[3])
        if x.ndim == 1:
            return x.view(-1, 1, 1, 1).expand(target.shape[0], 1, target.shape[2], target.shape[3])
        if x.ndim == 2:
            return x.unsqueeze(-1).unsqueeze(-1).expand(target.shape[0], 1, target.shape[2], target.shape[3])
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def _compute(
        self,
        observation: torch.Tensor,
        prediction: Optional[torch.Tensor],
        fabric_before: torch.Tensor,
        fabric_after: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        memory_pressure: Optional[torch.Tensor] = None,
    ) -> SalienceOutput:
        if observation.ndim != 4:
            raise ValueError("observation must be (B, C, H, W)")
        if fabric_before.shape != fabric_after.shape:
            raise ValueError("fabric_before and fabric_after must match")

        if prediction is None:
            prediction = torch.zeros_like(observation)

        residual = observation - prediction
        fabric_delta = fabric_after - fabric_before

        obs_feat = self.norm(self.obs_proj(observation))
        pred_feat = self.norm(self.pred_proj(prediction))
        fabric_feat = self.norm(self.fabric_proj(fabric_before))
        delta_feat = self.norm(self.delta_proj(fabric_delta))

        residual_energy = residual.pow(2).mean(dim=1, keepdim=True)
        delta_energy = fabric_delta.pow(2).mean(dim=1, keepdim=True)

        if uncertainty is None:
            uncertainty_map = torch.zeros_like(residual_energy)
        else:
            uncertainty_map = self._align_map(uncertainty, residual_energy)
            uncertainty_map = uncertainty_map.to(residual_energy.dtype)

        time_map = None
        if self.use_temporal:
            if timestep is None:
                time_map = torch.zeros_like(residual_energy)
            else:
                time_map = self._align_map(timestep, residual_energy).to(residual_energy.dtype)
        else:
            time_map = torch.zeros_like(residual_energy)

        if memory_pressure is None:
            memory_pressure_map = torch.zeros_like(residual_energy)
        else:
            memory_pressure_map = self._align_map(memory_pressure, residual_energy).to(residual_energy.dtype)

        energy = residual_energy + 0.5 * uncertainty_map + 0.25 * memory_pressure_map

        score_in = torch.cat(
            [obs_feat, pred_feat, fabric_feat, delta_feat, energy, time_map],
            dim=1,
        )

        raw_score = self.score_head(score_in)

        if self.use_attention:
            attn_logits = self.attn_head(score_in)
            attention = torch.softmax(attn_logits.flatten(2), dim=-1).view_as(attn_logits)
            raw_score = raw_score * attention
        else:
            attention = torch.zeros_like(raw_score)

        score_map = F.softplus(raw_score) * torch.clamp(self.scale, 0.1, 10.0)
        score = score_map.mean(dim=(1, 2, 3))

        temp = torch.clamp(self.temperature, 0.05, 5.0)
        thr = torch.clamp(self.threshold, 0.0, 1.0)
        gate = torch.sigmoid((score_map - thr) / temp)

        components = {
            "residual_energy": residual_energy,
            "fabric_delta_energy": delta_energy,
            "uncertainty": uncertainty_map,
            "time": time_map,
            "memory_pressure": memory_pressure_map,
            "attention": attention,
        }

        return SalienceOutput(
            score_map=score_map,
            score=score,
            gate=gate,
            components=components,
        )

    def analyze(
        self,
        observation: torch.Tensor,
        prediction: Optional[torch.Tensor],
        fabric_before: torch.Tensor,
        fabric_after: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        memory_pressure: Optional[torch.Tensor] = None,
    ) -> SalienceOutput:
        return self._compute(
            observation=observation,
            prediction=prediction,
            fabric_before=fabric_before,
            fabric_after=fabric_after,
            uncertainty=uncertainty,
            timestep=timestep,
            memory_pressure=memory_pressure,
        )

    def forward(
        self,
        observation: torch.Tensor,
        prediction: Optional[torch.Tensor],
        fabric_before: torch.Tensor,
        fabric_after: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        memory_pressure: Optional[torch.Tensor] = None,
        return_full: bool = False,
    ):
        out = self._compute(
            observation=observation,
            prediction=prediction,
            fabric_before=fabric_before,
            fabric_after=fabric_after,
            uncertainty=uncertainty,
            timestep=timestep,
            memory_pressure=memory_pressure,
        )
        if return_full:
            return out
        return out.score