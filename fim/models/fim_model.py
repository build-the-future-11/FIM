from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fim.models.encoder import FabricEncoder
from fim.models.decoder import FabricDecoder


def _make_groups(channels: int, max_groups: int = 8) -> int:
    groups = max(1, min(max_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def _softplus_inverse(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


def _spatial_expand(context: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = target_hw
    return context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


def _ensure_4d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(x.shape)}")
    return x


def _ensure_5d_state(state: torch.Tensor, num_layers: int) -> torch.Tensor:
    if state.ndim == 4:
        if num_layers != 1:
            raise ValueError("A 4D state is only valid when num_layers == 1.")
        state = state.unsqueeze(1)

    if state.ndim != 5:
        raise ValueError("state must have shape [batch, num_layers, channels, height, width]")

    if state.shape[1] != num_layers:
        raise ValueError(f"Expected state with {num_layers} layers, got {state.shape[1]}")

    return state


@dataclass
class SalienceOutput:
    score: torch.Tensor
    gate: torch.Tensor


@dataclass
class CompressionOutput:
    key: torch.Tensor
    value: torch.Tensor
    context: torch.Tensor


@dataclass
class RetrievalOutput:
    context: torch.Tensor
    fast_context: torch.Tensor
    slow_context: torch.Tensor
    weights: List[torch.Tensor]
    indices: List[torch.Tensor]
    level_weights: torch.Tensor


@dataclass
class FIMLayerOutput:
    state: torch.Tensor
    pre_retrieval_state: torch.Tensor
    salience_score: torch.Tensor
    salience_gate: torch.Tensor
    retrieval_context: torch.Tensor
    fast_retrieval_context: Optional[torch.Tensor] = None
    slow_retrieval_context: Optional[torch.Tensor] = None
    retrieval_weights: Optional[Sequence[torch.Tensor]] = None
    retrieval_indices: Optional[Sequence[torch.Tensor]] = None
    stored_mask: Optional[torch.Tensor] = None


@dataclass
class FIMStepOutput:
    prediction: torch.Tensor
    state: torch.Tensor
    stacked_state: torch.Tensor
    pre_retrieval_state: torch.Tensor
    stacked_pre_retrieval_state: torch.Tensor
    layer_states: List[torch.Tensor]
    layer_pre_retrieval_states: List[torch.Tensor]
    salience_score: torch.Tensor
    salience_gate: torch.Tensor
    retrieval_context: torch.Tensor
    fast_retrieval_context: torch.Tensor
    slow_retrieval_context: torch.Tensor
    retrieval_weights: List[Sequence[torch.Tensor]]
    retrieval_indices: List[Sequence[torch.Tensor]]
    retrieval_level_weights: torch.Tensor
    stored_mask: torch.Tensor
    layer_mixture: torch.Tensor
    layer_weights: torch.Tensor


class SpectralMix2d(nn.Module):
    def __init__(self, channels: int, modes: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = max(1, int(modes))
        self.weight = nn.Parameter(torch.randn(channels, self.modes, self.modes, 2) * 0.02)
        self.gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ensure_4d(x, "x")

        b, c, h, w = x.shape
        wf = w // 2 + 1
        mh = min(self.modes, h)
        mw = min(self.modes, wf)

        if mh <= 0 or mw <= 0:
            return x

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)

        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())
        out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        mixed = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        gain = torch.clamp(self.gain, 0.0, 1.5)
        return x + gain * mixed


class PhysicsOperatorBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        control_channels: int = 0,
        boundary_mode: str = "circular",
        complex_mode: bool = False,
    ) -> None:
        super().__init__()
        if complex_mode and channels % 2 != 0:
            raise ValueError("complex_mode requires an even channel count")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.control_channels = int(control_channels)
        self.complex_mode = bool(complex_mode)
        self.boundary_mode = boundary_mode
        padding_mode = boundary_mode if boundary_mode in {"circular", "reflect", "replicate"} else "zeros"

        self.diffusion = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
            padding_mode=padding_mode,
        )

        with torch.no_grad():
            kernel = torch.zeros(channels, 1, 3, 3)
            kernel[:, 0, 1, 1] = -4.0
            kernel[:, 0, 0, 1] = 1.0
            kernel[:, 0, 2, 1] = 1.0
            kernel[:, 0, 1, 0] = 1.0
            kernel[:, 0, 1, 2] = 1.0
            self.diffusion.weight.copy_(kernel)

        self.reaction = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

        self.velocity = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

        self.control_proj = (
            nn.Conv2d(control_channels, channels, kernel_size=1)
            if control_channels > 0
            else None
        )

        self.diffusion_gain = nn.Parameter(torch.tensor(0.35))
        self.advection_gain = nn.Parameter(torch.tensor(0.25))
        self.reaction_gain = nn.Parameter(torch.tensor(0.35))
        self.control_gain = nn.Parameter(torch.tensor(0.25))
        self.noise_log_scale = nn.Parameter(torch.tensor(_softplus_inverse(0.01)))
        self.phase_gain = nn.Parameter(torch.tensor(0.15)) if complex_mode else None
        self.phase_gate = nn.Conv2d(channels, channels // 2, kernel_size=1) if complex_mode else None

    def _grad_x(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.roll(x, shifts=-1, dims=-1) - torch.roll(x, shifts=1, dims=-1))

    def _grad_y(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.roll(x, shifts=-1, dims=-2) - torch.roll(x, shifts=1, dims=-2))

    def forward(
        self,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        dt: float = 1.0,
        stochastic: bool = False,
    ) -> torch.Tensor:
        _ensure_4d(state, "state")

        x = state

        if control is not None and self.control_proj is not None:
            _ensure_4d(control, "control")
            control_term = self.control_proj(control)
        else:
            control_term = torch.zeros_like(x)

        diffusion_term = self.diffusion(x)
        reaction_term = self.reaction(x)

        vel = torch.tanh(self.velocity(x))
        vx = vel[:, 0:1]
        vy = vel[:, 1:2]
        adv = -(vx * self._grad_x(x) + vy * self._grad_y(x))

        phase_term = torch.zeros_like(x)
        if self.complex_mode and self.phase_gate is not None and self.phase_gain is not None:
            real, imag = torch.chunk(x, 2, dim=1)
            theta = math.pi * torch.tanh(self.phase_gate(x))
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            rotated_real = real * cos_t - imag * sin_t
            rotated_imag = real * sin_t + imag * cos_t
            phase_term = torch.cat([rotated_real - real, rotated_imag - imag], dim=1)

        update = (
            torch.tanh(self.diffusion_gain) * diffusion_term
            + torch.tanh(self.advection_gain) * adv
            + torch.tanh(self.reaction_gain) * reaction_term
            + torch.tanh(self.control_gain) * control_term
            + (torch.tanh(self.phase_gain) * phase_term if self.complex_mode and self.phase_gain is not None else 0.0)
        )

        if stochastic:
            sigma = F.softplus(self.noise_log_scale)
            update = update + sigma * math.sqrt(max(float(dt), 1e-6)) * torch.randn_like(x)

        return update


class SalienceScorer(nn.Module):
    def __init__(
        self,
        obs_channels: int,
        pred_channels: int,
        latent_channels: int,
        hidden_channels: int,
        threshold: float = 0.5,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

        self.obs_proj = nn.Conv2d(obs_channels, hidden_channels, kernel_size=1)
        self.pred_proj = nn.Conv2d(pred_channels, hidden_channels, kernel_size=1)
        self.delta_proj = nn.Conv2d(latent_channels, hidden_channels, kernel_size=1)

        self.net = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        prediction: torch.Tensor,
        fabric_before: torch.Tensor,
        fabric_after: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> SalienceOutput:
        obs_h = self.obs_proj(observation)
        pred_h = self.pred_proj(prediction)
        delta_h = self.delta_proj(fabric_after - fabric_before)

        fused = torch.cat([obs_h, pred_h, delta_h], dim=1)
        score_map = torch.sigmoid(self.net(fused))
        score = score_map.mean(dim=(1, 2, 3), keepdim=True)

        if uncertainty is not None:
            unc = uncertainty
            if unc.ndim >= 2:
                unc = unc.mean(dim=tuple(range(1, unc.ndim)), keepdim=True)
            score = torch.clamp(score + 0.05 * torch.sigmoid(unc), 0.0, 1.0)

        temp = torch.clamp(self.temperature, 0.05, 5.0)
        gate = torch.sigmoid((score - self.threshold) / temp)
        return SalienceOutput(score=score, gate=gate)


class TraceCompressor(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        trace_dim: int,
        hidden_channels: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)

        self.key_net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, trace_dim, kernel_size=1),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, trace_dim, kernel_size=1),
        )
        self.context_net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, trace_dim, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> CompressionOutput:
        _ensure_4d(x, "x")

        key_map = self.key_net(x)
        value_map = self.value_net(x)
        context_map = self.context_net(x)

        key = self.pool(key_map).flatten(1)
        value = self.pool(value_map).flatten(1)
        context = self.pool(context_map).flatten(1)

        if self.normalize:
            key = _safe_normalize(key, dim=-1)
            value = _safe_normalize(value, dim=-1)
            context = _safe_normalize(context, dim=-1)

        return CompressionOutput(key=key, value=value, context=context)


class TraceBank(nn.Module):
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        max_traces: int,
        *,
        fast_threshold: float = 0.5,
        slow_threshold: float = 0.8,
        slow_capacity: Optional[int] = None,
        merge_threshold: float = 0.92,
    ) -> None:
        super().__init__()

        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.fast_capacity = int(max_traces)
        self.slow_capacity = int(slow_capacity if slow_capacity is not None else max(1, max_traces // 2))
        self.fast_threshold = float(fast_threshold)
        self.slow_threshold = float(slow_threshold)
        self.merge_threshold = float(merge_threshold)

        self.register_buffer("fast_keys", torch.zeros(self.fast_capacity, self.key_dim), persistent=False)
        self.register_buffer("fast_values", torch.zeros(self.fast_capacity, self.value_dim), persistent=False)
        self.register_buffer("fast_scores", torch.zeros(self.fast_capacity), persistent=False)
        self.register_buffer("fast_times", torch.zeros(self.fast_capacity, dtype=torch.long), persistent=False)
        self.register_buffer("fast_ptr", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("fast_size", torch.zeros((), dtype=torch.long), persistent=False)

        self.register_buffer("slow_keys", torch.zeros(self.slow_capacity, self.key_dim), persistent=False)
        self.register_buffer("slow_values", torch.zeros(self.slow_capacity, self.value_dim), persistent=False)
        self.register_buffer("slow_scores", torch.zeros(self.slow_capacity), persistent=False)
        self.register_buffer("slow_times", torch.zeros(self.slow_capacity, dtype=torch.long), persistent=False)
        self.register_buffer("slow_ptr", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("slow_size", torch.zeros((), dtype=torch.long), persistent=False)

    def clear(self) -> None:
        self.fast_keys.zero_()
        self.fast_values.zero_()
        self.fast_scores.zero_()
        self.fast_times.zero_()
        self.fast_ptr.zero_()
        self.fast_size.zero_()

        self.slow_keys.zero_()
        self.slow_values.zero_()
        self.slow_scores.zero_()
        self.slow_times.zero_()
        self.slow_ptr.zero_()
        self.slow_size.zero_()

    def __len__(self) -> int:
        return int(self.fast_size.item()) + int(self.slow_size.item())

    def occupancy(self) -> float:
        total = self.fast_capacity + self.slow_capacity
        if total <= 0:
            return 0.0
        return (float(self.fast_size.item()) + float(self.slow_size.item())) / float(total)

    @torch.no_grad()
    def _refresh_store(
        self,
        keys_buf: torch.Tensor,
        values_buf: torch.Tensor,
        scores_buf: torch.Tensor,
        times_buf: torch.Tensor,
        size_buf: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        timestamp: int,
    ) -> torch.Tensor:
        if key.numel() == 0 or int(size_buf.item()) == 0:
            return torch.zeros(key.shape[0], device=key.device, dtype=torch.bool)

        size = int(size_buf.item())
        key = key.detach().to(device=keys_buf.device, dtype=keys_buf.dtype)
        value = value.detach().to(device=values_buf.device, dtype=values_buf.dtype)
        score = score.detach().to(device=scores_buf.device, dtype=scores_buf.dtype).reshape(-1)

        if score.numel() == 1 and key.shape[0] > 1:
            score = score.expand(key.shape[0])
        elif score.numel() != key.shape[0]:
            score = score.mean().expand(key.shape[0])

        updated = torch.zeros(key.shape[0], device=key.device, dtype=torch.bool)

        for i in range(key.shape[0]):
            bank_keys = keys_buf[:size]
            sims = torch.matmul(bank_keys, key[i])
            idx = int(torch.argmax(sims).item())
            best = float(sims[idx].item())
            if best >= self.merge_threshold:
                updated[i] = True
                alpha = float(score[i].clamp(0.0, 1.0).item())
                beta = 1.0 - alpha
                keys_buf[idx].lerp_(key[i], alpha)
                values_buf[idx].lerp_(value[i], alpha)
                scores_buf[idx].mul_(beta).add_(alpha)
                times_buf[idx].fill_(int(timestamp))

        return updated

    @torch.no_grad()
    def compact(self, bank_name: str = "fast", target_keep: Optional[int] = None) -> int:
        if bank_name == "fast":
            keys_buf = self.fast_keys
            values_buf = self.fast_values
            scores_buf = self.fast_scores
            times_buf = self.fast_times
            ptr_buf = self.fast_ptr
            size_buf = self.fast_size
        elif bank_name == "slow":
            keys_buf = self.slow_keys
            values_buf = self.slow_values
            scores_buf = self.slow_scores
            times_buf = self.slow_times
            ptr_buf = self.slow_ptr
            size_buf = self.slow_size
        else:
            raise ValueError(f"Unknown bank_name: {bank_name}")

        size = int(size_buf.item())
        if size == 0:
            return 0

        if target_keep is None:
            target_keep = size
        target_keep = max(1, min(int(target_keep), size))

        if size == 1:
            return 1

        age = float(times_buf[:size].max().item()) - times_buf[:size].to(dtype=scores_buf.dtype)
        rank = scores_buf[:size] - 0.02 * age
        order = torch.argsort(rank, descending=True)[:target_keep]

        keys_new = keys_buf[:size][order].contiguous()
        values_new = values_buf[:size][order].contiguous()
        scores_new = scores_buf[:size][order].contiguous()
        times_new = times_buf[:size][order].contiguous()

        keys_buf.zero_()
        values_buf.zero_()
        scores_buf.zero_()
        times_buf.zero_()

        keys_buf[:target_keep].copy_(keys_new)
        values_buf[:target_keep].copy_(values_new)
        scores_buf[:target_keep].copy_(scores_new)
        times_buf[:target_keep].copy_(times_new)
        size_buf.fill_(target_keep)
        ptr_buf.fill_(target_keep % keys_buf.shape[0])
        return target_keep

    @torch.no_grad()
    def _write_store(
        self,
        keys_buf: torch.Tensor,
        values_buf: torch.Tensor,
        scores_buf: torch.Tensor,
        times_buf: torch.Tensor,
        ptr_buf: torch.Tensor,
        size_buf: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        timestamp: int,
    ) -> None:
        if key.numel() == 0:
            return

        cap = keys_buf.shape[0]
        n = key.shape[0]

        key = key.detach().to(device=keys_buf.device, dtype=keys_buf.dtype)
        value = value.detach().to(device=values_buf.device, dtype=values_buf.dtype)
        score = score.detach().to(device=scores_buf.device, dtype=scores_buf.dtype).reshape(-1)

        if score.numel() == 1 and n > 1:
            score = score.expand(n)
        elif score.numel() != n:
            score = score.mean().expand(n)

        if n >= cap:
            keys_buf.copy_(key[-cap:])
            values_buf.copy_(value[-cap:])
            scores_buf.copy_(score[-cap:])
            times_buf.copy_(torch.full((cap,), int(timestamp), device=times_buf.device, dtype=times_buf.dtype))
            ptr_buf.zero_()
            size_buf.fill_(cap)
            return

        ptr = int(ptr_buf.item())
        end = ptr + n

        if end <= cap:
            keys_buf[ptr:end].copy_(key)
            values_buf[ptr:end].copy_(value)
            scores_buf[ptr:end].copy_(score)
            times_buf[ptr:end].copy_(torch.full((n,), int(timestamp), device=times_buf.device, dtype=times_buf.dtype))
        else:
            first = cap - ptr
            keys_buf[ptr:].copy_(key[:first])
            values_buf[ptr:].copy_(value[:first])
            scores_buf[ptr:].copy_(score[:first])
            times_buf[ptr:].copy_(torch.full((first,), int(timestamp), device=times_buf.device, dtype=times_buf.dtype))

            remain = n - first
            keys_buf[:remain].copy_(key[first:])
            values_buf[:remain].copy_(value[first:])
            scores_buf[:remain].copy_(score[first:])
            times_buf[:remain].copy_(torch.full((remain,), int(timestamp), device=times_buf.device, dtype=times_buf.dtype))

        ptr_buf.fill_((ptr + n) % cap)
        size_buf.fill_(min(int(size_buf.item()) + n, cap))

    @torch.no_grad()
    def add(self, key: torch.Tensor, value: torch.Tensor, score: torch.Tensor, timestamp: int) -> torch.Tensor:
        if key.ndim == 1:
            key = key.unsqueeze(0)
        if value.ndim == 1:
            value = value.unsqueeze(0)

        key = _safe_normalize(key, dim=-1)

        score = torch.as_tensor(score, device=key.device, dtype=key.dtype).reshape(-1)
        n = key.shape[0]
        if score.numel() == 1 and n > 1:
            score = score.expand(n)
        elif score.numel() != n:
            score = score.mean().expand(n)

        fast_mask = score >= self.fast_threshold
        slow_mask = score >= self.slow_threshold

        if fast_mask.any():
            refreshed = self._refresh_store(
                self.fast_keys,
                self.fast_values,
                self.fast_scores,
                self.fast_times,
                self.fast_size,
                key[fast_mask],
                value[fast_mask],
                score[fast_mask],
                timestamp,
            )
            to_write = fast_mask.clone()
            to_write[fast_mask] = ~refreshed
            if to_write.any():
                self._write_store(
                    self.fast_keys,
                    self.fast_values,
                    self.fast_scores,
                    self.fast_times,
                    self.fast_ptr,
                    self.fast_size,
                    key[to_write],
                    value[to_write],
                    score[to_write],
                    timestamp,
                )

        if slow_mask.any():
            refreshed = self._refresh_store(
                self.slow_keys,
                self.slow_values,
                self.slow_scores,
                self.slow_times,
                self.slow_size,
                key[slow_mask],
                value[slow_mask],
                score[slow_mask],
                timestamp,
            )
            to_write = slow_mask.clone()
            to_write[slow_mask] = ~refreshed
            if to_write.any():
                self._write_store(
                    self.slow_keys,
                    self.slow_values,
                    self.slow_scores,
                    self.slow_times,
                    self.slow_ptr,
                    self.slow_size,
                    key[to_write],
                    value[to_write],
                    score[to_write],
                    timestamp,
                )

        return fast_mask | slow_mask

    def retrieve(self, bank_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if bank_name == "fast":
            size = int(self.fast_size.item())
            return self.fast_keys[:size], self.fast_values[:size], self.fast_scores[:size], self.fast_times[:size], size
        if bank_name == "slow":
            size = int(self.slow_size.item())
            return self.slow_keys[:size], self.slow_values[:size], self.slow_scores[:size], self.slow_times[:size], size
        raise ValueError(f"Unknown bank_name: {bank_name}")


class MemoryConsolidator(nn.Module):
    def __init__(
        self,
        salience_threshold: float = 0.5,
        decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.salience_threshold = float(salience_threshold)
        self.decay = float(decay)

    def forward(
        self,
        bank: TraceBank,
        key: torch.Tensor,
        value: torch.Tensor,
        salience: torch.Tensor,
        timestamp: int,
        layer: int,
        metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        score = torch.as_tensor(salience, device=key.device, dtype=key.dtype)
        if score.ndim > 1:
            score = score.reshape(score.shape[0], -1).mean(dim=-1)

        if self.decay > 0.0:
            score = score * math.exp(-self.decay * max(0, int(timestamp)))

        score = torch.clamp(score, 0.0, 1.0)
        score = torch.where(score >= self.salience_threshold, score, torch.zeros_like(score))
        return bank.add(key=key, value=value, score=score, timestamp=timestamp)


class MemoryRetrieval(nn.Module):
    def __init__(
        self,
        query_dim: int,
        value_dim: int,
        *,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        temperature: float = 0.2,
        top_k: Optional[int] = 64,
    ) -> None:
        super().__init__()

        self.query_dim = int(query_dim)
        self.value_dim = int(value_dim)
        self.num_heads = int(max(1, num_heads))
        self.head_dim = int(head_dim or max(1, math.ceil(self.query_dim / self.num_heads)))
        self.temperature = float(temperature)
        self.top_k = top_k

        proj_dim = self.num_heads * self.head_dim

        self.query_proj = nn.Linear(self.query_dim, proj_dim, bias=False)
        self.key_proj = nn.Linear(self.query_dim, proj_dim, bias=False)
        self.value_proj = nn.Linear(self.value_dim, proj_dim, bias=False)
        self.out_proj = nn.Linear(proj_dim, self.value_dim, bias=False)

        self.level_router = nn.Linear(self.query_dim, 2)
        self.fast_recency = nn.Parameter(torch.tensor(0.02))
        self.slow_recency = nn.Parameter(torch.tensor(0.005))
        self.score_bias = nn.Parameter(torch.tensor(0.1))

    def _empty(self, query: torch.Tensor) -> RetrievalOutput:
        b = query.shape[0]
        context = torch.zeros(b, self.value_dim, device=query.device, dtype=query.dtype)
        empty_weights = torch.empty(b, self.num_heads, 0, device=query.device, dtype=query.dtype)
        empty_indices = torch.empty(b, self.num_heads, 0, device=query.device, dtype=torch.long)
        level_weights = torch.softmax(self.level_router(query), dim=-1)
        return RetrievalOutput(
            context=context,
            fast_context=context,
            slow_context=context,
            weights=[empty_weights, empty_weights],
            indices=[empty_indices, empty_indices],
            level_weights=level_weights,
        )

    def _retrieve_bank(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        times: torch.Tensor,
        size: int,
        current_step: int,
        recency_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if size == 0:
            b = query.shape[0]
            context = torch.zeros(b, self.value_dim, device=query.device, dtype=query.dtype)
            empty_weights = torch.empty(b, self.num_heads, 0, device=query.device, dtype=query.dtype)
            empty_indices = torch.empty(b, self.num_heads, 0, device=query.device, dtype=torch.long)
            return context, empty_weights, empty_indices

        b = query.shape[0]
        device = query.device

        q = self.query_proj(query).view(b, self.num_heads, self.head_dim)
        q = _safe_normalize(q, dim=-1)

        k = self.key_proj(keys[:size]).view(size, self.num_heads, self.head_dim)
        k = _safe_normalize(k, dim=-1)

        v = self.value_proj(values[:size]).view(size, self.num_heads, self.head_dim)

        sim = torch.einsum("bhd,nhd->bhn", q, k) / math.sqrt(self.head_dim)

        if scores is not None and scores.numel() > 0:
            sim = sim + self.score_bias * scores[:size].to(device=device, dtype=query.dtype).view(1, 1, -1)

        age = (
            torch.as_tensor(current_step, device=device, dtype=query.dtype)
            - times[:size].to(device=device, dtype=query.dtype)
        ).clamp_min(0.0)
        sim = sim - torch.clamp(recency_scale, 0.0, 1.0) * age.view(1, 1, -1)

        if self.top_k is None:
            k_top = size
        else:
            k_top = min(max(1, int(self.top_k)), size)

        top_vals, top_idx = torch.topk(sim, k=k_top, dim=-1)

        v_h = v.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        gathered = torch.gather(v_h, dim=2, index=idx_exp)

        temp = max(float(self.temperature), 1e-6)
        weights = torch.softmax(top_vals / temp, dim=-1).unsqueeze(-1)
        head_ctx = (weights * gathered).sum(dim=2)

        context = self.out_proj(head_ctx.reshape(b, -1))
        return context, weights.squeeze(-1), top_idx

    def forward(
        self,
        query: torch.Tensor,
        bank: TraceBank,
        current_step: int,
    ) -> RetrievalOutput:
        if query.ndim != 2:
            raise ValueError(f"MemoryRetrieval expects [B, D] query, got {tuple(query.shape)}")

        fast_keys, fast_values, fast_scores, fast_times, fast_size = bank.retrieve("fast")
        slow_keys, slow_values, slow_scores, slow_times, slow_size = bank.retrieve("slow")

        if fast_size == 0 and slow_size == 0:
            return self._empty(query)

        fast_context, fast_weights, fast_idx = self._retrieve_bank(
            query,
            fast_keys,
            fast_values,
            fast_scores,
            fast_times,
            fast_size,
            current_step,
            self.fast_recency,
        )

        slow_context, slow_weights, slow_idx = self._retrieve_bank(
            query,
            slow_keys,
            slow_values,
            slow_scores,
            slow_times,
            slow_size,
            current_step,
            self.slow_recency,
        )

        level_weights = torch.softmax(self.level_router(query), dim=-1)
        context = level_weights[:, :1] * fast_context + level_weights[:, 1:2] * slow_context

        return RetrievalOutput(
            context=context,
            fast_context=fast_context,
            slow_context=slow_context,
            weights=[fast_weights, slow_weights],
            indices=[fast_idx, slow_idx],
            level_weights=level_weights,
        )


class MultiScaleLatentLayer(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        step_size: float = 1.0,
        damping: float = 0.08,
        norm_groups: int = 8,
        spectral_modes: int = 8,
        propagation: Optional[nn.Module] = None,
        control_channels: int = 0,
        complex_mode: bool = False,
        boundary_mode: str = "circular",
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.step_size = float(step_size)
        self.complex_mode = bool(complex_mode)
        self.damping = nn.Parameter(torch.tensor(_softplus_inverse(damping)))

        groups = _make_groups(self.latent_channels, norm_groups)
        self.state_norm = nn.GroupNorm(groups, self.latent_channels)
        self.drive_norm = nn.GroupNorm(groups, self.latent_channels)
        self.output_norm = nn.GroupNorm(groups, self.latent_channels)

        self.write_proj = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1)
        self.write_gate = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1)

        self.control_proj = (
            nn.Conv2d(control_channels, self.latent_channels, kernel_size=1)
            if control_channels > 0
            else None
        )

        if propagation is None:
            propagation = nn.Sequential(
                nn.Conv2d(
                    self.latent_channels,
                    self.latent_channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.latent_channels,
                    bias=False,
                ),
                nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1),
            )
        self.propagation = propagation

        self.physics = PhysicsOperatorBlock(
            channels=self.latent_channels,
            hidden_channels=max(16, self.latent_channels),
            control_channels=control_channels,
            boundary_mode=boundary_mode,
            complex_mode=self.complex_mode,
        )

        self.spectral = SpectralMix2d(self.latent_channels, modes=spectral_modes)

        self.nonlinear = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1),
        )

        self.top_down_proj = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1)
        self.blend_gate = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1)
        self.residual_mix = nn.Conv2d(self.latent_channels * 4, self.latent_channels, kernel_size=1)

    def forward(
        self,
        state: torch.Tensor,
        write_signal: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
        stochastic: bool = False,
    ) -> torch.Tensor:
        _ensure_4d(state, "state")
        _ensure_4d(write_signal, "write_signal")

        if top_down is None:
            top_down = torch.zeros_like(state)

        if control is not None and self.control_proj is None:
            raise ValueError("control was provided but this layer was created without control_channels")

        control_term = self.control_proj(control) if control is not None and self.control_proj is not None else torch.zeros_like(state)

        context = self.top_down_proj(top_down)
        drive = self.drive_norm(write_signal + context + control_term)

        write_delta = torch.tanh(self.write_proj(drive))
        write_gate = torch.sigmoid(self.write_gate(drive))

        written = state + write_gate * write_delta
        base = self.state_norm(written)

        propagated = self.propagation(base)
        spectral = self.spectral(base)
        nonlinear = self.nonlinear(base)
        physics_update = self.physics(base, control=control, dt=self.step_size, stochastic=stochastic)

        damping = F.softplus(self.damping)
        update = propagated + 0.5 * spectral + nonlinear + physics_update - damping * base
        candidate = written + self.step_size * update

        blend = torch.sigmoid(self.blend_gate(self.state_norm(candidate)))
        merged = blend * candidate + (1.0 - blend) * written

        mixed = self.residual_mix(torch.cat([written, candidate, spectral, physics_update], dim=1))
        return self.output_norm(merged + mixed)


LatentOperator = MultiScaleLatentLayer


class FIMModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        trace_dim: Optional[int] = None,
        hidden_channels: int = 128,
        num_layers: int = 4,
        control_channels: int = 0,
        memory_threshold: float = 0.5,
        retrieval_temperature: float = 0.2,
        retrieval_top_k: Optional[int] = 64,
        max_traces: int = 4096,
        layer_step_sizes: Optional[Sequence[float]] = None,
        layer_dampings: Optional[Sequence[float]] = None,
        spectral_modes: int = 8,
        propagation: Optional[nn.Module] = None,
        memory_heads: int = 4,
        complex_mode: bool = False,
        boundary_mode: str = "circular",
        memory_compaction_interval: int = 16,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_latent_channels = int(latent_channels)
        self.complex_mode = bool(complex_mode)
        self.state_channels = self.base_latent_channels * (2 if self.complex_mode else 1)
        self.trace_dim = int(trace_dim or self.base_latent_channels)
        self.num_layers = int(num_layers)
        self.control_channels = int(control_channels)
        self.spectral_modes = int(spectral_modes)
        self.boundary_mode = boundary_mode
        self.memory_compaction_interval = max(0, int(memory_compaction_interval))

        if layer_step_sizes is None:
            layer_step_sizes = [1.0 for _ in range(num_layers)]
        if layer_dampings is None:
            layer_dampings = [0.08 + 0.02 * i for i in range(num_layers)]

        if len(layer_step_sizes) != num_layers:
            raise ValueError("layer_step_sizes must match num_layers")
        if len(layer_dampings) != num_layers:
            raise ValueError("layer_dampings must match num_layers")

        self.encoder = FabricEncoder(
            in_channels=in_channels,
            latent_channels=self.state_channels,
            hidden_channels=hidden_channels,
            control_channels=control_channels,
            num_blocks=3,
        )

        self.decoder = FabricDecoder(
            latent_channels=self.state_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=3,
        )

        self.salience_readout = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.state_channels, hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                MultiScaleLatentLayer(
                    latent_channels=self.state_channels,
                    step_size=layer_step_sizes[i],
                    damping=layer_dampings[i],
                    spectral_modes=spectral_modes,
                    propagation=propagation,
                    control_channels=control_channels,
                    complex_mode=self.complex_mode,
                    boundary_mode=boundary_mode,
                )
                for i in range(num_layers)
            ]
        )

        self.bottom_up = nn.ModuleList(
            [nn.Conv2d(self.state_channels, self.state_channels, kernel_size=1) for _ in range(num_layers - 1)]
        )

        self.top_down = nn.ModuleList(
            [nn.Conv2d(self.state_channels, self.state_channels, kernel_size=1) for _ in range(num_layers - 1)]
        )

        self.salience = nn.ModuleList(
            [
                SalienceScorer(
                    obs_channels=in_channels,
                    pred_channels=out_channels,
                    latent_channels=self.state_channels,
                    hidden_channels=hidden_channels,
                    threshold=memory_threshold,
                    temperature=0.25,
                )
                for _ in range(num_layers)
            ]
        )

        self.compressor = nn.ModuleList(
            [
                TraceCompressor(
                    latent_channels=self.state_channels,
                    trace_dim=self.trace_dim,
                    hidden_channels=hidden_channels,
                    normalize=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.consolidator = MemoryConsolidator(
            salience_threshold=memory_threshold,
            decay=0.0,
        )

        slow_threshold = min(0.95, memory_threshold + 0.2)

        self.retrieval = nn.ModuleList(
            [
                MemoryRetrieval(
                    query_dim=self.trace_dim,
                    value_dim=self.trace_dim,
                    num_heads=memory_heads,
                    temperature=retrieval_temperature,
                    top_k=retrieval_top_k,
                )
                for _ in range(num_layers)
            ]
        )

        self.query_proj = nn.ModuleList(
            [nn.Linear(self.trace_dim * 2, self.trace_dim) for _ in range(num_layers)]
        )

        self.retrieval_context_proj = nn.ModuleList(
            [nn.Conv2d(self.trace_dim, self.state_channels, kernel_size=1, bias=False) for _ in range(num_layers)]
        )

        self.retrieval_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.state_channels * 2, self.state_channels * 2, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(self.state_channels * 2, self.state_channels, kernel_size=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.trace_banks = nn.ModuleList(
            [
                TraceBank(
                    key_dim=self.trace_dim,
                    value_dim=self.trace_dim,
                    max_traces=max_traces,
                    fast_threshold=memory_threshold,
                    slow_threshold=slow_threshold,
                )
                for _ in range(num_layers)
            ]
        )
        self.trace_bank = self.trace_banks[0]

        self.layer_router = nn.Sequential(
            nn.Linear(num_layers * self.state_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_layers),
        )

        self.router_temperature = nn.Parameter(torch.tensor(1.0))

        groups = _make_groups(self.state_channels, 8)
        self.mixture_norm = nn.GroupNorm(groups, self.state_channels)

        self._state: Optional[torch.Tensor] = None
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

    @property
    def step_index(self) -> int:
        return int(self._step.item())

    def reset_state(self, clear_memory: bool = False) -> None:
        self._state = None
        self._step.zero_()
        if clear_memory:
            self.clear_memory()

    def clear_memory(self) -> None:
        for bank in self.trace_banks:
            bank.clear()

    def compact_memory(self, target_keep_ratio: float = 0.9) -> None:
        if self.memory_compaction_interval <= 0:
            return
        for bank in self.trace_banks:
            fast_size = int(bank.fast_size.item())
            if fast_size > 0:
                bank.compact("fast", target_keep=max(1, int(fast_size * target_keep_ratio)))
            slow_size = int(bank.slow_size.item())
            if slow_size > 0:
                bank.compact("slow", target_keep=max(1, int(slow_size * target_keep_ratio)))

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = self._state.detach()

    def _init_state(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.zeros(
            b,
            self.num_layers,
            self.state_channels,
            h,
            w,
            device=x.device,
            dtype=x.dtype,
        )

    def _prepare_state(self, x: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
        if state is None:
            if (
                self._state is None
                or self._state.shape[0] != x.shape[0]
                or self._state.shape[2:] != x.shape[2:]
            ):
                self._state = self._init_state(x)
            state = self._state

        state = _ensure_5d_state(state, self.num_layers)

        if state.shape[0] != x.shape[0] or state.shape[-2:] != x.shape[-2:]:
            raise ValueError("state must match x in batch size and spatial dimensions")

        return state.to(device=x.device, dtype=x.dtype)

    def _update_one_layer(
        self,
        layer_idx: int,
        current_state: torch.Tensor,
        write_signal: torch.Tensor,
        top_down_context: Optional[torch.Tensor],
        control: Optional[torch.Tensor],
        stochastic: bool,
    ) -> torch.Tensor:
        return self.layers[layer_idx](
            state=current_state,
            write_signal=write_signal,
            top_down=top_down_context,
            control=control,
            stochastic=stochastic,
        )

    def _layer_route_weights(self, layer_states: Sequence[torch.Tensor]) -> torch.Tensor:
        summaries: List[torch.Tensor] = []
        for s in layer_states:
            mean = s.mean(dim=(2, 3))
            std = s.flatten(2).std(dim=-1, unbiased=False)
            summaries.append(torch.cat([mean, std], dim=-1))
        summary = torch.cat(summaries, dim=1)
        logits = self.layer_router(summary)
        temp = torch.clamp(self.router_temperature, 0.2, 5.0)
        return torch.softmax(logits / temp, dim=-1)

    def _stack_layers(self, layer_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(layer_states, dim=1)

    def _memory_update_for_layer(
        self,
        layer_idx: int,
        layer_state: torch.Tensor,
        observation: torch.Tensor,
        prediction: torch.Tensor,
        previous_state: torch.Tensor,
        timestamp: int,
        update_memory: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = layer_state.flatten(2).var(dim=-1, unbiased=False).mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

        salience_out = self.salience[layer_idx](
            observation=observation,
            prediction=prediction,
            fabric_before=previous_state,
            fabric_after=layer_state,
            uncertainty=uncertainty,
        )

        compressed = self.compressor[layer_idx](layer_state)

        stored_mask = torch.zeros(observation.shape[0], device=observation.device, dtype=torch.bool)
        if update_memory:
            stored_mask = self.consolidator(
                bank=self.trace_banks[layer_idx],
                key=compressed.key,
                value=compressed.value,
                salience=salience_out.score,
                timestamp=timestamp,
                layer=layer_idx,
                metadata={
                    "step": timestamp,
                    "layer": layer_idx,
                },
            )

        query_input = torch.cat([compressed.key, compressed.context], dim=-1)
        query = self.query_proj[layer_idx](query_input)

        retrieval_out = self.retrieval[layer_idx](query, self.trace_banks[layer_idx], timestamp)
        return (
            salience_out.score,
            salience_out.gate,
            compressed,
            retrieval_out,
            stored_mask,
            query,
        )

    def forward(
        self,
        x: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        cache_state: bool = True,
        detach_cache: bool = True,
        stochastic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, FIMStepOutput]:
        _ensure_4d(x, "x")

        if control is not None:
            if self.control_channels == 0:
                raise ValueError("control was provided but control_channels was initialized as 0")
            _ensure_4d(control, "control")
            if control.shape[0] != x.shape[0] or control.shape[2:] != x.shape[2:]:
                raise ValueError("control must match x in batch size and spatial dimensions")

        stochastic = self.training if stochastic is None else bool(stochastic)

        state = self._prepare_state(x, state)
        latent = self.encoder(x, control=control)

        provisional_states: List[torch.Tensor] = []
        drives: List[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                drive = latent
            else:
                drive = self.bottom_up[layer_idx - 1](provisional_states[layer_idx - 1])
                drive = drive + 0.5 * latent

            drives.append(drive)

            provisional_state = self._update_one_layer(
                layer_idx=layer_idx,
                current_state=state[:, layer_idx],
                write_signal=drive,
                top_down_context=None,
                control=control,
                stochastic=stochastic,
            )
            provisional_states.append(provisional_state)

        refined_states = list(provisional_states)
        for layer_idx in reversed(range(self.num_layers - 1)):
            top_context = self.top_down[layer_idx](refined_states[layer_idx + 1])
            refined_states[layer_idx] = self._update_one_layer(
                layer_idx=layer_idx,
                current_state=refined_states[layer_idx],
                write_signal=drives[layer_idx],
                top_down_context=top_context,
                control=control,
                stochastic=stochastic,
            )

        layer_pre_retrieval_states = list(refined_states)

        retrieval_contexts: List[torch.Tensor] = []
        fast_contexts: List[torch.Tensor] = []
        slow_contexts: List[torch.Tensor] = []
        retrieval_weights: List[Sequence[torch.Tensor]] = []
        retrieval_indices: List[Sequence[torch.Tensor]] = []
        retrieval_level_weights: List[torch.Tensor] = []
        stored_masks: List[torch.Tensor] = []
        salience_scores: List[torch.Tensor] = []
        salience_gates: List[torch.Tensor] = []
        fused_layers: List[torch.Tensor] = []

        for layer_idx, layer_state in enumerate(refined_states):
            aux_pred = self.salience_readout[layer_idx](layer_state)

            uncertainty = layer_state.flatten(2).var(dim=-1, unbiased=False).mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

            salience_out = self.salience[layer_idx](
                observation=x,
                prediction=aux_pred,
                fabric_before=state[:, layer_idx],
                fabric_after=layer_state,
                uncertainty=uncertainty,
            )

            compressed = self.compressor[layer_idx](layer_state)

            stored_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
            if update_memory:
                stored_mask = self.consolidator(
                    bank=self.trace_banks[layer_idx],
                    key=compressed.key,
                    value=compressed.value,
                    salience=salience_out.score,
                    timestamp=self.step_index,
                    layer=layer_idx,
                    metadata={
                        "step": self.step_index,
                        "layer": layer_idx,
                    },
                )

            query_input = torch.cat([compressed.key, compressed.context], dim=-1)
            query = self.query_proj[layer_idx](query_input)

            fast_size = int(self.trace_banks[layer_idx].fast_size.item())
            slow_size = int(self.trace_banks[layer_idx].slow_size.item())
            has_memory = (fast_size + slow_size) > 0

            retrieval_out = self.retrieval[layer_idx](query, self.trace_banks[layer_idx], self.step_index)
            retrieved_context = retrieval_out.context

            if has_memory:
                retrieval_map = _spatial_expand(retrieved_context, layer_state.shape[-2:])
                retrieval_map = self.retrieval_context_proj[layer_idx](retrieval_map)
                fused_candidate = self.retrieval_fuse[layer_idx](torch.cat([layer_state, retrieval_map], dim=1))
                retrieval_strength = salience_out.gate.view(-1, 1, 1, 1)
                fused_state = layer_state + retrieval_strength * (fused_candidate - layer_state)
            else:
                fused_state = layer_state

            retrieval_contexts.append(retrieved_context)
            fast_contexts.append(retrieval_out.fast_context)
            slow_contexts.append(retrieval_out.slow_context)
            retrieval_weights.append(retrieval_out.weights)
            retrieval_indices.append(retrieval_out.indices)
            retrieval_level_weights.append(retrieval_out.level_weights)
            stored_masks.append(stored_mask)
            salience_scores.append(salience_out.score)
            salience_gates.append(salience_out.gate)
            fused_layers.append(fused_state)

        stacked_fused = self._stack_layers(fused_layers)
        layer_mixture = self._layer_route_weights(fused_layers)

        mixture = torch.sum(
            layer_mixture.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked_fused,
            dim=1,
        )
        mixture = self.mixture_norm(mixture)
        prediction = self.decoder(mixture)

        stacked_state = stacked_fused
        stacked_pre_retrieval = self._stack_layers(layer_pre_retrieval_states)
        pre_mixture = torch.sum(
            layer_mixture.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked_pre_retrieval,
            dim=1,
        )
        pre_mixture = self.mixture_norm(pre_mixture)
        stacked_salience = torch.stack(salience_scores, dim=1)
        stacked_gates = torch.stack(salience_gates, dim=1)
        stacked_masks = torch.stack(stored_masks, dim=1)
        stacked_contexts = torch.stack(retrieval_contexts, dim=1)
        stacked_fast_contexts = torch.stack(fast_contexts, dim=1)
        stacked_slow_contexts = torch.stack(slow_contexts, dim=1)
        stacked_level_weights = torch.stack(retrieval_level_weights, dim=1)

        if cache_state:
            self._state = stacked_state.detach() if detach_cache else stacked_state
        self._step += 1

        if self.memory_compaction_interval > 0 and self.step_index % self.memory_compaction_interval == 0:
            self.compact_memory()

        output = FIMStepOutput(
            prediction=prediction,
            state=mixture,
            stacked_state=stacked_state,
            pre_retrieval_state=pre_mixture,
            stacked_pre_retrieval_state=stacked_pre_retrieval,
            layer_states=fused_layers,
            layer_pre_retrieval_states=layer_pre_retrieval_states,
            salience_score=stacked_salience,
            salience_gate=stacked_gates,
            retrieval_context=stacked_contexts,
            fast_retrieval_context=stacked_fast_contexts,
            slow_retrieval_context=stacked_slow_contexts,
            retrieval_weights=retrieval_weights,
            retrieval_indices=retrieval_indices,
            retrieval_level_weights=stacked_level_weights,
            stored_mask=stacked_masks,
            layer_mixture=layer_mixture,
            layer_weights=layer_mixture,
        )

        return prediction, output

    def step(
        self,
        x: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        cache_state: bool = True,
        detach_cache: bool = True,
        stochastic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, FIMStepOutput]:
        return self.forward(
            x=x,
            control=control,
            state=state,
            update_memory=update_memory,
            cache_state=cache_state,
            detach_cache=detach_cache,
            stochastic=stochastic,
        )

    def rollout(
        self,
        x_seq: torch.Tensor,
        control_seq: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        cache_state: bool = True,
        detach_cache: bool = True,
        stochastic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[FIMStepOutput]]:
        if x_seq.ndim != 5:
            raise ValueError(f"x_seq must have shape [B, T, C, H, W], got {tuple(x_seq.shape)}")
        if control_seq is not None and control_seq.ndim != 5:
            raise ValueError(f"control_seq must have shape [B, T, C, H, W], got {tuple(control_seq.shape)}")

        preds: List[torch.Tensor] = []
        outs: List[FIMStepOutput] = []
        state = None

        for t in range(x_seq.shape[1]):
            ctrl = None if control_seq is None else control_seq[:, t]
            pred, out = self.forward(
                x=x_seq[:, t],
                control=ctrl,
                state=state,
                update_memory=update_memory,
                cache_state=cache_state,
                detach_cache=detach_cache,
                stochastic=stochastic,
            )
            state = out.stacked_state
            preds.append(pred.unsqueeze(1))
            outs.append(out)

        return torch.cat(preds, dim=1), outs

    def physics_residual(
        self,
        x: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, out = self.forward(
            x=x,
            control=control,
            state=state,
            update_memory=False,
            cache_state=False,
            stochastic=False,
        )
        return out.state - out.pre_retrieval_state

    def memory_summary(self) -> List[dict]:
        summary: List[dict] = []
        for i, bank in enumerate(self.trace_banks):
            summary.append(
                {
                    "layer": i,
                    "fast_size": int(bank.fast_size.item()),
                    "slow_size": int(bank.slow_size.item()),
                    "fast_capacity": bank.fast_capacity,
                    "slow_capacity": bank.slow_capacity,
                }
            )
        return summary

    def state_summary(self) -> dict:
        if self._state is None:
            return {
                "step": self.step_index,
                "has_state": False,
                "memory": self.memory_summary(),
            }

        return {
            "step": self.step_index,
            "has_state": True,
            "state_shape": tuple(self._state.shape),
            "state_norm": float(self._state.norm().item()),
            "memory": self.memory_summary(),
            "memory_occupancy": [bank.occupancy() for bank in self.trace_banks],
        }


__all__ = [
    "LatentOperator",
    "SpectralMix2d",
    "PhysicsOperatorBlock",
    "SalienceOutput",
    "CompressionOutput",
    "RetrievalOutput",
    "TraceBank",
    "MemoryConsolidator",
    "MemoryRetrieval",
    "SalienceScorer",
    "TraceCompressor",
    "MultiScaleLatentLayer",
    "FIMLayerOutput",
    "FIMStepOutput",
    "FIMModel",
]