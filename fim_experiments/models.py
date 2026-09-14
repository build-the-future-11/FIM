from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(c: int, max_groups: int = 8):
    if c <= 0:
        raise ValueError("channels must be positive")
    g = min(max_groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return g


def _safe_norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, dim=dim, eps=eps)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        g = _norm_groups(channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + self.scale * h


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.out = nn.Conv2d(hidden, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.out(x)


class SpectralMix(nn.Module):
    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.modes = int(modes)
        self.weight = nn.Parameter(torch.randn(channels, modes, modes, 2) * 0.02)
        self.gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")

        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        out = torch.zeros_like(xf)
        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())

        out[:, :, :mh, :mw] = xf[:, :, :mh, :mw] * weight.unsqueeze(0)

        mixed = torch.fft.irfft2(out, s=(h, w), norm="ortho")
        g = torch.clamp(self.gain, 0.0, 1.0)

        return x + g * mixed


class LatentDynamics(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.diffusion = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.advection = nn.Conv2d(channels, channels, 1)
        self.reaction = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.spectral = SpectralMix(channels)

        self.damping = nn.Parameter(torch.tensor(-3.0))
        self.step = nn.Parameter(torch.tensor(1.0))

        g = _norm_groups(channels)
        self.norm = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        base = self.norm(x)

        diff = self.diffusion(base)
        adv = self.advection(base)
        react = self.reaction(base)
        spec = self.spectral(base)

        damping = F.softplus(self.damping)

        update = diff + adv + react + 0.5 * spec - damping * base
        step = torch.clamp(self.step, 0.01, 2.0)

        return x + step * update


class Salience(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        score_map = torch.sigmoid(self.net(x))
        mean = score_map.mean(dim=(1, 2, 3))
        var = score_map.var(dim=(1, 2, 3), unbiased=False)
        return mean + 0.5 * var


class TraceCompressor(nn.Module):
    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.key = nn.Linear(channels, dim)
        self.value = nn.Linear(channels, dim)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        z = self.pool(x).flatten(1)
        k = _safe_norm(self.key(z))
        v = _safe_norm(self.value(z))
        return k, v


class Retrieval(nn.Module):
    def __init__(self, dim: int, heads: int = 4, top_k: int = 64):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.top_k = top_k

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.temperature = nn.Parameter(torch.tensor(0.2))

    def forward(self, q, keys, values):
        if q.ndim == 1:
            q = q.unsqueeze(0)

        if q.ndim != 2:
            raise ValueError(f"Expected q [B, D], got {tuple(q.shape)}")

        if keys.numel() == 0 or values.numel() == 0:
            return torch.zeros_like(q)

        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must both be [N, D]")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if keys.shape[1] != self.dim or values.shape[1] != self.dim:
            raise ValueError("keys/values dim mismatch")

        b = q.shape[0]

        qh = self.q_proj(q).view(b, self.heads, self.head_dim)
        kh = self.k_proj(keys).view(-1, self.heads, self.head_dim)
        vh = self.v_proj(values).view(-1, self.heads, self.head_dim)

        qh = _safe_norm(qh, dim=-1)
        kh = _safe_norm(kh, dim=-1)

        sim = torch.einsum("bhd,nhd->bhn", qh, kh)

        k = min(self.top_k, sim.shape[-1])
        if k <= 0:
            return torch.zeros_like(q)

        topv, topi = torch.topk(sim, k=k, dim=-1)

        vh = vh.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
        gather_i = topi.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        gathered = torch.gather(vh, 2, gather_i)

        temp = torch.clamp(self.temperature, 0.05, 1.0)
        w = torch.softmax(topv / temp, dim=-1).unsqueeze(-1)

        ctx = (w * gathered).sum(dim=2)
        return self.out(ctx.reshape(b, -1))