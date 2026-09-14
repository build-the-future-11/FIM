from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BenchmarkBase


def laplacian_periodic(u: torch.Tensor) -> torch.Tensor:
    return (
        -4.0 * u
        + torch.roll(u, 1, dims=-2)
        + torch.roll(u, -1, dims=-2)
        + torch.roll(u, 1, dims=-1)
        + torch.roll(u, -1, dims=-1)
    )


def laplacian_reflect(u: torch.Tensor) -> torch.Tensor:
    u_pad = F.pad(u, (1, 1, 1, 1), mode="reflect")
    return (
        -4.0 * u
        + u_pad[:, :, 1:-1, :-2]
        + u_pad[:, :, 1:-1, 2:]
        + u_pad[:, :, :-2, 1:-1]
        + u_pad[:, :, 2:, 1:-1]
    )


@dataclass
class WaveConfig:
    speed: float = 1.0
    damping: float = 0.01
    dt: float = 0.01
    steps: int = 100
    stability_scale: float = 0.99
    boundary: str = "periodic"
    spectral: bool = False
    forcing_scale: float = 0.0
    noise_scale: float = 0.0
    normalize: bool = True
    state_clip: float | None = 10.0
    height: int = 32
    width: int = 32
    channels: int = 1


class WaveBenchmark(BenchmarkBase):
    def __init__(self, config: WaveConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_speed = nn.Parameter(torch.tensor(float(config.speed)))
        self.raw_damping = nn.Parameter(torch.tensor(float(config.damping)))
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_std", torch.tensor(1.0))
        self.register_buffer("norm_initialized", torch.tensor(0, dtype=torch.long))
        self._dimension = int(config.channels * config.height * config.width)

    @property
    def default_steps(self) -> int:
        return int(self.config.steps)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (int(self.config.channels), int(self.config.height), int(self.config.width))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dt(self) -> float:
        return float(self.config.dt)

    @property
    def stochastic(self) -> bool:
        return float(self.config.noise_scale) > 0.0

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        if self.config.boundary == "reflect":
            return laplacian_reflect(u)
        return laplacian_periodic(u)

    def spectral_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        b, c, h, w = u.shape
        uf = torch.fft.rfft2(u, norm="ortho")
        ky = torch.fft.fftfreq(h, device=u.device, dtype=u.dtype).view(h, 1)
        kx = torch.fft.rfftfreq(w, device=u.device, dtype=u.dtype).view(1, -1)
        k2 = (kx**2 + ky**2).unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft2(-k2 * uf, s=(h, w), norm="ortho")

    def effective_speed(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        speed = F.softplus(self.raw_speed.to(device=device, dtype=dtype))
        cfl_max = self.config.stability_scale / max(math.sqrt(2.0) * self.config.dt, 1e-8)
        return torch.clamp(speed, 0.0, cfl_max)

    def effective_damping(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        damp = F.softplus(self.raw_damping.to(device=device, dtype=dtype))
        damp_max = self.config.stability_scale / max(self.config.dt, 1e-8)
        return torch.clamp(damp, 0.0, damp_max)

    def wave_operator(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        c = self.effective_speed(u.device, u.dtype)
        gamma = self.effective_damping(u.device, u.dtype)

        lap = self.spectral_laplacian(u) if self.config.spectral else self.laplacian(u)
        a = (c ** 2) * lap - gamma * v

        if self.config.forcing_scale > 0.0:
            a = a + self.config.forcing_scale * torch.randn_like(u)

        return a

    def step(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = self.config.dt

        a = self.wave_operator(u, v)
        v_half = v + 0.5 * dt * a
        u_next = u + dt * v_half
        a_next = self.wave_operator(u_next, v_half)
        v_next = v_half + 0.5 * dt * a_next

        if self.config.noise_scale > 0.0:
            u_next = u_next + self.config.noise_scale * torch.randn_like(u_next)

        s = torch.clamp(torch.tensor(self.config.stability_scale, device=u.device, dtype=u.dtype), 0.0, 1.0)
        u_next = s * u_next
        v_next = s * v_next

        if self.config.state_clip is not None:
            u_next = torch.clamp(u_next, -self.config.state_clip, self.config.state_clip)
            v_next = torch.clamp(v_next, -self.config.state_clip, self.config.state_clip)

        return u_next, v_next

    def rollout(
        self,
        u0: torch.Tensor,
        v0: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = self.default_steps if steps is None else int(steps)
        if v0 is None:
            v0 = torch.zeros_like(u0)

        traj = torch.zeros(
            u0.shape[0],
            steps + 1,
            *u0.shape[1:],
            device=u0.device,
            dtype=u0.dtype,
        )

        traj[:, 0] = u0
        u, v = u0, v0

        for t in range(steps):
            u, v = self.step(u, v)
            traj[:, t + 1] = u

        self._update_norm(traj)
        if self.config.normalize:
            traj = self._normalize(traj)

        return traj

    def _update_norm(self, traj: torch.Tensor) -> None:
        if self.config.normalize and self.norm_initialized.item() == 0:
            mean = traj.mean()
            std = traj.std().clamp_min(1e-6)
            self.running_mean.copy_(mean)
            self.running_std.copy_(std)
            self.norm_initialized.fill_(1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize:
            return x
        return (x - self.running_mean) / self.running_std

    @staticmethod
    def sample_initial_state(
        batch_size: int,
        height: int,
        width: int,
        channels: int = 1,
        device: str | torch.device = "cpu",
        mode: str = "gaussian",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)

        if mode == "impulse":
            u = torch.zeros(batch_size, channels, height, width, device=device)
            v = torch.zeros_like(u)
            cx = torch.randint(0, height, (batch_size, channels), device=device)
            cy = torch.randint(0, width, (batch_size, channels), device=device)
            b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, channels)
            c_idx = torch.arange(channels, device=device).view(1, -1).expand(batch_size, -1)
            u[b_idx, c_idx, cx, cy] = 1.0
        elif mode == "gaussian":
            u = torch.randn(batch_size, channels, height, width, device=device) * 0.1
            v = torch.randn_like(u) * 0.1
        elif mode == "smooth":
            u = torch.randn(batch_size, channels, height, width, device=device)
            u = F.avg_pool2d(u, kernel_size=5, stride=1, padding=2)
            v = torch.zeros_like(u)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return u, v

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
        mode: str = "gaussian",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            u0, v0 = self.sample_initial_state(
                batch_size,
                self.config.height,
                self.config.width,
                channels=self.config.channels,
                device=device,
                mode=mode,
            )
            traj = self.rollout(u0, v0, steps)

        x = traj[:, :-1]
        y = traj[:, 1:]
        return x.contiguous(), y.contiguous()

    def metrics(self) -> dict:
        return {
            "name": "WaveBenchmark",
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": False,
            "stochastic": self.stochastic,
            "dt": self.dt,
            "dimension": self.dimension,
            "speed": float(F.softplus(self.raw_speed).item()),
            "damping": float(F.softplus(self.raw_damping).item()),
            "boundary": self.config.boundary,
            "spectral": bool(self.config.spectral),
            "stability_scale": float(self.config.stability_scale),
        }