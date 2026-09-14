from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn


def _as_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x.view(1, 1)
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def _normalize_if_needed(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return x
    return x / (x.abs().sum(dim=-1, keepdim=True) + 1e-8)


def _periodic_laplacian_1d(u: torch.Tensor) -> torch.Tensor:
    return torch.roll(u, shifts=1, dims=-1) - 2.0 * u + torch.roll(u, shifts=-1, dims=-1)


def _periodic_laplacian_2d(u: torch.Tensor) -> torch.Tensor:
    return (
        -4.0 * u
        + torch.roll(u, shifts=1, dims=-2)
        + torch.roll(u, shifts=-1, dims=-2)
        + torch.roll(u, shifts=1, dims=-1)
        + torch.roll(u, shifts=-1, dims=-1)
    )


def _rk4_step(x: torch.Tensor, rhs_fn, dt: float) -> torch.Tensor:
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def fractional_weights(
    alpha: float,
    steps: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    normalize: bool = True,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    w = torch.zeros(steps, device=device, dtype=dtype)
    w[0] = 1.0
    for k in range(1, steps):
        w[k] = -w[k - 1] * (alpha - (k - 1)) / k
    return _normalize_if_needed(w, normalize)


@dataclass
class BenchmarkMetrics:
    name: str
    state_shape: Tuple[int, ...]
    horizon: int
    is_non_markovian: bool
    stochastic: bool
    dt: float
    dimension: int


class UnifiedBenchmark(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Sample a protocol-valid initial state."""

    @abstractmethod
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """Advance a batch of states by one integration step."""

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        traj = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            traj.append(x)
        return torch.stack(traj, dim=1)

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        return {
            "name": type(self).__name__,
            "state_shape": self.state_shape,
            "horizon": self.default_steps,
            "is_non_markovian": False,
            "stochastic": False,
            "dt": getattr(self, "dt", 0.0),
            "dimension": getattr(self, "dimension", 0),
        }


@dataclass
class Lorenz96Config:
    dimension: int = 32
    forcing: float = 8.0
    dt: float = 0.01
    steps: int = 100
    burn_in: int = 200
    process_noise: float = 0.0
    observation_noise: float = 0.0
    stochastic: bool = False
    init_scale: float = 1.0


class Lorenz96Benchmark(UnifiedBenchmark):
    def __init__(self, config: Lorenz96Config) -> None:
        super().__init__()
        if config.dimension < 4:
            raise ValueError("Lorenz-96 dimension must be at least 4.")
        self.config = config
        self.dimension = config.dimension
        self.dt = config.dt
        self.default_steps = config.steps
        self.state_shape = (config.dimension,)
        self.stochastic = config.stochastic

    def rhs(self, x: torch.Tensor) -> torch.Tensor:
        xp1 = torch.roll(x, shifts=-1, dims=-1)
        xm1 = torch.roll(x, shifts=1, dims=-1)
        xm2 = torch.roll(x, shifts=2, dims=-1)
        return (xp1 - xm2) * xm1 - x + self.config.forcing

    def step_rk4(self, x: torch.Tensor) -> torch.Tensor:
        return _rk4_step(x, self.rhs, self.config.dt)

    def step_sde(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt
        drift = self.rhs(x)
        noise = torch.randn_like(x)
        return x + drift * dt + self.config.process_noise * math.sqrt(dt) * noise

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        if self.config.stochastic:
            x_next = self.step_sde(x)
        else:
            x_next = self.step_rk4(x)
        if (not self.config.stochastic) and self.config.process_noise > 0.0:
            x_next = x_next + self.config.process_noise * torch.randn_like(x_next)
        return x_next

    def burn_in_state(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.config.burn_in):
            x = self.step(x)
        return x

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        x = self.config.init_scale * torch.randn(batch_size, self.config.dimension, device=device)
        return self.burn_in_state(x)

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        traj = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            traj.append(x)
        traj = torch.stack(traj, dim=1)
        if self.config.observation_noise > 0.0:
            traj = traj + self.config.observation_noise * torch.randn_like(traj)
        return traj

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "Lorenz96Benchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": self.config.stochastic,
                "dt": self.config.dt,
                "dimension": self.config.dimension,
                "forcing": self.config.forcing,
                "burn_in": self.config.burn_in,
            }
        )
        return base


@dataclass
class BurgersConfig:
    dimension: int = 64
    viscosity: float = 0.01
    dt: float = 0.01
    steps: int = 100
    init_scale: float = 1.0
    forcing: float = 0.0
    stochastic: bool = False
    process_noise: float = 0.0


class BurgersBenchmark(UnifiedBenchmark):
    def __init__(self, config: BurgersConfig) -> None:
        super().__init__()
        self.config = config
        self.dimension = config.dimension
        self.dt = config.dt
        self.default_steps = config.steps
        self.state_shape = (config.dimension,)
        self.stochastic = config.stochastic

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        ux = 0.5 * (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1))
        uxx = _periodic_laplacian_1d(u)
        return -u * ux + self.config.viscosity * uxx + self.config.forcing

    def step(self, state: torch.Tensor) -> torch.Tensor:
        u = state
        u_next = _rk4_step(u, self.rhs, self.config.dt)
        if self.config.stochastic and self.config.process_noise > 0.0:
            u_next = u_next + self.config.process_noise * torch.randn_like(u_next)
        return u_next

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        x = torch.linspace(0, 2 * math.pi, self.dimension, device=device)
        base = torch.sin(x) + 0.5 * torch.sin(2 * x) + 0.25 * torch.sin(3 * x)
        base = base.unsqueeze(0).repeat(batch_size, 1)
        noise = 0.1 * torch.randn(batch_size, self.dimension, device=device)
        return self.config.init_scale * (base + noise)

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "BurgersBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": self.config.stochastic,
                "dt": self.config.dt,
                "dimension": self.config.dimension,
                "viscosity": self.config.viscosity,
            }
        )
        return base


@dataclass
class ReactionDiffusionConfig:
    height: int = 64
    width: int = 64
    channels: int = 2
    dt: float = 1e-2
    steps: int = 100
    diff_u: float = 0.16
    diff_v: float = 0.08
    feed: float = 0.060
    kill: float = 0.062
    init_scale: float = 0.1
    stochastic: bool = False
    process_noise: float = 0.0


class ReactionDiffusionBenchmark(UnifiedBenchmark):
    def __init__(self, config: ReactionDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.dt = config.dt
        self.default_steps = config.steps
        self.state_shape = (config.channels, config.height, config.width)
        self.dimension = config.height * config.width * config.channels
        self.stochastic = config.stochastic

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0:1]
        v = state[:, 1:2]

        lap_u = _periodic_laplacian_2d(u)
        lap_v = _periodic_laplacian_2d(v)

        uvv = u * v * v
        du = self.config.diff_u * lap_u - uvv + self.config.feed * (1.0 - u)
        dv = self.config.diff_v * lap_v + uvv - (self.config.feed + self.config.kill) * v
        return torch.cat([du, dv], dim=1)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        nxt = _rk4_step(state, self.rhs, self.config.dt)
        if self.config.stochastic and self.config.process_noise > 0.0:
            nxt = nxt + self.config.process_noise * torch.randn_like(nxt)
        return nxt

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        u = torch.ones(batch_size, 1, self.config.height, self.config.width, device=device)
        v = torch.zeros_like(u)

        r = min(self.config.height, self.config.width) // 10
        c0 = self.config.height // 2
        c1 = self.config.width // 2
        u[:, :, c0 - r : c0 + r, c1 - r : c1 + r] = 0.5
        v[:, :, c0 - r : c0 + r, c1 - r : c1 + r] = 0.25
        noise = self.config.init_scale * torch.randn(batch_size, 2, self.config.height, self.config.width, device=device)
        return torch.clamp(torch.cat([u, v], dim=1) + noise, 0.0, 1.0)

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "ReactionDiffusionBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": self.config.stochastic,
                "dt": self.config.dt,
                "dimension": self.dimension,
                "diff_u": self.config.diff_u,
                "diff_v": self.config.diff_v,
                "feed": self.config.feed,
                "kill": self.config.kill,
            }
        )
        return base


@dataclass
class KuramotoConfig:
    n_oscillators: int = 64
    coupling: float = 1.0
    dt: float = 0.01
    steps: int = 100
    natural_frequency_scale: float = 0.5
    init_scale: float = 1.0
    stochastic: bool = False
    process_noise: float = 0.0


class KuramotoBenchmark(UnifiedBenchmark):
    def __init__(self, config: KuramotoConfig) -> None:
        super().__init__()
        self.config = config
        self.dimension = config.n_oscillators
        self.dt = config.dt
        self.default_steps = config.steps
        self.state_shape = (config.n_oscillators,)
        self.stochastic = config.stochastic
        self._omega: Optional[torch.Tensor] = None

    def _natural_frequencies(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._omega is None or self._omega.shape[0] != batch_size or self._omega.device != device:
            self._omega = self.config.natural_frequency_scale * torch.randn(batch_size, self.config.n_oscillators, device=device)
        return self._omega

    def rhs(self, theta: torch.Tensor) -> torch.Tensor:
        batch_size = theta.shape[0]
        device = theta.device
        omega = self._natural_frequencies(batch_size, device)
        pairwise = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        coupling = torch.sin(-pairwise).mean(dim=-1)
        return omega + self.config.coupling * coupling

    def step(self, state: torch.Tensor) -> torch.Tensor:
        theta = state
        theta_next = _rk4_step(theta, self.rhs, self.config.dt)
        theta_next = torch.remainder(theta_next, 2 * math.pi)
        if self.config.stochastic and self.config.process_noise > 0.0:
            theta_next = theta_next + self.config.process_noise * torch.randn_like(theta_next)
        return torch.remainder(theta_next, 2 * math.pi)

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        self._omega = self.config.natural_frequency_scale * torch.randn(batch_size, self.config.n_oscillators, device=device)
        return 2 * math.pi * torch.rand(batch_size, self.config.n_oscillators, device=device)

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "KuramotoBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": self.config.stochastic,
                "dt": self.config.dt,
                "dimension": self.dimension,
                "coupling": self.config.coupling,
                "natural_frequency_scale": self.config.natural_frequency_scale,
            }
        )
        return base


@dataclass
class FractionalConfig:
    alpha: float = 0.8
    diffusivity: float = 0.05
    dt: float = 0.01
    history: int = 64
    steps: int = 100
    height: int = 32
    width: int = 32
    channels: int = 1
    spectral: bool = True
    init_scale: float = 0.1
    normalize_history_weights: bool = True


class FractionalDiffusionBenchmark(UnifiedBenchmark):
    def __init__(self, config: FractionalConfig) -> None:
        super().__init__()
        self.config = config
        self.dt = config.dt
        self.default_steps = config.steps
        self.state_shape = (config.channels, config.height, config.width)
        self.dimension = config.channels * config.height * config.width
        self.stochastic = False

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        return _periodic_laplacian_2d(u)

    def spectral_diffusion(self, u: torch.Tensor) -> torch.Tensor:
        h, w = u.shape[-2:]
        fft = torch.fft.rfft2(u, dim=(-2, -1), norm="ortho")
        fy = torch.fft.fftfreq(h, device=u.device, dtype=u.dtype).view(*([1] * (u.ndim - 2)), h, 1)
        fx = torch.fft.rfftfreq(w, device=u.device, dtype=u.dtype).view(*([1] * (u.ndim - 2)), 1, w // 2 + 1)
        k2 = fy**2 + fx**2
        decay = torch.exp(-self.config.diffusivity * k2 * self.config.dt)
        return torch.fft.irfft2(fft * decay, s=(h, w), dim=(-2, -1), norm="ortho")

    def step(self, state: torch.Tensor, history: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        if history is None:
            history = [state]
        hlen = min(len(history), self.config.history)
        if hlen <= 0:
            raise ValueError("history must contain at least one tensor")
        hist = list(history)[-hlen:]
        weights = fractional_weights(
            self.config.alpha,
            hlen,
            device=hist[0].device,
            dtype=hist[0].dtype,
            normalize=self.config.normalize_history_weights,
        )
        frac_state = torch.zeros_like(hist[0])
        for w, u in zip(weights, reversed(hist)):
            frac_state = frac_state + w * u
        if self.config.spectral:
            return self.spectral_diffusion(frac_state)
        return frac_state + self.config.dt * self.config.diffusivity * self.laplacian(frac_state)

    def rollout(self, u0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        history: List[torch.Tensor] = [u0]
        traj = [u0]
        u = u0
        for _ in range(steps):
            u = self.step(u, history=history)
            history.append(u)
            traj.append(u)
        return torch.stack(traj, dim=1)

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        return self.config.init_scale * torch.randn(
            batch_size,
            self.config.channels,
            self.config.height,
            self.config.width,
            device=device,
        )

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(u0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "FractionalDiffusionBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": True,
                "stochastic": False,
                "dt": self.config.dt,
                "dimension": self.dimension,
                "alpha": self.config.alpha,
                "diffusivity": self.config.diffusivity,
                "history": self.config.history,
                "spectral": self.config.spectral,
            }
        )
        return base


@dataclass
class LevyStableConfig:
    alpha: float = 1.5
    beta: float = 0.0
    scale: float = 1.0
    loc: float = 0.0
    dim: int = 1
    steps: int = 100


class LevyStableProcess(UnifiedBenchmark):
    def __init__(self, config: LevyStableConfig) -> None:
        super().__init__()
        self.config = config
        self.dt = 1.0
        self.default_steps = config.steps
        self.dimension = config.dim
        self.state_shape = (config.dim,)
        self.stochastic = True

    def sample_increment(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        a = float(self.config.alpha)
        b = float(self.config.beta)
        scale = float(self.config.scale)
        device = _as_device(device)

        if not (0 < a <= 2):
            raise ValueError("alpha must be in (0, 2].")

        U = torch.empty(batch_size, self.config.dim, device=device, dtype=dtype).uniform_(
            -math.pi / 2, math.pi / 2
        )
        W = torch.empty(batch_size, self.config.dim, device=device, dtype=dtype).exponential_()

        if abs(a - 1.0) > 1e-6:
            tan_term = math.tan(math.pi * a / 2)
            phi = math.atan(b * tan_term) / a
            S = (1 + (b * tan_term) ** 2) ** (1 / (2 * a))
            num = torch.sin(a * (U + phi))
            den = torch.cos(U).clamp_min(1e-8) ** (1 / a)
            frac = num / den
            term = (torch.cos(U - a * (U + phi)).clamp_min(1e-8) / (W + 1e-8)) ** ((1 - a) / a)
            X = S * frac * term
        else:
            X = (2 / math.pi) * (
                (math.pi / 2 + b * U) * torch.tan(U)
                - b * torch.log(
                    (
                        (math.pi / 2) * W * torch.cos(U).clamp_min(1e-8)
                    )
                    / (math.pi / 2 + b * U).clamp_min(1e-8)
                    + 1e-8
                )
            )

        return scale * X + self.config.loc

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.sample_increment(state.shape[0], device=state.device, dtype=state.dtype)

    def sample_initial_state(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device = _as_device(device)
        return torch.zeros(batch_size, self.config.dim, device=device)

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        traj = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            traj.append(x)
        return torch.stack(traj, dim=1)

    def generate_batch(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "LevyStableProcess",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": False,
                "stochastic": True,
                "dt": 1.0,
                "dimension": self.dimension,
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "scale": self.config.scale,
            }
        )
        return base


@dataclass
class DelayedRecallConfig:
    dimension: int = 33
    memory_dim: int = 32
    delay: int = 8
    steps: int = 16
    cue_noise: float = 0.02
    distractor_scale: float = 0.3
    reveal_sharpness: float = 0.35


class DelayedRecallBenchmark(UnifiedBenchmark):
    def __init__(self, config: DelayedRecallConfig) -> None:
        super().__init__()
        if config.memory_dim + 1 != config.dimension:
            config = DelayedRecallConfig(
                dimension=config.memory_dim + 1,
                memory_dim=config.memory_dim,
                delay=config.delay,
                steps=config.steps,
                cue_noise=config.cue_noise,
                distractor_scale=config.distractor_scale,
                reveal_sharpness=config.reveal_sharpness,
            )
        self.config = config
        self.dimension = config.dimension
        self.dt = 1.0
        self.default_steps = config.steps
        self.state_shape = (config.dimension,)
        self.stochastic = False
        self.is_non_markovian = True

    def _split(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = state[..., : self.config.memory_dim]
        clock = state[..., -1:]
        return mem, clock

    def sample_initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        device = _as_device(device)
        mem = torch.randn(batch_size, self.config.memory_dim, device=device)
        clock = torch.zeros(batch_size, 1, device=device)
        return torch.cat([mem, clock], dim=-1)

    def observe(self, hidden: torch.Tensor, t: int) -> torch.Tensor:
        mem, clock = self._split(hidden)
        if t == 0:
            visible = mem + self.config.cue_noise * torch.randn_like(mem)
        else:
            gate = torch.sigmoid(torch.tensor((t - self.config.delay) / max(1e-6, self.config.reveal_sharpness), device=hidden.device, dtype=hidden.dtype))
            visible = gate * mem + self.config.distractor_scale * torch.randn_like(mem)
        return torch.cat([visible, clock], dim=-1)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        mem, clock = self._split(state)
        clock_next = clock + (1.0 / max(1, self.config.steps))
        return torch.cat([mem, clock_next], dim=-1)

    def rollout(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = self.default_steps if steps is None else steps
        hidden = x0
        traj = []
        for t in range(steps + 1):
            traj.append(self.observe(hidden, t))
            hidden = self.step(hidden)
        return torch.stack(traj, dim=1)

    def generate_batch(self, batch_size: int, steps: Optional[int] = None, device: str | torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sample_initial_state(batch_size, device=device)
        traj = self.rollout(x0, steps=steps)
        return traj[:, :-1], traj[:, 1:]

    def metrics(self) -> Dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "name": "DelayedRecallBenchmark",
                "state_shape": self.state_shape,
                "horizon": self.default_steps,
                "is_non_markovian": True,
                "stochastic": False,
                "dt": 1.0,
                "dimension": self.dimension,
                "memory_dim": self.config.memory_dim,
                "delay": self.config.delay,
            }
        )
        return base


__all__ = [
    "BenchmarkMetrics",
    "UnifiedBenchmark",
    "Lorenz96Config",
    "Lorenz96Benchmark",
    "BurgersConfig",
    "BurgersBenchmark",
    "ReactionDiffusionConfig",
    "ReactionDiffusionBenchmark",
    "KuramotoConfig",
    "KuramotoBenchmark",
    "FractionalConfig",
    "FractionalDiffusionBenchmark",
    "LevyStableConfig",
    "LevyStableProcess",
    "DelayedRecallConfig",
    "DelayedRecallBenchmark",
    "fractional_weights",
]
