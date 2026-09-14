from __future__ import annotations

from .brownian import BrownianMotion, BrownianConfig
from .levy import LevyStableProcess, LevyStableConfig
from .noise import GaussianNoise, ColoredNoise, PoissonNoise, HeteroscedasticNoise
from .sde import SDEIntegrator, SDEConfig


__all__ = [
    "BrownianMotion",
    "BrownianConfig",
    "LevyStableProcess",
    "LevyStableConfig",
    "GaussianNoise",
    "ColoredNoise",
    "PoissonNoise",
    "HeteroscedasticNoise",
    "SDEIntegrator",
    "SDEConfig",
]


REGISTRY = {
    "brownian": BrownianMotion,
    "levy": LevyStableProcess,
    "gaussian_noise": GaussianNoise,
    "colored_noise": ColoredNoise,
    "poisson_noise": PoissonNoise,
    "heteroscedastic_noise": HeteroscedasticNoise,
    "sde": SDEIntegrator,
}


CONFIG_REGISTRY = {
    "brownian": BrownianConfig,
    "levy": LevyStableConfig,
    "sde": SDEConfig,
}