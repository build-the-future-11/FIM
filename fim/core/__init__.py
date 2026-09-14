from __future__ import annotations

"""Core fabric dynamics."""

from .state import FabricState
from .fabric import Fabric
from .dynamics import FabricDynamics
from .fractional import FractionalMemory
from .integrator import ExplicitEulerIntegrator, RungeKutta2Integrator
from .stability import (
    fabric_energy,
    is_bounded,
    clamp_norm_,
    estimate_growth_rate,
)


__all__ = [
    "FabricState",
    "Fabric",
    "FabricDynamics",
    "FractionalMemory",
    "ExplicitEulerIntegrator",
    "RungeKutta2Integrator",
    "fabric_energy",
    "is_bounded",
    "clamp_norm_",
    "estimate_growth_rate",
    "INTEGRATOR_REGISTRY",
]


INTEGRATOR_REGISTRY = {
    "euler": ExplicitEulerIntegrator,
    "rk2": RungeKutta2Integrator,
}


def build_integrator(name: str, *args, **kwargs):
    if name not in INTEGRATOR_REGISTRY:
        raise ValueError(f"Unknown integrator: {name}")
    return INTEGRATOR_REGISTRY[name](*args, **kwargs)