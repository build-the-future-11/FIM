from __future__ import annotations

from .write import LocalWrite
from .diffusion import DiffusionOperator
from .advection import AdvectionOperator
from .reaction import ReactionOperator
from .damping import DampingOperator
from .spectral import SpectralOperator


__all__ = [
    "LocalWrite",
    "DiffusionOperator",
    "AdvectionOperator",
    "ReactionOperator",
    "DampingOperator",
    "SpectralOperator",
    "OPERATOR_REGISTRY",
]


OPERATOR_REGISTRY = {
    "write": LocalWrite,
    "diffusion": DiffusionOperator,
    "advection": AdvectionOperator,
    "reaction": ReactionOperator,
    "damping": DampingOperator,
    "spectral": SpectralOperator,
}


def build_operator(name: str, *args, **kwargs):
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator: {name}")
    return OPERATOR_REGISTRY[name](*args, **kwargs)