from __future__ import annotations

from .encoder import FabricEncoder
from .decoder import FabricDecoder
from .fim_model import FIMModel, FIMStepOutput, LatentOperator


__all__ = [
    "FabricEncoder",
    "FabricDecoder",
    "FIMModel",
    "FIMStepOutput",
    "LatentOperator",
    "MODEL_REGISTRY",
]


MODEL_REGISTRY = {
    "fim": FIMModel,
}


def build_model(name: str, *args, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](*args, **kwargs)