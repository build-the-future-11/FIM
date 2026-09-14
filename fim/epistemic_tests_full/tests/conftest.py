from __future__ import annotations

import importlib
from typing import Iterable, Sequence

import pytest
import torch


def import_module_candidates(candidates: Sequence[str]):
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ImportError("No candidates provided")


def import_attr_candidates(module_candidates: Sequence[str], attr: str):
    last_exc = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ImportError(f"Could not import {attr}")


def maybe_import_attr(module_candidates: Sequence[str], attr: str):
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception:
            continue
    return None


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)
    try:
        import random

        random.seed(7)
    except Exception:
        pass


@pytest.fixture

def import_helpers():
    return {
        "import_module_candidates": import_module_candidates,
        "import_attr_candidates": import_attr_candidates,
        "maybe_import_attr": maybe_import_attr,
    }
