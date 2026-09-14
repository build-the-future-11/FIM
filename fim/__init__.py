from __future__ import annotations

import os
import torch

from .core.state import FabricState
from .models.fim_model import FIMModel
from .utils.seed import configure_runtime, set_seed

if os.environ.get('FIM_SKIP_RUNTIME_CONFIG', '0') != '1':
    configure_runtime()

__all__ = [
    'FabricState',
    'FIMModel',
    'set_seed',
    'configure_runtime',
    'build_model',
    'get_device',
    'get_dtype',
]


def build_model(config):
    return FIMModel(config)


def get_device(config):
    device = config.get('project', {}).get('device', 'cpu')

    if device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    if device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    return torch.device(device)


def get_dtype(config):
    dtype_str = config.get('project', {}).get('dtype', 'float32')

    if dtype_str == 'float32':
        return torch.float32
    if dtype_str == 'float64':
        return torch.float64
    if dtype_str == 'float16':
        return torch.float16
    if dtype_str == 'bfloat16':
        return torch.bfloat16

    raise ValueError(f'Unsupported dtype: {dtype_str}')
