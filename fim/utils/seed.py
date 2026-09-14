from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_runtime(num_threads: int | None = None, num_interop_threads: int | None = None) -> None:
    threads = int(os.environ.get('FIM_NUM_THREADS', num_threads if num_threads is not None else 1))
    interop = int(os.environ.get('FIM_INTEROP_THREADS', num_interop_threads if num_interop_threads is not None else 1))
    try:
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, interop))
    except Exception:
        pass


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    configure_runtime()

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass
