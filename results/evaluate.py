from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def _canonical_output(out: Any) -> torch.Tensor:
    if hasattr(out, 'prediction'):
        return out.prediction
    if isinstance(out, tuple):
        return out[0]
    return out


def _forward(model, x):
    if hasattr(model, 'step'):
        try:
            out = model.step(x, store_traces=False, retrieve=False)
        except TypeError:
            out = model.step(x)
    else:
        out = model(x)
    return _canonical_output(out)


def _reset_model_state(model):
    for name in ('reset_state', 'clear_memory', 'detach_state'):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


@torch.no_grad()
def rollout(model, x0, steps):
    _reset_model_state(model)
    preds = []
    x = x0
    for _ in range(steps):
        pred = _forward(model, x)
        preds.append(pred)
        x = pred.detach()
    return torch.stack(preds, dim=1)


def compute_mse(traj_pred, traj_true):
    return ((traj_pred - traj_true) ** 2).mean(dim=[2])


def evaluate(model, benchmark, device='cpu', steps=100, save_dir='results'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'trajectories').mkdir(parents=True, exist_ok=True)
    (save_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        x0 = benchmark.sample_initial_state(batch_size=8, device=device)
        traj_true = benchmark.rollout(x0, steps)
        traj_pred = rollout(model, x0, steps)

        mse = compute_mse(traj_pred, traj_true[:, 1:])

        torch.save(traj_true, save_dir / 'trajectories' / 'true.pt')
        torch.save(traj_pred, save_dir / 'trajectories' / 'pred.pt')

        metrics = {
            'mse_mean': mse.mean().item(),
            'mse_final': mse[:, -1].mean().item(),
        }

        with (save_dir / 'metrics' / 'metrics.json').open('w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    return metrics
