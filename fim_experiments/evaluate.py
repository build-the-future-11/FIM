from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import torch


def _canonical_output(out: Any):
    if hasattr(out, 'prediction'):
        return out.prediction
    if isinstance(out, tuple):
        return out[0]
    return out


def rollout(model, x0, steps):
    preds = []
    x = x0
    for _ in range(steps):
        out = model(x)
        pred = _canonical_output(out)
        preds.append(pred)
        x = pred.detach()
    return torch.stack(preds, dim=1)


def compute_mse(traj_pred, traj_true):
    return ((traj_pred - traj_true) ** 2).mean(dim=[2])


def evaluate(model, benchmark, device='cpu', steps=100, save_dir='results'):
    import json
    import os

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/trajectories', exist_ok=True)
    os.makedirs(f'{save_dir}/metrics', exist_ok=True)

    model.eval()

    with torch.no_grad():
        x0 = benchmark.sample_initial_state(batch_size=8, device=device)
        traj_true = benchmark.rollout(x0, steps)
        traj_pred = rollout(model, x0, steps)

        mse = compute_mse(traj_pred, traj_true[:, 1:])

        torch.save(traj_true, f'{save_dir}/trajectories/true.pt')
        torch.save(traj_pred, f'{save_dir}/trajectories/pred.pt')

        metrics = {
            'mse_mean': mse.mean().item(),
            'mse_final': mse[:, -1].mean().item(),
        }

        with open(f'{save_dir}/metrics/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    return metrics
