from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import math
import os
import torch

from results.evaluate import evaluate

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass
class RolloutMetrics:
    model_name: str
    rollout_steps: int
    params: int
    mse: float
    mae: float
    final_step_mse: float
    final_step_mae: float
    divergence_time: float
    stability_under_perturbation: float
    latent_norm_drift: float
    retrieval_utilization: float
    throughput_items_per_sec: float
    trajectory_fidelity: float
    rollout_score: float


def _params_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def _canonical_output(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    prediction = None
    latent = None
    retrieved = None

    if hasattr(out, "prediction"):
        prediction = out.prediction
        latent = getattr(out, "latent", None)
        retrieved = getattr(out, "retrieved", None)
    elif isinstance(out, tuple):
        prediction = out[0]
        if len(out) > 1:
            aux = out[1]
            if torch.is_tensor(aux):
                latent = aux
            elif hasattr(aux, "latent"):
                latent = getattr(aux, "latent", None)
                retrieved = getattr(aux, "retrieved", None)
    else:
        prediction = out

    if prediction is None:
        raise TypeError("Model output did not contain a prediction tensor")

    return prediction, latent, retrieved


def _model_forward(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if hasattr(model, "step"):
        try:
            out = model.step(x, store_traces=False, retrieve=False)
        except TypeError:
            out = model.step(x)
    else:
        out = model(x)
    return _canonical_output(out)


def _reset_model_state(model: torch.nn.Module) -> None:
    for name in ("reset_state", "clear_memory", "detach_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _prepare_rollout_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def _as_rollout_target(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    return x


def _flatten_traj(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim == 2:
        return x.unsqueeze(0).reshape(x.shape[0], 1, -1)
    return x.reshape(1, 1, -1)


def _step_error(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    return mse, mae


def _trajectory_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred_f = _flatten_traj(pred)
    gt_f = _flatten_traj(gt)
    t = min(pred_f.shape[1], gt_f.shape[1])
    pred_f = pred_f[:, :t]
    gt_f = gt_f[:, :t]

    mse_per_step = torch.mean((pred_f - gt_f) ** 2, dim=-1)
    mae_per_step = torch.mean(torch.abs(pred_f - gt_f), dim=-1)

    mse = mse_per_step.mean().item()
    mae = mae_per_step.mean().item()
    final_step_mse = mse_per_step[:, -1].mean().item()
    final_step_mae = mae_per_step[:, -1].mean().item()

    gt_centered = gt_f - gt_f.mean(dim=1, keepdim=True)
    pred_centered = pred_f - pred_f.mean(dim=1, keepdim=True)
    numerator = torch.sum(gt_centered * pred_centered, dim=-1)
    denom = torch.sqrt(torch.sum(gt_centered**2, dim=-1).clamp_min(1e-12) * torch.sum(pred_centered**2, dim=-1).clamp_min(1e-12))
    corr = (numerator / denom).clamp(-1.0, 1.0)
    trajectory_fidelity = corr.mean().item()

    return {
        "mse": mse,
        "mae": mae,
        "final_step_mse": final_step_mse,
        "final_step_mae": final_step_mae,
        "trajectory_fidelity": trajectory_fidelity,
    }


def _divergence_time(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.15) -> float:
    pred_f = _flatten_traj(pred)
    gt_f = _flatten_traj(gt)
    t = min(pred_f.shape[1], gt_f.shape[1])
    pred_f = pred_f[:, :t]
    gt_f = gt_f[:, :t]

    err = torch.mean(torch.abs(pred_f - gt_f), dim=-1)
    for i in range(t):
        if torch.any(err[:, i] > threshold):
            return float(i)
    return float(t)


def _latent_norm_drift(latents: List[torch.Tensor]) -> float:
    if len(latents) < 2:
        return 0.0
    norms = []
    for z in latents:
        if z is None:
            continue
        z = z.float()
        norms.append(torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean().item())
    if len(norms) < 2:
        return 0.0
    return float(abs(norms[-1] - norms[0]))


def _retrieval_utilization(outputs: List[Any]) -> float:
    if not outputs:
        return 0.0
    used = 0.0
    total = 0.0
    for out in outputs:
        if out is None:
            continue
        retrieved = getattr(out, "retrieved", None)
        if retrieved is not None:
            total += 1.0
            if torch.is_tensor(retrieved) and retrieved.numel() > 0:
                used += 1.0
        else:
            total += 1.0
    if total == 0:
        return 0.0
    return used / total


@torch.no_grad()
def _autoregressive_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Any], float]:
    model.eval()
    _reset_model_state(model)
    current = _prepare_rollout_input(x0)
    preds = []
    latents: List[Optional[torch.Tensor]] = []
    outputs: List[Any] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()

    for _ in range(steps):
        pred, latent, retrieved = _model_forward(model, current)
        outputs.append(type("OutputView", (), {"retrieved": retrieved, "latent": latent})())
        preds.append(pred)
        latents.append(latent)
        current = _prepare_rollout_input(pred)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start

    pred_traj = torch.stack(preds, dim=1)
    return pred_traj, latents, outputs, elapsed


@torch.no_grad()
def _perturbation_stability(
    model: torch.nn.Module,
    x0: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    device: torch.device,
    noise_scale: float = 0.01,
    trials: int = 5,
) -> float:
    scores = []
    base = x0.float()
    for _ in range(trials):
        noise = noise_scale * torch.randn_like(base)
        x_perturbed = base + noise
        pred, _, _, _ = _autoregressive_rollout(model, x_perturbed, steps, device)
        metrics = _trajectory_metrics(pred, gt)
        scores.append(1.0 / (1.0 + metrics["mse"]))
    return float(sum(scores) / max(len(scores), 1))


def _plot_rollout(gt: torch.Tensor, pred: torch.Tensor, title: str, save_path: Optional[str | Path] = None) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    gt_ = gt.detach().cpu()
    pred_ = pred.detach().cpu()
    gt_f = _flatten_traj(gt_)[0]
    pred_f = _flatten_traj(pred_)[0]

    t = min(gt_f.shape[0], pred_f.shape[0])
    gt_f = gt_f[:t]
    pred_f = pred_f[:t]

    plt.figure(figsize=(10, 4))
    plt.plot(gt_f.numpy(), label="ground truth")
    plt.plot(pred_f.numpy(), label="prediction")
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()

    plt.close()


def _run_baseline(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    steps: int,
) -> Dict[str, float]:
    _reset_model_state(model)
    x0 = benchmark.sample_initial_state(1, device=device)
    gt = benchmark.rollout(x0, steps=steps)
    pred, latents, outputs, elapsed = _autoregressive_rollout(model, x0, steps, device)
    traj_stats = _trajectory_metrics(pred, gt)
    return {
        "mse": traj_stats["mse"],
        "mae": traj_stats["mae"],
        "final_step_mse": traj_stats["final_step_mse"],
        "final_step_mae": traj_stats["final_step_mae"],
        "divergence_time": _divergence_time(pred, gt),
        "stability_under_perturbation": _perturbation_stability(model, x0, gt, steps, device),
        "latent_norm_drift": _latent_norm_drift([z for z in latents if z is not None]),
        "retrieval_utilization": _retrieval_utilization(outputs),
        "throughput_items_per_sec": float(steps / max(elapsed, 1e-8)),
        "trajectory_fidelity": traj_stats["trajectory_fidelity"],
        "rollout_score": 1.0 / (1.0 + traj_stats["mse"]),
    }


def _make_baseline_registry(
    baselines: Optional[Mapping[str, torch.nn.Module]],
) -> Dict[str, torch.nn.Module]:
    if baselines is None:
        return {}
    return dict(baselines)


def run(
    model: torch.nn.Module,
    benchmark: Any,
    device: str | torch.device = "cpu",
    rollout_steps: int = 100,
    perturbation_scale: float = 0.01,
    perturbation_trials: int = 5,
    divergence_threshold: float = 0.15,
    plot_path: Optional[str | Path] = None,
    baselines: Optional[Mapping[str, torch.nn.Module]] = None,
    save_results_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    _reset_model_state(model)

    x0 = benchmark.sample_initial_state(1, device=device)
    gt = benchmark.rollout(x0, steps=rollout_steps)

    pred, latents, outputs, elapsed = _autoregressive_rollout(model, x0, rollout_steps, device)
    traj_stats = _trajectory_metrics(pred, gt)

    metrics = RolloutMetrics(
        model_name=type(model).__name__,
        rollout_steps=rollout_steps,
        params=_params_count(model),
        mse=traj_stats["mse"],
        mae=traj_stats["mae"],
        final_step_mse=traj_stats["final_step_mse"],
        final_step_mae=traj_stats["final_step_mae"],
        divergence_time=_divergence_time(pred, gt, threshold=divergence_threshold),
        stability_under_perturbation=_perturbation_stability(
            model=model,
            x0=x0,
            gt=gt,
            steps=rollout_steps,
            device=device,
            noise_scale=perturbation_scale,
            trials=perturbation_trials,
        ),
        latent_norm_drift=_latent_norm_drift([z for z in latents if z is not None]),
        retrieval_utilization=_retrieval_utilization(outputs),
        throughput_items_per_sec=float(rollout_steps / max(elapsed, 1e-8)),
        trajectory_fidelity=traj_stats["trajectory_fidelity"],
        rollout_score=1.0 / (1.0 + traj_stats["mse"]),
    )

    _plot_rollout(gt, pred, title="Rollout Comparison", save_path=plot_path)

    evaluation = evaluate(model, benchmark, device=device)
    result: Dict[str, Any] = {
        "model": type(model).__name__,
        "metrics": asdict(metrics),
        "evaluation": evaluation,
    }

    baseline_registry = _make_baseline_registry(baselines)
    if baseline_registry:
        result["baselines"] = {}
        for name, baseline_model in baseline_registry.items():
            baseline_model = baseline_model.to(device)
            result["baselines"][name] = _run_baseline(
                baseline_model,
                benchmark=benchmark,
                device=device,
                steps=rollout_steps,
            )

    if save_results_path is not None:
        save_results_path = Path(save_results_path)
        save_results_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json
            with save_results_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    return result