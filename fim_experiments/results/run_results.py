from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

# This module only writes plot artifacts.  Selecting a non-interactive backend
# before importing pyplot keeps the evaluator usable in CI/headless processes
# (the macOS GUI backend otherwise aborts the interpreter).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    from fim_experiments.provenance import sha256_file, tensor_sha256
except ImportError:  # direct execution with fim_experiments on sys.path
    from provenance import sha256_file, tensor_sha256


def _canonical_prediction(output: Any) -> torch.Tensor:
    """Return the prediction tensor from supported model output conventions."""
    if hasattr(output, "prediction"):
        return output.prediction
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Model returned an empty tuple.")
        return output[0]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _canonical_salience(output: Any) -> Optional[torch.Tensor]:
    if hasattr(output, "salience"):
        value = output.salience
        return value if torch.is_tensor(value) else None
    if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
        return output[1]
    return None


def _coerce_like(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Map field-shaped model output back to the benchmark state shape safely."""
    if prediction.shape == reference.shape:
        return prediction
    if prediction.numel() == reference.numel():
        return prediction.reshape_as(reference)
    raise ValueError(
        "Prediction cannot be mapped to benchmark state shape: "
        f"prediction={tuple(prediction.shape)} reference={tuple(reference.shape)}"
    )


def _write_trajectory_plot(path: Path, predicted: torch.Tensor, truth: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_series = predicted[0].detach().cpu().reshape(predicted.shape[1], -1)[:, 0]
    true_series = truth[0].detach().cpu().reshape(truth.shape[1], -1)[:, 0]

    plt.figure()
    plt.plot(pred_series.numpy(), label="prediction")
    plt.plot(true_series.numpy(), label="truth")
    plt.xlabel("rollout step")
    plt.ylabel("state[0]")
    plt.title("Autoregressive rollout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run(
    model,
    benchmark,
    device="cpu",
    rollout_steps: int = 50,
    perturbation_scale: float = 0.01,
    perturbation_trials: int = 5,
    divergence_threshold: float = 0.15,
    plot_path: str | Path | None = None,
    save_results_path: str | Path | None = None,
    batch_size: int = 32,
    predictions_path: str | Path | None = None,
    per_example_path: str | Path | None = None,
):
    """Evaluate a model with a benchmark-native autoregressive rollout.

    The perturbation arguments are accepted for API compatibility with the main
    experiment runner, but this function deliberately reports only metrics it
    actually computes. Stability/perturbation claims must come from a dedicated
    implemented analysis rather than placeholder fields.
    """
    del perturbation_scale, perturbation_trials, divergence_threshold

    device = torch.device(device)
    steps = int(rollout_steps)
    if steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()
    elif hasattr(model, "clear_memory"):
        model.clear_memory()

    x0 = benchmark.sample_initial_state(batch_size=int(batch_size), device=device)
    truth = benchmark.rollout(x0, steps=steps)

    predictions = [x0]
    salience_seen = False
    x = x0

    with torch.no_grad():
        for _ in range(steps):
            output = model(x)
            prediction = _coerce_like(_canonical_prediction(output), x)
            salience_seen = salience_seen or _canonical_salience(output) is not None
            predictions.append(prediction)
            x = prediction.detach()

    predicted = torch.stack(predictions, dim=1)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"Rollout shape mismatch: prediction={tuple(predicted.shape)} "
            f"truth={tuple(truth.shape)}"
        )

    # Exclude the seeded initial condition from forecast error. Including t=0
    # would add an artificial zero-error point and bias the reported rollout
    # metric downward, especially for short horizons.
    forecast_error = predicted[:, 1:] - truth[:, 1:]
    mse_by_step = forecast_error.reshape(
        forecast_error.shape[0], forecast_error.shape[1], -1
    ).pow(2).mean(dim=-1)
    mae_by_step = forecast_error.reshape(
        forecast_error.shape[0], forecast_error.shape[1], -1
    ).abs().mean(dim=-1)

    metrics = {
        "rollout_mse": float(mse_by_step.mean().item()),
        "final_step_mse": float(mse_by_step[:, -1].mean().item()),
        "rollout_mae": float(mae_by_step.mean().item()),
        "final_step_mae": float(mae_by_step[:, -1].mean().item()),
        "rollout_steps": steps,
        "evaluation_batch_size": int(batch_size),
        "salience_available": bool(salience_seen),
        "evaluation_data_sha256": tensor_sha256((x0, truth)),
    }

    per_example = []
    for sample_id in range(predicted.shape[0]):
        sample_mse = mse_by_step[sample_id]
        sample_mae = mae_by_step[sample_id]
        per_example.append(
            {
                "sample_id": sample_id,
                "rollout_mse": float(sample_mse.mean().item()),
                "final_step_mse": float(sample_mse[-1].item()),
                "rollout_mae": float(sample_mae.mean().item()),
                "final_step_mae": float(sample_mae[-1].item()),
            }
        )

    if plot_path is not None:
        _write_trajectory_plot(Path(plot_path), predicted, truth)

    if save_results_path is not None:
        output_path = Path(save_results_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    if predictions_path is not None:
        output_path = Path(predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "initial_state": x0.detach().cpu(),
                "prediction": predicted.detach().cpu(),
                "truth": truth.detach().cpu(),
            },
            output_path,
        )
        metrics["predictions_sha256"] = sha256_file(output_path)

    if per_example_path is not None:
        output_path = Path(per_example_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in per_example:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        metrics["per_example_sha256"] = sha256_file(output_path)

    # Re-write after artifact hashes are available.
    if save_results_path is not None:
        with Path(save_results_path).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    return metrics
