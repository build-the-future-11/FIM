from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fim_experiments.main import (  # noqa: E402
    build_system,
    get_device,
    load_yaml_config,
    select_benchmark,
    set_seed,
)
from fim_experiments.results.run_results import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing FIM experiment checkpoint without retraining.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint mapping, got {type(payload)!r}")
    return payload


def _resolve_config(args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    if args.config is not None:
        return load_yaml_config(args.config)

    embedded = payload.get("config")
    if isinstance(embedded, dict):
        return embedded

    candidate = args.checkpoint.parent.parent / "configs" / "resolved_config.yaml"
    if candidate.exists():
        return load_yaml_config(candidate)

    raise FileNotFoundError(
        "No experiment config was supplied or embedded. Pass --config explicitly, "
        f"or place resolved_config.yaml at {candidate}."
    )


def _state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError("Checkpoint contains neither 'model_state_dict' nor trainer 'model' state.")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    payload = _load_checkpoint(args.checkpoint, device)
    config = _resolve_config(args, payload)

    runtime = config.get("runtime", {}) or {}
    seed = int(runtime.get("seed", config.get("seed", 42)))
    set_seed(seed, deterministic=bool(runtime.get("deterministic", True)))

    benchmark = select_benchmark(config.get("benchmark", {}) or {})
    model = build_system(config, benchmark, device)
    missing, unexpected = model.load_state_dict(_state_dict(payload), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )

    if args.output_dir is None:
        output_dir = args.checkpoint.parent.parent / "evaluation"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("eval", {}) or {}
    benchmark_cfg = config.get("benchmark", {}) or {}
    rollout_steps = (
        int(args.rollout_steps)
        if args.rollout_steps is not None
        else int(eval_cfg.get("rollout_steps", benchmark_cfg.get("config", {}).get("steps", 50)))
    )

    metrics = run(
        model,
        benchmark,
        device=device,
        rollout_steps=rollout_steps,
        batch_size=args.batch_size,
        plot_path=output_dir / "rollout_comparison.pdf",
        save_results_path=output_dir / "metrics.json",
        predictions_path=output_dir / "predictions.pt",
        per_example_path=output_dir / "per_example.jsonl",
    )

    provenance = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config is not None else "embedded-or-sibling-resolved-config",
        "device": str(device),
        "seed": seed,
        "rollout_steps": rollout_steps,
        "batch_size": int(args.batch_size),
    }
    with (output_dir / "evaluation_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print(json.dumps({"metrics": metrics, "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
