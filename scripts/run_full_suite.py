from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
from typing import Any, Dict, List

from fim_experiments.main import resolve_config, run_experiment


DEFAULT_BENCHMARKS = ["lorenz", "burgers", "fractional", "reaction_diffusion", "delayed_recall"]
DEFAULT_MODELS = ["fim_plus", "transformer", "ssm", "deeponet", "mlp"]


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_args(benchmark: str, model: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(argparse.Namespace(
        config=None,
        set=[],
        benchmark=benchmark,
        model=model,
        exp_name=args.exp_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        workers=args.workers,
        dynamic_data=args.dynamic_data,
        hidden=args.hidden,
        trace_dim=args.trace_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        eval_steps=args.eval_steps,
        rollout_steps=args.rollout_steps,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        horizon_decay=args.horizon_decay,
        results_root=str(args.results_root),
        resume_from=None,
        no_eval=args.no_eval,
        compile=args.compile,
    ))
    cfg["experiment"]["name"] = args.exp_name
    cfg["benchmark"]["name"] = benchmark
    cfg["model"]["name"] = model
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--results_root", type=Path, default=Path("results") / "compact_suite")
    parser.add_argument("--exp_name", type=str, default="compact_suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dynamic_data", action="store_true")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--trace_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--horizon_decay", type=float, default=1.0)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = _parse_csv(args.benchmarks)
    models = _parse_csv(args.models)
    args.results_root = Path(args.results_root)
    args.results_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "benchmarks": benchmarks,
        "models": models,
        "runs": [],
    }

    for benchmark in benchmarks:
        for model in models:
            cfg = _build_args(benchmark, model, args)
            result = run_experiment(cfg)
            summary["runs"].append({
                "benchmark": benchmark,
                "model": model,
                "result": result,
            })
            run_dir = args.results_root / args.exp_name / f"{benchmark}_{model}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    with (args.results_root / args.exp_name / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2)[:12000])


if __name__ == "__main__":
    main()
