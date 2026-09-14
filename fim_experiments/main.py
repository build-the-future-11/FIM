from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

if ROOT not in sys.path:
    sys.path.append(ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from benchmark import (
    BurgersBenchmark,
    BurgersConfig,
    DelayedRecallBenchmark,
    DelayedRecallConfig,
    FractionalConfig,
    FractionalDiffusionBenchmark,
    KuramotoBenchmark,
    KuramotoConfig,
    LevyStableConfig,
    LevyStableProcess,
    Lorenz96Benchmark,
    Lorenz96Config,
    ReactionDiffusionBenchmark,
    ReactionDiffusionConfig,
)
from results.run_results import run
from systems import (
    DeepONetFieldSystem,
    FIMSystem,
    FieldMLPSystem,
    SelectiveSSMSystem,
    SpectralFieldSystem,
    TransformerFieldSystem,
)
from train import train
try:
    from fim_experiments.provenance import runtime_metadata, tensor_sha256, utc_now
except ImportError:
    from provenance import runtime_metadata, tensor_sha256, utc_now


def get_device(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _loader_kwargs(device: torch.device, workers: int) -> Dict[str, Any]:
    use_accel = device.type in {"cuda", "mps"}
    return {
        "num_workers": workers if use_accel else 0,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(workers > 0 and use_accel),
    }


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config support.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    return data


def save_yaml_config(path: str | Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for config export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def setup_logging(exp_dir: Path, level: int = logging.INFO) -> logging.Logger:
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fim_experiment")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(exp_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _call_with_supported_kwargs(fn: Callable[..., Any], *args, **kwargs):
    sig = inspect.signature(fn)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kwargs:
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)


def _shape_product(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _state_shape_from_benchmark(benchmark: Any) -> Tuple[int, ...]:
    if hasattr(benchmark, "state_shape"):
        shape = tuple(int(x) for x in getattr(benchmark, "state_shape"))
        if len(shape) > 0:
            return shape
    if hasattr(benchmark, "config"):
        cfg = getattr(benchmark, "config")
        if hasattr(cfg, "dimension"):
            return (int(cfg.dimension),)
    raise ValueError("Could not infer benchmark state shape.")


def _benchmark_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def select_benchmark(cfg: Dict[str, Any]) -> Any:
    name = _benchmark_name(str(cfg["name"]))
    params = dict(cfg.get("config", {}) or {})

    if name in {"lorenz", "lorenz96", "lorenz_96"}:
        return Lorenz96Benchmark(Lorenz96Config(**params))
    if name in {"fractional", "fractional_diffusion"}:
        return FractionalDiffusionBenchmark(FractionalConfig(**params))
    if name in {"burgers", "burgers1d", "burgers_1d"}:
        return BurgersBenchmark(BurgersConfig(**params))
    if name in {"reaction", "reaction_diffusion", "rd"}:
        return ReactionDiffusionBenchmark(ReactionDiffusionConfig(**params))
    if name in {"kuramoto", "oscillators"}:
        return KuramotoBenchmark(KuramotoConfig(**params))
    if name in {"levy", "levy_stable", "levy_process"}:
        return LevyStableProcess(LevyStableConfig(**params))
    if name in {"delayed_recall", "delayed", "memory_delay", "copy_delay"}:
        return DelayedRecallBenchmark(DelayedRecallConfig(**params))
    raise ValueError(f"Unknown benchmark: {name}")


class ShapeAwareMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_shape: tuple[int, ...]):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        output_dim = _shape_product(output_shape)

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = self.net(x)
        return out.view(b, *self.output_shape)


def build_system(cfg: Dict[str, Any], benchmark: Any, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg.get("model", {}) or {}
    model_name = _benchmark_name(str(model_cfg.get("name", "fim")))
    hidden = int(model_cfg.get("hidden", 64))
    trace_dim = int(model_cfg.get("trace_dim", 32))
    compile_model = bool(model_cfg.get("compile", False))

    state_shape = _state_shape_from_benchmark(benchmark)

    in_channels = state_shape[0] if len(state_shape) > 1 else 1

    if model_name == "mlp":
        output_shape = (1, 1, state_shape[0]) if len(state_shape) == 1 else state_shape
        input_dim = _shape_product(state_shape)
        model = ShapeAwareMLP(input_dim=input_dim, hidden_dim=hidden, output_shape=output_shape)
    elif model_name in {"transformer", "ssm", "spectral", "fno", "deeponet"}:
        if model_name == "transformer":
            model = TransformerFieldSystem(in_channels=in_channels, hidden=hidden, heads=int(model_cfg.get("heads", 4)), depth=int(model_cfg.get("layers", 3)), ff_mult=int(model_cfg.get("ff_mult", 4)), max_tokens=int(model_cfg.get("max_tokens", 4096)))
        elif model_name == "ssm":
            model = SelectiveSSMSystem(in_channels=in_channels, hidden=hidden, depth=int(model_cfg.get("layers", 4)))
        elif model_name in {"spectral", "fno"}:
            model = SpectralFieldSystem(in_channels=in_channels, hidden=hidden, modes=int(model_cfg.get("modes", 8)), depth=int(model_cfg.get("layers", 4)))
        else:
            model = DeepONetFieldSystem(in_channels=in_channels, hidden=hidden, basis_dim=int(model_cfg.get("basis_dim", max(32, hidden))))
    else:
        model = FIMSystem(
            in_channels=in_channels,
            hidden=hidden,
            trace_dim=trace_dim,
            memory_capacity=int(model_cfg.get("memory_capacity", 2048)),
            retrieval_topk=int(model_cfg.get("retrieval_topk", 32)),
            memory_decay=float(model_cfg.get("memory_decay", 0.02)),
            salience_threshold=float(model_cfg.get("salience_threshold", 0.45)),
        )

    model = model.to(device)

    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model


def reshape_if_needed(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 2:
        x = x.unsqueeze(1).unsqueeze(1)
        y = y.unsqueeze(1).unsqueeze(1)
    return x.contiguous(), y.contiguous()


def build_loader_static(
    benchmark: Any,
    batch_size: int,
    device: torch.device,
    dataset_size: int,
    workers: int,
    steps: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    x, y = benchmark.generate_batch(batch_size=dataset_size, steps=steps, device="cpu")
    x, y = reshape_if_needed(x, y)
    dataset = TensorDataset(x, y)

    if len(dataset) < 2:
        raise ValueError("Dataset size is too small to create train/validation splits.")

    split = max(1, int(0.85 * len(dataset)))
    split = min(split, len(dataset) - 1)
    split_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    train_set, val_set = random_split(
        dataset, [split, len(dataset) - split], generator=split_generator
    )

    kw = _loader_kwargs(device, workers)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed) + 1),
        **kw,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )
    return train_loader, val_loader


def rollout_eval(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    steps: int = 50,
    batch_size: int = 32,
) -> Dict[str, float]:
    x0 = benchmark.sample_initial_state(batch_size, device=device)

    model.eval()
    with torch.no_grad():
        gt = benchmark.rollout(x0, steps=steps)

        preds = [x0]
        x = x0

        for _ in range(steps):
            if x.ndim == 2:
                x_in = x.unsqueeze(1).unsqueeze(1)
            else:
                x_in = x

            y = model(x_in)
            if hasattr(y, "prediction"):
                x = y.prediction
            elif isinstance(y, tuple):
                x = y[0]
            else:
                x = y

            if x.ndim > 2 and gt.ndim > 2 and x.shape != gt[:, 0].shape:
                try:
                    x = x.view_as(gt[:, 0])
                except Exception:
                    x = x.squeeze()

            preds.append(x)

        pred = torch.stack(preds, dim=1)

    mse = ((gt - pred) ** 2).mean().item()
    mae = (gt - pred).abs().mean().item()

    return {
        "rollout_mse": mse,
        "rollout_mae": mae,
    }


def build_experiment_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    runtime = cfg.get("runtime", {}) or {}
    exp = cfg.get("experiment", {}) or {}
    exp_name = str(exp.get("name", "default"))
    benchmark_name = _benchmark_name(str(cfg.get("benchmark", {}).get("name", "benchmark")))
    model_name = _benchmark_name(str(cfg.get("model", {}).get("name", "fim")))

    results_root = Path(runtime.get("results_root", "results"))
    exp_dir = results_root / exp_name / f"{benchmark_name}_{model_name}"

    dirs = {
        "root": exp_dir,
        "checkpoints": exp_dir / "checkpoints",
        "metrics": exp_dir / "metrics",
        "logs": exp_dir / "logs",
        "configs": exp_dir / "configs",
        "figures": exp_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    started_utc = utc_now()
    started_monotonic = time.perf_counter()
    runtime = cfg.get("runtime", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    benchmark_cfg = cfg.get("benchmark", {}) or {}

    seed = int(runtime.get("seed", cfg.get("seed", 42)))
    data_seed = int(runtime.get("data_seed") if runtime.get("data_seed") is not None else seed + 10_000)
    model_seed = int(runtime.get("model_seed") if runtime.get("model_seed") is not None else seed + 20_000)
    train_seed = int(runtime.get("train_seed") if runtime.get("train_seed") is not None else seed + 30_000)
    eval_seed = int(runtime.get("eval_seed") if runtime.get("eval_seed") is not None else seed + 40_000)
    deterministic = bool(runtime.get("deterministic", True))
    device = get_device(str(runtime.get("device", "auto")))

    benchmark = select_benchmark(benchmark_cfg)
    dirs = build_experiment_dirs(cfg)
    logger = setup_logging(dirs["logs"])

    logger.info("Device: %s", device)
    logger.info("Benchmark: %s", benchmark_cfg.get("name", "unknown"))
    logger.info("Model: %s", cfg.get("model", {}).get("name", "fim"))
    logger.info(
        "Seed streams: base=%d data=%d model=%d train=%d eval=%d",
        seed, data_seed, model_seed, train_seed, eval_seed,
    )

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    train_loader = None
    val_loader = None

    if not bool(train_cfg.get("dynamic_data", False)):
        set_seed(data_seed, deterministic=deterministic)
        train_loader, val_loader = build_loader_static(
            benchmark=benchmark,
            batch_size=int(train_cfg.get("batch_size", 128)),
            device=device,
            dataset_size=int(train_cfg.get("dataset_size", 4096)),
            workers=int(train_cfg.get("workers", 2)),
            steps=int(train_cfg.get("rollout_steps", benchmark_cfg.get("config", {}).get("steps", 50))),
            seed=data_seed,
        )

    set_seed(model_seed, deterministic=deterministic)
    model = build_system(cfg, benchmark, device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{params:,}")

    set_seed(train_seed, deterministic=deterministic)

    train_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        benchmark=benchmark if bool(train_cfg.get("dynamic_data", False)) else None,
        epochs=int(train_cfg.get("epochs", 40)),
        device=device,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        use_amp=bool(train_cfg.get("amp", False) and device.type == "cuda"),
        batch_size=int(train_cfg.get("batch_size", 128)),
        steps_per_epoch=int(train_cfg.get("steps_per_epoch", 100)),
        val_batches=int(train_cfg.get("val_batches", 20)),
        rollout_steps=int(train_cfg.get("rollout_steps", 4)),
        teacher_forcing_ratio=float(train_cfg.get("teacher_forcing_ratio", 0.5)),
        horizon_decay=float(train_cfg.get("horizon_decay", 1.0)),
        ema_decay=float(train_cfg.get("ema_decay", 0.999)),
        use_ema=bool(train_cfg.get("use_ema", True)),
        evaluate_ema=bool(train_cfg.get("evaluate_ema", True)),
        scheduler_per_batch=bool(train_cfg.get("scheduler_per_batch", True)),
        checkpoint_dir=dirs["checkpoints"],
        resume_from=train_cfg.get("resume_from"),
        save_best=bool(train_cfg.get("save_best", True)),
        save_last=bool(train_cfg.get("save_last", True)),
        checkpoint_name=str(train_cfg.get("checkpoint_name", "fim_checkpoint.pt")),
    )

    logger.info("Training start")
    history = _call_with_supported_kwargs(train, model, **train_kwargs)
    logger.info("Training complete")

    auto_eval = bool(eval_cfg.get("enabled", True))
    metrics: Dict[str, Any] = {
        "experiment": cfg.get("experiment", {}),
        "runtime": runtime,
        "benchmark": benchmark.metrics() if hasattr(benchmark, "metrics") else benchmark_cfg,
        "model": cfg.get("model", {}),
        "train_history": history,
        "parameters": params,
        "seed_streams": {
            "base": seed,
            "data": data_seed,
            "model": model_seed,
            "train": train_seed,
            "evaluation": eval_seed,
        },
    }

    if train_loader is not None:
        tensors = train_loader.dataset.dataset.tensors
        metrics["generated_dataset_sha256"] = tensor_sha256(tensors)
        metrics["split_indices_sha256"] = tensor_sha256(
            (
                torch.tensor(train_loader.dataset.indices, dtype=torch.int64),
                torch.tensor(val_loader.dataset.indices, dtype=torch.int64),
            )
        )

    if auto_eval:
        logger.info("Evaluation start")
        set_seed(eval_seed, deterministic=deterministic)
        eval_metrics = _call_with_supported_kwargs(
            run,
            model,
            benchmark,
            device=device,
            rollout_steps=int(eval_cfg.get("rollout_steps", benchmark_cfg.get("config", {}).get("steps", 50))),
            perturbation_scale=float(eval_cfg.get("perturbation_scale", 0.01)),
            perturbation_trials=int(eval_cfg.get("perturbation_trials", 5)),
            divergence_threshold=float(eval_cfg.get("divergence_threshold", 0.15)),
            plot_path=dirs["figures"] / "rollout_comparison.pdf",
            save_results_path=dirs["metrics"] / "run_results.json",
            predictions_path=dirs["metrics"] / "predictions.pt",
            per_example_path=dirs["metrics"] / "per_example.jsonl",
        )
        if isinstance(eval_metrics, dict):
            metrics.update(eval_metrics)
        else:
            metrics["run_results"] = eval_metrics
        logger.info("Evaluation complete")

    config_path = dirs["configs"] / "resolved_config.yaml"
    save_yaml_config(config_path, cfg)

    summary_path = dirs["metrics"] / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)

    ckpt_path = dirs["checkpoints"] / "final_state.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _to_jsonable(cfg),
            "history": _to_jsonable(history),
            "metrics": _to_jsonable(metrics),
        },
        ckpt_path,
    )

    metadata = runtime_metadata(device=device, config=cfg, started_utc=started_utc)
    metadata["runtime_seconds"] = time.perf_counter() - started_monotonic
    metadata["seed_streams"] = metrics["seed_streams"]
    metadata["dataset"] = {
        "kind": "generated_at_runtime",
        "benchmark": benchmark_cfg,
        "generated_dataset_sha256": metrics.get("generated_dataset_sha256"),
        "split_indices_sha256": metrics.get("split_indices_sha256"),
        "evaluation_data_sha256": metrics.get("evaluation_data_sha256"),
    }
    metadata["artifacts"] = {
        "config": str(config_path),
        "summary": str(summary_path),
        "checkpoint": str(ckpt_path),
        "predictions": str(dirs["metrics"] / "predictions.pt") if auto_eval else None,
        "per_example": str(dirs["metrics"] / "per_example.jsonl") if auto_eval else None,
    }
    metadata_path = dirs["root"] / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(metadata), handle, indent=2)

    logger.info("Saved checkpoint: %s", ckpt_path)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved config: %s", config_path)
    logger.info("Saved provenance: %s", metadata_path)

    return metrics


def default_config() -> Dict[str, Any]:
    return {
        "experiment": {
            "name": "default",
        },
        "runtime": {
            "seed": 42,
            "data_seed": None,
            "model_seed": None,
            "train_seed": None,
            "eval_seed": None,
            "device": "auto",
            "deterministic": True,
            "results_root": "results",
        },
        "benchmark": {
            "name": "lorenz96",
            "config": {},
        },
        "model": {
            "name": "fim_plus",
            "hidden": 64,
            "trace_dim": 32,
            "compile": False,
            "memory_capacity": 2048,
            "retrieval_topk": 32,
            "memory_decay": 0.02,
            "salience_threshold": 0.45,
            "layers": 3,
            "heads": 4,
            "ff_mult": 4,
            "modes": 8,
            "basis_dim": 64,
        },
        "train": {
            "epochs": 40,
            "batch_size": 128,
            "dataset_size": 4096,
            "workers": 2,
            "dynamic_data": False,
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "grad_clip": 1.0,
            "amp": False,
            "rollout_steps": 4,
            "teacher_forcing_ratio": 0.5,
            "horizon_decay": 1.0,
            "ema_decay": 0.999,
            "use_ema": True,
            "evaluate_ema": True,
            "scheduler_per_batch": True,
            "steps_per_epoch": 100,
            "val_batches": 20,
            "resume_from": None,
            "save_best": True,
            "save_last": True,
            "checkpoint_name": "fim_checkpoint.pt",
        },
        "eval": {
            "enabled": True,
            "rollout_steps": 50,
            "perturbation_scale": 0.01,
            "perturbation_trials": 5,
            "divergence_threshold": 0.15,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dynamic_data", action="store_true")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--trace_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=None)
    parser.add_argument("--horizon_decay", type=float, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = default_config()

    if args.config is not None:
        cfg = _deep_update(cfg, load_yaml_config(args.config))

    if args.benchmark is not None:
        cfg["benchmark"]["name"] = args.benchmark
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.exp_name is not None:
        cfg["experiment"]["name"] = args.exp_name
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.results_root is not None:
        cfg["runtime"]["results_root"] = args.results_root
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.dataset_size is not None:
        cfg["train"]["dataset_size"] = args.dataset_size
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    if args.dynamic_data:
        cfg["train"]["dynamic_data"] = True
    if args.hidden is not None:
        cfg["model"]["hidden"] = args.hidden
    if args.trace_dim is not None:
        cfg["model"]["trace_dim"] = args.trace_dim
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.grad_clip is not None:
        cfg["train"]["grad_clip"] = args.grad_clip
    if args.amp:
        cfg["train"]["amp"] = True
    if args.eval_steps is not None:
        cfg["eval"]["rollout_steps"] = args.eval_steps
    if args.rollout_steps is not None:
        cfg["train"]["rollout_steps"] = args.rollout_steps
    if args.teacher_forcing_ratio is not None:
        cfg["train"]["teacher_forcing_ratio"] = args.teacher_forcing_ratio
    if args.horizon_decay is not None:
        cfg["train"]["horizon_decay"] = args.horizon_decay
    if args.resume_from is not None:
        cfg["train"]["resume_from"] = args.resume_from
    if args.no_eval:
        cfg["eval"]["enabled"] = False
    if args.compile:
        cfg["model"]["compile"] = True

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use dotted.key=value.")
        key, value = item.split("=", 1)
        _set_nested(cfg, key, _parse_value(value))

    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    metrics = run_experiment(cfg)
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        try:
            print(f"{key}: {float(value):.6f}")
        except Exception:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
