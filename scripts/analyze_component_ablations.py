#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .validate_component_ablation_manifest import (
        EXPECTED_BENCHMARKS,
        EXPECTED_SEEDS,
        EXPECTED_VARIANTS,
        PROTOCOL_FIELDS,
        validate_manifest,
    )
except ImportError:  # direct script execution
    from validate_component_ablation_manifest import (
        EXPECTED_BENCHMARKS,
        EXPECTED_SEEDS,
        EXPECTED_VARIANTS,
        PROTOCOL_FIELDS,
        validate_manifest,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/current_component_ablations/manifest.json"
OUTPUT_DIR = ROOT / "results/current_component_ablations"


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite JSON number")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return float(value)


def exact_sign_pvalue(values: list[float]) -> float:
    if not values:
        raise ValueError("sign test requires nonempty paired differences")
    values = [_finite(value, "paired difference") for value in values]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected = {
        (benchmark, seed, variant)
        for benchmark in EXPECTED_BENCHMARKS
        for seed in EXPECTED_SEEDS
        for variant in EXPECTED_VARIANTS
    }
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("historical analysis requires exactly 24 retained cells")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("seed")) is not int:
            raise ValueError("each cell must have a JSON integer seed")
        cell = row.get("benchmark"), row["seed"], row.get("variant")
        if cell not in expected or cell in seen:
            raise ValueError(f"duplicate or unexpected historical cell: {cell}")
        _finite(row.get("rollout_mse"), f"{cell} rollout MSE", nonnegative=True)
        seen.add(cell)
    if seen != expected:
        raise ValueError("historical matrix is incomplete")
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_cell[(str(row["benchmark"]), str(row["variant"]))].append(row)
        by_seed[(str(row["benchmark"]), int(row["seed"]))][str(row["variant"])] = row

    summaries = []
    for (benchmark, variant), group in sorted(by_cell.items()):
        values = [float(row["rollout_mse"]) for row in group]
        summaries.append(
            {
                "benchmark": benchmark,
                "variant": variant,
                "n_seeds": len(values),
                "rollout_mse_mean": statistics.mean(values),
                "rollout_mse_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "rollout_mse_min": min(values),
                "rollout_mse_max": max(values),
            }
        )

    paired = []
    benchmarks = sorted({key[0] for key in by_seed})
    variants = sorted({variant for values in by_seed.values() for variant in values if variant != "full"})
    for benchmark in benchmarks:
        for variant in variants:
            seed_rows = []
            for (candidate_benchmark, seed), values in sorted(by_seed.items()):
                if candidate_benchmark != benchmark or "full" not in values or variant not in values:
                    continue
                delta = float(values[variant]["rollout_mse"]) - float(values["full"]["rollout_mse"])
                seed_rows.append((seed, delta))
            deltas = [delta for _, delta in seed_rows]
            paired.append(
                {
                    "benchmark": benchmark,
                    "comparison": f"{variant}_minus_full",
                    "variant": variant,
                    "n_pairs": len(deltas),
                    "mean_delta_rollout_mse": statistics.mean(deltas),
                    "std_delta_rollout_mse": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "min_delta_rollout_mse": min(deltas),
                    "max_delta_rollout_mse": max(deltas),
                    "seeds_variant_better": sum(delta < 0 for delta in deltas),
                    "seeds_full_better": sum(delta > 0 for delta in deltas),
                    "exact_sign_pvalue": exact_sign_pvalue(deltas),
                    "seed_deltas": [{"seed": seed, "delta": delta} for seed, delta in seed_rows],
                    "interpretation": "negative favors the reduced variant; positive favors full",
                }
            )
    return {"summaries": summaries, "paired": paired}


def load_retained_text_evidence(root: Path, manifest_path: Path | None = None) -> tuple[dict, dict]:
    """Bind historical manifest metrics to 96 retained text artifacts.

    Checkpoints are neither read nor evaluated here. Their 24 manifest hashes
    remain provenance declarations and are explicitly outside this text audit.
    """
    root = root.resolve()
    path = manifest_path or root / "results/current_component_ablations/manifest.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    errors = validate_manifest(data)
    if errors:
        raise ValueError("invalid historical manifest: " + "; ".join(errors))
    summarize_rows(data["runs"])
    expected_protocol = {
        "epochs": 12, "batch_size": 64, "dataset_size": 2048,
        "train_rollout_steps": 4, "eval_rollout_steps": 30,
    }
    if data.get("protocol") != expected_protocol or any(
        type(data["protocol"][field]) is not int for field in PROTOCOL_FIELDS
    ):
        raise ValueError("historical shared-minibatch protocol identity changed")
    digests = {}
    for row in data["runs"]:
        for field in PROTOCOL_FIELDS:
            if type(row[field]) is not int or row[field] != expected_protocol[field]:
                raise ValueError(f"historical protocol mismatch in {field}")
        documents = {}
        for field in ("summary_file", "run_results_file", "resolved_config_file", "log_file"):
            relative = Path(row[field])
            target = (root / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(root):
                raise ValueError("retained artifact path leaves repository")
            blob = target.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != row["artifact_sha256"][field]:
                raise ValueError(f"retained artifact hash mismatch: {relative}")
            if str(relative) in digests:
                raise ValueError("historical cells must not alias retained artifacts")
            digests[str(relative)] = digest
            if field in ("summary_file", "run_results_file"):
                documents[field] = json.loads(blob)
        summary = documents["summary_file"]
        if type(summary["runtime"]["seed"]) is not int or summary["runtime"]["seed"] != row["seed"]:
            raise ValueError("retained summary seed disagrees with manifest")
        if summary["experiment"]["name"] != row["experiment_name"]:
            raise ValueError("retained summary experiment disagrees with manifest")
        for document in documents.values():
            for metric in ("rollout_mse", "rollout_mae"):
                actual = _finite(document.get(metric), metric, nonnegative=True)
                reported = _finite(row.get(metric), metric, nonnegative=True)
                if actual != reported:
                    raise ValueError(f"manifest {metric} disagrees with retained cell reports")
            if type(document.get("rollout_steps")) is not int or document["rollout_steps"] != 30:
                raise ValueError("retained rollout horizon mismatch")
    return data, {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_text_artifacts": len(digests), "artifact_sha256": digests,
        "checkpoint_artifacts_read": 0, "checkpoint_replay": False,
        "source_identity_kind": data["identity_kind"],
        "declared_execution_source_identity": data["source_identity"],
        "declared_execution_git_commit": data["git_commit"],
        "source_boundary": "Manifest records a source-tree hash and UNKNOWN git commit; this audit does not reconstruct that executed source tree.",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    data, _ = load_retained_text_evidence(ROOT, DEFAULT_MANIFEST)
    errors = validate_manifest(data, ROOT)
    if errors:
        raise SystemExit("invalid component-ablation evidence:\n- " + "\n- ".join(errors))
    analysis = summarize_rows(data["runs"])
    payload = {
        "schema_version": 1,
        "source_identity": data["source_identity"],
        "manifest": str(DEFAULT_MANIFEST.relative_to(ROOT)),
        "inference_boundary": (
            "Three paired training seeds per benchmark. Seed variation is reported descriptively; "
            "it is not independent-domain evidence, and the exact sign test has minimum two-sided p=0.25."
        ),
        **analysis,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        OUTPUT_DIR / "summary.csv",
        analysis["summaries"],
        ["benchmark", "variant", "n_seeds", "rollout_mse_mean", "rollout_mse_std", "rollout_mse_min", "rollout_mse_max"],
    )
    _write_csv(
        OUTPUT_DIR / "paired_deltas.csv",
        analysis["paired"],
        ["benchmark", "comparison", "variant", "n_pairs", "mean_delta_rollout_mse", "std_delta_rollout_mse", "min_delta_rollout_mse", "max_delta_rollout_mse", "seeds_variant_better", "seeds_full_better", "exact_sign_pvalue", "interpretation"],
    )
    print(f"WROTE {OUTPUT_DIR / 'analysis.json'} comparisons={len(analysis['paired'])}")


if __name__ == "__main__":
    main()
