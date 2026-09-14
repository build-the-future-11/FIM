#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .validate_component_ablation_manifest import validate_manifest
except ImportError:  # direct script execution
    from validate_component_ablation_manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/current_component_ablations/manifest.json"
OUTPUT_DIR = ROOT / "results/current_component_ablations"


def exact_sign_pvalue(values: list[float]) -> float:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    data = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
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
