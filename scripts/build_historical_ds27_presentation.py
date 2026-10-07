#!/usr/bin/env python3
"""Render a separate SIAM candidate from the closed historical FIM study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_component_ablations import (
    ROOT,
    load_retained_text_evidence,
    summarize_rows,
)

BASE_COMMIT = "142739465f0c3aa8efc674f64c793b818c9e48e5"
TITLE = "Retained Component Ablations in FIM: A Historical Three-Seed Evaluation"
ABSTRACT = (
    "We analyze a closed historical component study of Fabric-Induced Memory "
    "(FIM), comprising 24 cells: Lorenz96 and delayed recall, three training "
    "seeds, and four configurations per benchmark. We compare the full model "
    "with memory, retrieval, or salience-gating removals using paired rollout "
    "MSE differences. Mean directions favor the full model in three of six "
    "comparisons and a removal in the other three; the smallest exact two-sided "
    "sign-test p-value is 0.5. The no-memory and no-retrieval error sequences "
    "coincide in every retained seed, so they do not provide distinct evidence "
    "of component necessity. We verify 96 retained text-artifact hashes and "
    "cross-check manifest metrics against cell reports before generating the "
    "analysis. The shared-minibatch protocol and three training seeds constrain "
    "mechanistic interpretation. These are historical reported outcomes, "
    "without a new trajectory-isolated experiment or checkpoint replay. The "
    "closed result establishes neither consistent benefit nor equivalence, "
    "computational efficiency, or universal uselessness of memory."
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figure(analysis: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "fim-historical-24-cells",
        "pdf.fonttype": 42, "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.9), layout="constrained")
    variants = ["no_memory", "no_retrieval", "no_salience_gating"]
    for axis, benchmark, title in zip(axes, ("lorenz96", "delayed_recall"),
                                      ("A  Lorenz96", "B  Delayed recall")):
        for index, variant in enumerate(variants):
            row = next(r for r in analysis["paired"] if r["benchmark"] == benchmark and r["variant"] == variant)
            for offset, seed_row, marker, color in zip(
                (-0.11, 0, 0.11), row["seed_deltas"], ("o", "s", "^"),
                ("#146b8c", "#925aa4", "#a44538"),
            ):
                axis.scatter(index + offset, seed_row["delta"], marker=marker,
                             color=color, s=48, label=f"Seed {seed_row['seed']}" if index == 0 else None)
            axis.hlines(row["mean_delta_rollout_mse"], index - 0.23, index + 0.23,
                        color="#222222", linewidth=2, label="Mean" if index == 0 else None)
        axis.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
        axis.set(title=title, xticks=[0, 1, 2], xticklabels=["No memory", "No retrieval", "No salience"],
                 ylabel="Removal − full rollout MSE")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Historical FIM: mixed mean directions across six paired comparisons", fontsize=13)
    fig.supxlabel("Positive favors full; negative favors removal · panels use independent scales\n"
                  "Three paired training seeds per benchmark · smallest exact sign-test p = 0.5", fontsize=9)
    stem = destination / "historical_paired_comparisons"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


def build(destination: Path) -> dict:
    data, verification = load_retained_text_evidence(ROOT)
    analysis = summarize_rows(data["runs"])
    if len(ABSTRACT) > 1500:
        raise ValueError("SIAM candidate abstract exceeds 1,500 characters")
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "analysis.json", {
        "lineage": "historical-24-cell-shared-minibatch-lorenz96-delayed-recall",
        "baseline_commit": BASE_COMMIT, "protocol": data["protocol"],
        "disposition": "closed negative; descriptive seed comparisons",
        "scope": "Distinct from Burgers V2, delayed-recall V2 and the held trajectory-isolated successor.",
        **analysis,
    })
    _json(destination / "SOURCE_ARTIFACTS.json", verification)
    _csv(destination / "cells.csv", data["runs"], ["benchmark", "seed", "variant", "rollout_mse", "rollout_mae"])
    _csv(destination / "paired_comparisons.csv", analysis["paired"], [
        "benchmark", "variant", "n_pairs", "mean_delta_rollout_mse", "std_delta_rollout_mse",
        "seeds_variant_better", "seeds_full_better", "exact_sign_pvalue", "interpretation",
    ])
    (destination / "abstract_title.txt").write_text(TITLE + "\n", encoding="utf-8")
    (destination / "abstract.txt").write_text(ABSTRACT + "\n", encoding="utf-8")
    _json(destination / "abstract_metadata.json", {
        "status": "separate historical-study candidate; not submitted",
        "characters_including_spaces": len(ABSTRACT), "limit": 1500,
        "title_counted_separately": True,
        "remaining_inputs": ["Author/presenter list, affiliation and historical-study scope approval",
                             "SIAM DS27 topic fit, presentation choice and attendance/date-conflict confirmation"],
    })
    _figure(analysis, destination)
    _json(destination / "MANIFEST.json", {
        "schema": "fim-historical-presentation-v1", "baseline_commit": BASE_COMMIT,
        "manifest_sha256": verification["manifest_sha256"],
        "text_artifacts_verified": 96, "checkpoints_read": 0,
        "scripts_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(__file__).resolve(), ROOT / "scripts/analyze_component_ablations.py")},
        "generated_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(destination.iterdir())
                             if path.is_file() and path.name not in ("MANIFEST.json", "README.md", "poster_text.md")},
    })
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "publication/historical_ds27_candidate")
    args = parser.parse_args()
    verification = build(args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "abstract_characters": len(ABSTRACT),
                      "verified_text_artifacts": verification["verified_text_artifacts"],
                      "checkpoints_read": 0, "manifest_sha256": verification["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
