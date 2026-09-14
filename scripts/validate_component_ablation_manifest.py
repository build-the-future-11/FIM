#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

EXPECTED_BENCHMARKS = ["lorenz96", "delayed_recall"]
EXPECTED_SEEDS = [11, 23, 37]
EXPECTED_VARIANTS = ["full", "no_memory", "no_retrieval", "no_salience_gating"]
EXPECTED_SWITCHES = {
    "full": (True, True, True),
    "no_memory": (False, False, False),
    "no_retrieval": (True, False, True),
    "no_salience_gating": (True, True, False),
}
PROTOCOL_FIELDS = (
    "epochs",
    "batch_size",
    "dataset_size",
    "train_rollout_steps",
    "eval_rollout_steps",
)
REQUIRED_RUN_FIELDS = {
    "benchmark",
    "seed",
    "variant",
    "memory_enabled",
    "retrieval_enabled",
    "salience_gating_enabled",
    "started_utc",
    "ended_utc",
    "git_commit",
    "source_identity",
    "identity_kind",
    "experiment_name",
    "results_root",
    "summary_file",
    "run_results_file",
    "resolved_config_file",
    "checkpoint_file",
    "log_file",
    "artifact_sha256",
    *PROTOCOL_FIELDS,
}


def _cell(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("benchmark"), row.get("seed"), row.get("variant")


ARTIFACT_FIELDS = (
    "summary_file",
    "run_results_file",
    "resolved_config_file",
    "checkpoint_file",
    "log_file",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(data: dict[str, Any], root: Path | None = None) -> list[str]:
    errors: list[str] = []

    if data.get("benchmarks") != EXPECTED_BENCHMARKS:
        errors.append(f"benchmarks must equal {EXPECTED_BENCHMARKS!r}")
    if data.get("seeds") != EXPECTED_SEEDS:
        errors.append(f"seeds must equal {EXPECTED_SEEDS!r}")
    if data.get("variants") != EXPECTED_VARIANTS:
        errors.append(f"variants must equal {EXPECTED_VARIANTS!r}")

    top_commit = data.get("git_commit")
    top_identity = data.get("source_identity")
    if data.get("identity_kind") != "sha256_source_tree":
        errors.append("identity_kind must be 'sha256_source_tree'")
    if not isinstance(top_identity, str) or len(top_identity) != 64:
        errors.append("top-level source_identity must be a 64-character SHA-256")
    else:
        try:
            int(top_identity, 16)
        except ValueError:
            errors.append("top-level source_identity must be hexadecimal")

    runs = data.get("runs")
    if not isinstance(runs, list):
        errors.append("runs must be a list")
        return errors

    expected_cells = set(product(EXPECTED_BENCHMARKS, EXPECTED_SEEDS, EXPECTED_VARIANTS))
    observed_cells: list[tuple[Any, Any, Any]] = []
    protocol_signatures: set[tuple[Any, ...]] = set()

    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            errors.append(f"run[{index}] must be an object")
            continue

        missing_fields = sorted(REQUIRED_RUN_FIELDS - row.keys())
        if missing_fields:
            errors.append(f"run[{index}] missing required fields: {missing_fields}")

        cell = _cell(row)
        observed_cells.append(cell)
        if cell not in expected_cells:
            errors.append(f"run[{index}] unexpected cell: {cell!r}")

        if top_commit not in (None, "", "UNKNOWN") and row.get("git_commit") != top_commit:
            errors.append(
                f"run[{index}] git_commit {row.get('git_commit')!r} != manifest git_commit {top_commit!r}"
            )
        if row.get("source_identity") != top_identity:
            errors.append(
                f"run[{index}] source_identity {row.get('source_identity')!r} != manifest source_identity {top_identity!r}"
            )
        if row.get("identity_kind") != "sha256_source_tree":
            errors.append(f"run[{index}] identity_kind must be 'sha256_source_tree'")

        hashes = row.get("artifact_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_FIELDS):
            errors.append(f"run[{index}] artifact_sha256 must cover every required artifact")
        elif root is not None:
            for field in ARTIFACT_FIELDS:
                relative = row.get(field)
                path = root / relative if isinstance(relative, str) else None
                if path is None or not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"run[{index}] missing/nonempty artifact for {field}: {relative!r}")
                elif hashes[field] != _sha256(path):
                    errors.append(f"run[{index}] {field} hash differs from manifest")

        variant = row.get("variant")
        if variant in EXPECTED_SWITCHES:
            observed_switches = (
                row.get("memory_enabled"),
                row.get("retrieval_enabled"),
                row.get("salience_gating_enabled"),
            )
            if observed_switches != EXPECTED_SWITCHES[variant]:
                errors.append(
                    f"run[{index}] switch state {observed_switches!r} does not match variant {variant!r}"
                )

        if all(field in row for field in PROTOCOL_FIELDS):
            protocol_signatures.add(tuple(row[field] for field in PROTOCOL_FIELDS))

    observed_set = set(observed_cells)
    duplicate_count = len(observed_cells) - len(observed_set)
    if duplicate_count:
        errors.append(f"manifest contains {duplicate_count} duplicate benchmark/seed/variant cell(s)")

    missing_cells = sorted(expected_cells - observed_set)
    if missing_cells:
        errors.append(f"manifest is missing {len(missing_cells)} expected cell(s): {missing_cells}")

    extra_cells = sorted(observed_set - expected_cells)
    if extra_cells:
        errors.append(f"manifest contains {len(extra_cells)} unexpected cell(s): {extra_cells}")

    if len(runs) != len(expected_cells):
        errors.append(f"runs length must be {len(expected_cells)}, found {len(runs)}")

    if len(protocol_signatures) > 1:
        errors.append(
            "manifest mixes experiment protocol settings across cells: "
            f"{sorted(protocol_signatures, key=repr)!r}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed unless a FIM current-component ablation manifest is complete and internally consistent."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("results/current_component_ablations/manifest.json"),
    )
    args = parser.parse_args()

    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"INVALID manifest: unable to read/parse {args.manifest}: {exc}")
        return 2

    if not isinstance(data, dict):
        print("INVALID manifest: top level must be a JSON object")
        return 2

    errors = validate_manifest(data, Path(__file__).resolve().parents[1])
    if errors:
        print("INVALID current-component ablation evidence:")
        for error in errors:
            print(f"- {error}")
        return 2

    print(
        "VALID current-component ablation evidence: "
        f"{len(data['runs'])} unique cells, one commit, one frozen protocol"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
