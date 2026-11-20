"""Regression coverage for the closed 24-cell historical FIM lineage."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from test_component_ablation_analysis import _rows

from scripts.analyze_component_ablations import (
    exact_sign_pvalue,
    load_retained_text_evidence,
    summarize_rows,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/current_component_ablations/manifest.json"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "0.1"])
def test_rejects_invalid_seed_mse(value):
    rows = _rows()
    rows[0]["rollout_mse"] = value
    with pytest.raises(ValueError, match="rollout MSE"):
        summarize_rows(rows)


def test_rejects_missing_cell_instead_of_silently_using_two_pairs():
    rows = _rows()
    rows.pop()
    with pytest.raises(ValueError, match="24"):
        summarize_rows(rows)


def test_rejects_duplicate_cell_instead_of_overwriting_seed_pair():
    rows = _rows()
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(rows)


@pytest.mark.parametrize("value", [11.5, True, "11"])
def test_rejects_coerced_seed_identity(value):
    rows = _rows()
    rows[0]["seed"] = value
    with pytest.raises(ValueError, match="integer seed"):
        summarize_rows(rows)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True]])
def test_rejects_undefined_sign_test(values):
    with pytest.raises(ValueError):
        exact_sign_pvalue(values)


def test_valid_zero_ties_are_retained():
    assert exact_sign_pvalue([0.0, 0.0, 0.0]) == 1.0


def test_row_permutation_preserves_complete_analysis():
    rows = _rows()
    expected = summarize_rows(rows)
    rows.reverse()
    assert summarize_rows(rows) == expected


def test_audits_all_96_text_artifacts_without_checkpoint_access(monkeypatch):
    read = Path.read_bytes
    paths = []

    def track(path):
        paths.append(path)
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", track)
    data, verification = load_retained_text_evidence(ROOT)
    assert verification["verified_text_artifacts"] == 96
    assert verification["manifest_sha256"] == "6350bd2252e75a4d1886f9f6f6f817ea1ff3256f30549a81b27396e37c555e9a"
    assert verification["checkpoint_artifacts_read"] == 0
    assert all(path.suffix != ".pt" for path in paths)
    analysis = summarize_rows(data["runs"])
    assert min(row["exact_sign_pvalue"] for row in analysis["paired"]) == 0.5
    assert sum(row["mean_delta_rollout_mse"] > 0 for row in analysis["paired"]) == 3
    assert sum(row["mean_delta_rollout_mse"] < 0 for row in analysis["paired"]) == 3


def test_manifest_metrics_must_match_retained_reports(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["runs"][0]["rollout_mse"] += 0.01
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagrees with retained cell reports"):
        load_retained_text_evidence(ROOT, altered)


def test_rejects_escape_from_retained_repository(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["runs"][0]["summary_file"] = "../outside.json"
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leaves repository"):
        load_retained_text_evidence(ROOT, altered)


def test_rejects_protocol_drift(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["protocol"]["batch_size"] = 1
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shared-minibatch protocol"):
        load_retained_text_evidence(ROOT, altered)


@pytest.mark.parametrize("rehash", [False, True])
def test_rejects_changed_cell_report_even_if_its_digest_is_updated(tmp_path, rehash):
    local = tmp_path / "repository"
    shutil.copytree(ROOT / "results", local / "results")
    manifest = local / "results/current_component_ablations/manifest.json"
    data = json.loads(manifest.read_text())
    row = data["runs"][0]
    target = local / row["summary_file"]
    report = json.loads(target.read_text())
    report["rollout_mse"] += 0.02
    target.write_text(json.dumps(report))
    if rehash:
        row["artifact_sha256"]["summary_file"] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(data))
    message = "disagrees with retained cell reports" if rehash else "hash mismatch"
    with pytest.raises(ValueError, match=message):
        load_retained_text_evidence(local)
