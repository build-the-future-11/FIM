import json

import pytest
import torch

from fim.data import prepare_dataset, validate_dataset


def test_prepare_is_deterministic_and_splits_do_not_overlap(tmp_path):
    first = prepare_dataset(benchmark_name="lorenz96", benchmark_config={"dimension": 8, "burn_in": 2}, output_dir=tmp_path / "a", size=8, steps=3, seed=7)
    second = prepare_dataset(benchmark_name="lorenz96", benchmark_config={"dimension": 8, "burn_in": 2}, output_dir=tmp_path / "b", size=8, steps=3, seed=7)
    assert first["tensor_sha256"] == second["tensor_sha256"]
    assert validate_dataset(tmp_path / "a")["valid"] is True
    payload = torch.load(tmp_path / "a" / "dataset.pt", weights_only=True)
    assert set(payload["splits"]["train"]).isdisjoint(payload["splits"]["validation"])


def test_validate_rejects_manifest_checksum_tampering(tmp_path):
    prepare_dataset(benchmark_name="lorenz96", benchmark_config={"dimension": 8, "burn_in": 1}, output_dir=tmp_path, size=6, steps=2, seed=3)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tensor_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor checksum mismatch"):
        validate_dataset(tmp_path)
