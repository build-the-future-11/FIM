from __future__ import annotations

import pytest
import torch

MODULES = ["benchmark.levy_stable", "fim.benchmark.levy_stable"]


def _maybe(import_helpers, attr):
    return import_helpers["maybe_import_attr"](MODULES, attr)


def test_levy_increment_path_and_moments(import_helpers):
    LevyStableProcess = _maybe(import_helpers, "LevyStableProcess")
    LevyStableConfig = _maybe(import_helpers, "LevyStableConfig")
    if LevyStableProcess is None or LevyStableConfig is None:
        pytest.skip("Levy stable module not available")

    proc = LevyStableProcess(LevyStableConfig(alpha=1.5, beta=0.0, scale=0.5, loc=0.0, dim=4, steps=5, dt=1.0, stability_clip=20.0, normalize=False))
    inc = proc.sample_increment(3, device="cpu")
    assert inc.shape == (3, 4)
    assert torch.isfinite(inc).all()

    path = proc.sample_path(steps=4, batch_size=3, device="cpu")
    assert path.shape == (3, 5, 4)
    assert torch.isfinite(path).all()

    x, y = proc.generate_batch(batch_size=3, steps=4, device="cpu")
    assert x.shape == (3, 4, 4)
    assert y.shape == (3, 4, 4)

    moments = proc.sample_moments(num_samples=200, device="cpu")
    assert "mean" in moments and "std" in moments and "mad" in moments
    assert moments["mean"].shape == (4,)

    tail = proc.tail_index_estimate(num_samples=200, device="cpu")
    assert torch.isfinite(tail)
    assert tail.item() > 0.0

    assert proc.metrics()["stochastic"] is True
