from __future__ import annotations

import pytest
import torch

MODULES = ["benchmark.fractional_pde", "fim.benchmark.fractional_pde"]


def _maybe(import_helpers, attr):
    return import_helpers["maybe_import_attr"](MODULES, attr)


def test_fractional_weights_and_rollout(import_helpers):
    FractionalDiffusionBenchmark = _maybe(import_helpers, "FractionalDiffusionBenchmark")
    FractionalConfig = _maybe(import_helpers, "FractionalConfig")
    fractional_weights = _maybe(import_helpers, "fractional_weights")
    if FractionalDiffusionBenchmark is None or FractionalConfig is None or fractional_weights is None:
        pytest.skip("Fractional diffusion module not available")

    weights = fractional_weights(alpha=0.8, steps=5, device=torch.device("cpu"), dtype=torch.float32, normalize=True)
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()
    assert abs(float(weights.abs().sum().item()) - 1.0) < 1e-4

    bench = FractionalDiffusionBenchmark(FractionalConfig(height=8, width=8, channels=1, steps=4, history=4, alpha=0.8, diffusivity=0.05, stability_clip=5.0))
    u0 = bench.sample_initial_state(2, device="cpu")
    assert u0.shape == (2, 1, 8, 8)
    traj = bench.rollout(u0, steps=3)
    assert traj.shape == (2, 4, 1, 8, 8)
    assert torch.isfinite(traj).all()
    x, y = bench.generate_batch(2, steps=3, device="cpu")
    assert x.shape == (2, 3, 1, 8, 8)
    assert y.shape == (2, 3, 1, 8, 8)
    assert bench.metrics()["is_non_markovian"] is True
