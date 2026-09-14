from __future__ import annotations

import pytest
import torch

MODULES = {
    "diffusion": ["benchmark.diffusion_equation", "fim.benchmark.diffusion_equation"],
    "lorenz": ["benchmark.navier_stokes", "fim.benchmark.navier_stokes"],
}


def _maybe(import_helpers, module_names, attr):
    return import_helpers["maybe_import_attr"](module_names, attr)


def test_long_horizon_rollout_has_no_nans(import_helpers):
    DiffusionBenchmark = _maybe(import_helpers, MODULES["diffusion"], "DiffusionBenchmark")
    DiffusionConfig = _maybe(import_helpers, MODULES["diffusion"], "DiffusionConfig")
    Lorenz96Benchmark = _maybe(import_helpers, MODULES["lorenz"], "Lorenz96Benchmark")
    Lorenz96Config = _maybe(import_helpers, MODULES["lorenz"], "Lorenz96Config")
    if DiffusionBenchmark is None or DiffusionConfig is None or Lorenz96Benchmark is None or Lorenz96Config is None:
        pytest.skip("Required benchmarks not available")

    diffusion = DiffusionBenchmark(DiffusionConfig(height=8, width=8, channels=1, steps=20, stability_scale=0.9))
    u0 = diffusion.sample_initial_state(1, device="cpu")
    traj = diffusion.rollout(u0, steps=20)
    assert traj.shape == (1, 21, 1, 8, 8)
    assert torch.isfinite(traj).all()
    assert float(traj.abs().max().item()) < 100.0

    lorenz = Lorenz96Benchmark(Lorenz96Config(dimension=8, steps=20, burn_in=3, process_noise=0.0, observation_noise=0.0, normalize=False, stability_clip=10.0))
    x0 = lorenz.sample_initial_state(1, device="cpu")
    traj2 = lorenz.rollout(x0, steps=20)
    assert traj2.shape == (1, 21, 8)
    assert torch.isfinite(traj2).all()
    assert float(traj2.abs().max().item()) < 100.0
