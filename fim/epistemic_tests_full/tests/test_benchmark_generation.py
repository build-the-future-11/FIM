from __future__ import annotations

import pytest
import torch

MODULES = {
    "diffusion": ["benchmark.diffusion_equation", "fim.benchmark.diffusion_equation"],
    "lorenz": ["benchmark.navier_stokes", "fim.benchmark.navier_stokes"],
    "wave": ["benchmark.wave_equation", "fim.benchmark.wave_equation"],
}


def _maybe(import_helpers, module_names, attr):
    return import_helpers["maybe_import_attr"](module_names, attr)


def test_diffusion_benchmark_generation(import_helpers):
    DiffusionBenchmark = _maybe(import_helpers, MODULES["diffusion"], "DiffusionBenchmark")
    DiffusionConfig = _maybe(import_helpers, MODULES["diffusion"], "DiffusionConfig")
    if DiffusionBenchmark is None or DiffusionConfig is None:
        pytest.skip("Diffusion benchmark not available")

    bench = DiffusionBenchmark(DiffusionConfig(height=8, width=8, channels=1, steps=4, stability_scale=0.95))
    u0 = bench.sample_initial_state(2, device="cpu")
    assert u0.shape == (2, 1, 8, 8)
    traj = bench.rollout(u0, steps=3)
    assert traj.shape == (2, 4, 1, 8, 8)
    x, y = bench.generate_batch(2, steps=3, device="cpu")
    assert x.shape == (2, 3, 1, 8, 8)
    assert y.shape == (2, 3, 1, 8, 8)
    assert torch.isfinite(traj).all()
    assert bench.metrics()["state_shape"] == (1, 8, 8)


def test_lorenz_benchmark_generation(import_helpers):
    Lorenz96Benchmark = _maybe(import_helpers, MODULES["lorenz"], "Lorenz96Benchmark")
    Lorenz96Config = _maybe(import_helpers, MODULES["lorenz"], "Lorenz96Config")
    if Lorenz96Benchmark is None or Lorenz96Config is None:
        pytest.skip("Lorenz benchmark not available")

    bench = Lorenz96Benchmark(Lorenz96Config(dimension=8, steps=5, burn_in=2, process_noise=0.0, observation_noise=0.0, normalize=False))
    x0 = bench.sample_initial_state(3, device="cpu")
    assert x0.shape == (3, 8)
    traj = bench.rollout(x0, steps=4)
    assert traj.shape == (3, 5, 8)
    x, y = bench.generate_batch(3, steps=4, device="cpu")
    assert x.shape[-1] == 8
    assert y.shape[-1] == 8
    assert torch.isfinite(traj).all()
    assert bench.metrics()["dimension"] == 8


def test_wave_benchmark_generation(import_helpers):
    WaveBenchmark = _maybe(import_helpers, MODULES["wave"], "WaveBenchmark")
    WaveConfig = _maybe(import_helpers, MODULES["wave"], "WaveConfig")
    if WaveBenchmark is None or WaveConfig is None:
        pytest.skip("Wave benchmark not available")

    bench = WaveBenchmark(WaveConfig(height=8, width=8, channels=1, steps=4, spectral=False, boundary="periodic", stability_scale=0.95))
    u0, v0 = bench.sample_initial_state(2, 8, 8, channels=1, device="cpu", mode="gaussian")
    assert u0.shape == (2, 1, 8, 8)
    assert v0.shape == (2, 1, 8, 8)
    traj = bench.rollout(u0, v0, steps=3)
    assert traj.shape == (2, 4, 1, 8, 8)
    x, y = bench.generate_batch(2, steps=3, device="cpu")
    assert x.shape == (2, 3, 1, 8, 8)
    assert y.shape == (2, 3, 1, 8, 8)
    assert torch.isfinite(traj).all()
