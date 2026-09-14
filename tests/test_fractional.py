import torch

from fim.physics.fractional_pde import FractionalConfig, FractionalDiffusionBenchmark, fractional_weights


def test_fractional_weights_sum_is_finite():
    w = fractional_weights(alpha=0.8, steps=12)
    assert torch.isfinite(w).all()
    assert w.numel() == 12


def test_fractional_rollout_shape():
    cfg = FractionalConfig(alpha=0.8, diffusivity=0.05, dt=0.01, history=8, steps=6)
    model = FractionalDiffusionBenchmark(cfg)

    u0 = torch.randn(2, 1, 16, 16)
    traj = model.rollout(u0, steps=6)

    assert traj.shape == (2, 7, 1, 16, 16)
    assert torch.isfinite(traj).all()