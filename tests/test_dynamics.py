import torch

from fim.operators.diffusion import DiffusionOperator
from fim.operators.advection import AdvectionOperator
from fim.operators.reaction import ReactionOperator
from fim.operators.damping import DampingOperator


def test_diffusion_smooths_signal():
    B, C, H, W = 1, 1, 16, 16

    x = torch.zeros(B, C, H, W)
    x[:, :, 8, 8] = 1.0  # impulse

    diffusion = DiffusionOperator(channels=1, diffusivity=0.1)

    y = diffusion(x)

    # center should decrease, neighbors increase
    assert y.abs().sum() > 0
    assert y[0, 0, 8, 8] < 0  # Laplacian center negative


def test_advection_moves_signal():
    B, C, H, W = 1, 1, 16, 16

    x = torch.zeros(B, C, H, W)
    x[:, :, 8, 8] = 1.0

    advection = AdvectionOperator(channels=1, velocity=(1.0, 0.0))

    y = advection(x)

    # should not be zero
    assert y.abs().sum() > 0


def test_reaction_nontrivial():
    B, C, H, W = 2, 8, 8, 8

    x = torch.randn(B, C, H, W)
    reaction = ReactionOperator(channels=C)

    y = reaction(x)

    assert y.shape == x.shape
    assert not torch.allclose(x, y)


def test_damping_reduces_magnitude():
    B, C, H, W = 2, 4, 8, 8

    x = torch.randn(B, C, H, W)
    damping = DampingOperator(channels=C, damping=0.5)

    y = damping(x)

    # damping is negative proportional
    assert torch.allclose(y, -0.5 * x, atol=1e-5)


def test_combined_dynamics_stability():
    B, C, H, W = 2, 8, 8, 8

    x = torch.randn(B, C, H, W)

    diffusion = DiffusionOperator(C, 0.1)
    advection = AdvectionOperator(C, velocity=(0.5, 0.5))
    reaction = ReactionOperator(C)
    damping = DampingOperator(C, 0.1)

    for _ in range(20):
        dx = (
            diffusion(x)
            + advection(x)
            + reaction(x)
            + damping(x)
        )
        x = x + 0.1 * dx

    assert torch.isfinite(x).all()
    assert torch.norm(x) < 1e5