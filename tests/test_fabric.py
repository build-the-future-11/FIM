import torch

from fim.core.fabric import FabricState
from fim.models.latent_operator import LatentOperator


def test_fabric_initialization():
    fabric = FabricState(
        batch_size=2,
        channels=16,
        height=8,
        width=8,
        device="cpu",
    )

    F = fabric.tensor
    assert F.shape == (2, 16, 8, 8)
    assert torch.allclose(F, torch.zeros_like(F))


def test_fabric_update_step_stability():
    B, C, H, W = 2, 16, 8, 8

    fabric = FabricState(B, C, H, W)
    operator = LatentOperator(channels=C)

    F = fabric.tensor
    z = torch.randn(B, C, H, W)

    F_next = operator(F, z)

    assert F_next.shape == F.shape
    assert torch.isfinite(F_next).all()


def test_fabric_multiple_steps_no_explosion():
    B, C, H, W = 2, 16, 8, 8

    fabric = FabricState(B, C, H, W)
    operator = LatentOperator(channels=C)

    F = fabric.tensor

    for _ in range(50):
        z = torch.randn(B, C, H, W)
        F = operator(F, z)

    norm = torch.norm(F)

    # sanity: should not explode
    assert torch.isfinite(norm)
    assert norm < 1e4


def test_fabric_with_retrieval_injection():
    B, C, H, W = 2, 16, 8, 8

    fabric = FabricState(B, C, H, W)
    operator = LatentOperator(channels=C)

    F = fabric.tensor
    z = torch.randn(B, C, H, W)
    retrieval = torch.randn(B, C)

    F_next = operator(F, z, retrieval=retrieval)

    assert F_next.shape == (B, C, H, W)
    assert torch.isfinite(F_next).all()