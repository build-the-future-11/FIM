from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'fim_experiments'
_exp_path = str(EXP)
_added_path = _exp_path not in sys.path
if _added_path:
    sys.path.insert(0, _exp_path)
try:
    from ablation_systems import AblatedFIMSystem, variant_switches
finally:
    # Do not leak the legacy experiment directory into pytest's global import
    # search path. Some epistemic tests intentionally probe top-level `train`
    # availability; leaving this path installed changes which module they test.
    if _added_path:
        sys.path.remove(_exp_path)


def _model(**switches):
    return AblatedFIMSystem(
        in_channels=1,
        hidden=8,
        trace_dim=4,
        memory_capacity=16,
        retrieval_topk=4,
        memory_decay=0.0,
        salience_threshold=-1.0,
        **switches,
    )


def test_no_memory_never_stores_or_retrieves():
    model = _model(**variant_switches('no_memory'))
    x = torch.randn(2, 1, 1, 8)
    first = model(x)
    second = model(x)
    assert len(model.bank) == 0
    assert first.retrieved is None
    assert second.retrieved is None


def test_no_retrieval_stores_but_never_reads():
    model = _model(**variant_switches('no_retrieval'))
    x = torch.randn(2, 1, 1, 8)
    first = model(x)
    stored_after_first = len(model.bank)
    second = model(x)
    assert stored_after_first > 0
    assert len(model.bank) >= stored_after_first
    assert first.retrieved is None
    assert second.retrieved is None


def test_full_stores_then_retrieves():
    model = _model(**variant_switches('full'))
    x = torch.randn(2, 1, 1, 8)
    first = model(x)
    second = model(x)
    assert first.retrieved is None
    assert len(model.bank) > 0
    assert second.retrieved is not None


def test_no_salience_gating_stores_even_above_threshold():
    model = AblatedFIMSystem(
        in_channels=1,
        hidden=8,
        trace_dim=4,
        memory_capacity=16,
        retrieval_topk=4,
        memory_decay=0.0,
        salience_threshold=2.0,
        **variant_switches('no_salience_gating'),
    )
    x = torch.randn(2, 1, 1, 8)
    model(x)
    assert len(model.bank) > 0


def test_unknown_historical_variant_fails_closed():
    try:
        variant_switches('no_spectral_mixing')
    except ValueError as exc:
        assert 'Unsupported current-code FIM ablation variant' in str(exc)
    else:
        raise AssertionError('historical unsupported variant must fail closed')


def test_memory_retrieval_is_isolated_between_batch_trajectories():
    model = _model(**variant_switches('full'))
    first = torch.stack((torch.full((1, 1, 8), -3.0), torch.full((1, 1, 8), 3.0)))
    model(first)

    owners = torch.tensor([0, 1])
    queries = model.compressor(model.dynamics(model.encoder(first)))[0]
    retrieved = model.bank.retrieve(queries, topk=16, owner=owners)

    assert retrieved is not None
    owner0_values = model.bank.values[: len(model.bank)][model.bank.owners[: len(model.bank)] == 0]
    owner1_values = model.bank.values[: len(model.bank)][model.bank.owners[: len(model.bank)] == 1]
    assert owner0_values.numel() > 0 and owner1_values.numel() > 0
    assert torch.cdist(retrieved[0:1], owner0_values).min() < torch.cdist(retrieved[0:1], owner1_values).min()
    assert torch.cdist(retrieved[1:2], owner1_values).min() < torch.cdist(retrieved[1:2], owner0_values).min()
