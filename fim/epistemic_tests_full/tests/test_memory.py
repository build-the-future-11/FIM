from __future__ import annotations

import pytest
import torch


def _load(name_candidates, attr, import_helpers):
    return import_helpers["maybe_import_attr"](name_candidates, attr)


@pytest.mark.parametrize(
    "trace_bank_candidates,retrieval_candidates,consolidator_candidates,persistent_candidates",
    [(
        ["memory.trace_bank", "fim.memory.trace_bank"],
        ["memory.retrieval", "fim.memory.retrieval"],
        ["memory.consolidation", "fim.memory.consolidation"],
        ["memory", "fim.memory"],
    )],
)
def test_trace_bank_and_retrieval(import_helpers, trace_bank_candidates, retrieval_candidates, consolidator_candidates, persistent_candidates):
    TraceBank = _load(trace_bank_candidates, "TraceBank", import_helpers)
    MemoryRetrieval = _load(retrieval_candidates, "MemoryRetrieval", import_helpers)
    MemoryConsolidator = _load(consolidator_candidates, "MemoryConsolidator", import_helpers)
    if TraceBank is None or MemoryRetrieval is None or MemoryConsolidator is None:
        pytest.skip("Memory subsystem not available")

    bank = TraceBank(key_dim=4, value_dim=3, max_traces=8, merge_threshold=0.9999)
    key1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    val1 = torch.tensor([0.5, 0.1, -0.2])
    key2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    val2 = torch.tensor([-0.5, 0.2, 0.9])

    bank.add(key1, val1, salience=0.9, timestamp=1)
    bank.add(key2, val2, salience=0.8, timestamp=2)

    assert len(bank) == 2
    keys, values, saliences, timestamps = bank.as_tensors()
    assert keys.shape == (2, 4)
    assert values.shape == (2, 3)
    assert saliences.shape == (2, 1)
    assert timestamps.shape == (2, 1)

    context = bank.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]), topk=1)
    assert context.shape == (1, 3)
    assert torch.allclose(context.squeeze(0), val1, atol=1e-4)

    retrieval = MemoryRetrieval(query_dim=4, value_dim=3, top_k=1)
    out = retrieval(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), bank)
    assert out.context.shape == (1, 3)
    assert out.weights.shape[-1] == 1
    assert out.indices.shape == (1, 1)
    assert torch.isfinite(out.context).all()

    consolidator = MemoryConsolidator(salience_threshold=0.5, min_store_probability=0.0, decay=0.0)
    # Training mode intentionally permits stochastic sub-threshold writes. This
    # assertion is about the hard salience threshold, so evaluate it using the
    # deterministic inference policy rather than relying on a random draw.
    consolidator.eval()
    bank2 = TraceBank(key_dim=4, value_dim=3, max_traces=8)
    stored = consolidator(
        bank=bank2,
        key=torch.stack([key1, key2], dim=0),
        value=torch.stack([val1, val2], dim=0),
        salience=torch.tensor([0.2, 0.8]),
        timestamp=3,
    )
    assert stored.dtype == torch.bool
    assert stored.tolist() == [False, True]
    assert len(bank2) == 1


@pytest.mark.parametrize("persistent_candidates", [["memory", "fim.memory"]])
def test_persistent_memory_system(import_helpers, persistent_candidates):
    PersistentMemorySystem = _load(persistent_candidates, "PersistentMemorySystem", import_helpers)
    if PersistentMemorySystem is None:
        pytest.skip("Persistent memory system not available")

    system = PersistentMemorySystem(obs_channels=1, latent_channels=4, trace_dim=6, hidden_channels=8, max_traces=16, top_k=2)
    system.scorer.threshold.data.fill_(-10.0)
    system.consolidator.salience_threshold.data.fill_(-10.0)

    obs = torch.randn(2, 1, 8, 8)
    pred = torch.randn(2, 1, 8, 8)
    before = torch.randn(2, 4, 8, 8)
    after = before + 0.1 * torch.randn_like(before)

    out = system(
        observation=obs,
        prediction=pred,
        fabric_before=before,
        fabric_after=after,
        timestamp=1,
        layer=0,
        return_full=True,
    )

    assert out.bank_size >= 1
    assert out.stored_mask.shape == (2,)
    assert out.salience.score.shape[0] == 2
    assert out.trace.key.shape[0] == 2
    assert "write_gate_mean" in out.trace.regularization
    assert "entropy" in out.retrieval.regularization
