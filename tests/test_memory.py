import torch

from fim.memory.trace_bank import TraceBank
from fim.memory.compression import TraceCompressor
from fim.memory.consolidation import MemoryConsolidator
from fim.memory.retrieval import MemoryRetrieval
from fim.models.fim_model import FIMModel


def test_trace_bank_add_and_retrieve():
    bank = TraceBank(key_dim=8, value_dim=8, max_traces=16)
    key = torch.randn(8)
    value = torch.randn(8)

    bank.add(key=key, value=value, salience=0.9, timestamp=1)
    assert len(bank) == 1

    retriever = MemoryRetrieval(query_dim=8, value_dim=8, top_k=1)
    out = retriever(key.unsqueeze(0), bank)

    assert out.context.shape == (1, 8)
    assert out.weights.shape[-1] == 1
    assert torch.isfinite(out.context).all()


def test_memory_consolidation_and_compression():
    fabric = torch.randn(2, 16, 8, 8)
    compressor = TraceCompressor(latent_channels=16, trace_dim=8)
    compressed = compressor(fabric)

    bank = TraceBank(key_dim=8, value_dim=8, max_traces=8)
    consolidator = MemoryConsolidator(salience_threshold=0.4)

    salience = torch.tensor([0.9, 0.1])
    stored = consolidator(
        bank=bank,
        key=compressed.key,
        value=compressed.value,
        salience=salience,
        timestamp=0,
    )

    assert stored.dtype == torch.bool
    assert stored.shape == (2,)
    assert len(bank) == 1


def test_memory_consolidation_honors_novelty_and_closed_write_gate():
    bank = TraceBank(key_dim=4, value_dim=3, max_traces=8)
    consolidator = MemoryConsolidator(
        salience_threshold=0.5,
        min_store_probability=0.0,
        novelty_bias=1.0,
    )
    consolidator.eval()
    salience = torch.tensor([0.2, 0.9, 0.6])
    novelty = torch.tensor([0.4, 0.0, 0.0])
    write_gate = torch.tensor([1.0, 1.0, 0.0])

    stored = consolidator(
        bank=bank,
        key=torch.randn(3, 4),
        value=torch.randn(3, 3),
        salience=salience,
        novelty=novelty,
        write_gate=write_gate,
        timestamp=0,
    )
    analysis = consolidator.analyze(
        salience=salience,
        novelty=novelty,
        write_gate=write_gate,
    )

    assert stored.tolist() == [True, True, False]
    assert analysis["write_probability"].tolist() == [1.0, 1.0, 0.0]
    assert len(bank) == 2


def test_fim_model_step_updates_memory():
    model = FIMModel(
        in_channels=1,
        out_channels=1,
        latent_channels=16,
        trace_dim=8,
        hidden_channels=32,
        memory_threshold=0.0,
    )

    x = torch.randn(2, 1, 16, 16)
    y, out = model.step(x, update_memory=True)

    assert y.shape == (2, 1, 16, 16)
    assert out.state.shape == (2, 16, 16, 16)
    assert torch.isfinite(y).all()
    assert len(model.trace_bank) >= 1
