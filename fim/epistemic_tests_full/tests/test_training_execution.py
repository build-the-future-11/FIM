from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn


class TinyBenchmark:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, batch_size: int, device="cpu", steps=None):
        self.calls += 1
        x = torch.randn(batch_size, 1, 4, 4, device=device)
        y = 0.5 * x + 0.1
        return x, y


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.net(x)


def _load_train(import_helpers):
    candidates = ["train", "fim.train"]
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["train"])
            return module.train
        except Exception:
            continue
    pytest.skip("train module not available")


def _filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def test_training_execution_and_checkpoint(tmp_path: Path, import_helpers):
    train_fn = _load_train(import_helpers)
    model = TinyModel()
    bench = TinyBenchmark()

    ckpt_dir = tmp_path / "ckpts"
    kwargs = _filter_kwargs(
        train_fn,
        dict(
            model=model,
            train_loader=None,
            val_loader=None,
            benchmark=bench,
            epochs=2,
            device="cpu",
            lr=1e-2,
            weight_decay=0.0,
            grad_clip=1.0,
            use_amp=False,
            batch_size=4,
            steps_per_epoch=2,
            val_batches=1,
            rollout_steps=1,
            teacher_forcing_ratio=0.0,
            horizon_decay=1.0,
            ema_decay=0.9,
            use_ema=True,
            evaluate_ema=True,
            scheduler_per_batch=True,
            checkpoint_dir=ckpt_dir,
            resume_from=None,
            save_best=True,
            save_last=True,
            checkpoint_name="test_checkpoint.pt",
        ),
    )

    history = train_fn(**kwargs)
    assert isinstance(history, dict)
    assert len(history.get("train_loss", [])) >= 1
    assert len(history.get("val_loss", [])) >= 1
    assert (ckpt_dir / "test_checkpoint.pt").exists()
    assert (ckpt_dir / "best_test_checkpoint.pt").exists()

    kwargs2 = dict(kwargs)
    kwargs2["resume_from"] = ckpt_dir / "test_checkpoint.pt"
    history2 = train_fn(**_filter_kwargs(train_fn, kwargs2))
    assert isinstance(history2, dict)
