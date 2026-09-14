from types import SimpleNamespace

import torch

from fim_experiments.results.run_results import run


class IncrementBenchmark:
    def sample_initial_state(self, batch_size: int, device="cpu"):
        return torch.zeros(batch_size, 3, device=device)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return state + 1.0

    def rollout(self, x0: torch.Tensor, steps: int):
        values = [x0]
        x = x0
        for _ in range(steps):
            x = self.step(x)
            values.append(x)
        return torch.stack(values, dim=1)


class StructuredIncrementModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def reset_state(self):
        self.reset_calls += 1

    def forward(self, x: torch.Tensor):
        prediction = (x + 1.0).reshape(x.shape[0], 1, 1, -1)
        salience = torch.ones(x.shape[0], 1, device=x.device)
        return SimpleNamespace(prediction=prediction, salience=salience)


class TupleIncrementModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x + 1.0, torch.ones(x.shape[0], 1, device=x.device)


class HoldStateModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x


def test_run_results_supports_structured_output_and_requested_artifacts(tmp_path):
    model = StructuredIncrementModel()
    benchmark = IncrementBenchmark()
    metrics_path = tmp_path / "metrics" / "run_results.json"
    plot_path = tmp_path / "figures" / "rollout.pdf"

    metrics = run(
        model,
        benchmark,
        device="cpu",
        rollout_steps=4,
        batch_size=2,
        plot_path=plot_path,
        save_results_path=metrics_path,
    )

    assert model.reset_calls == 1
    assert metrics["rollout_mse"] == 0.0
    assert metrics["final_step_mse"] == 0.0
    assert metrics["rollout_mae"] == 0.0
    assert metrics["final_step_mae"] == 0.0
    assert metrics["rollout_steps"] == 4
    assert metrics["evaluation_batch_size"] == 2
    assert metrics["salience_available"] is True
    assert metrics_path.exists()
    assert plot_path.exists()


def test_run_results_supports_legacy_tuple_output(tmp_path):
    metrics = run(
        TupleIncrementModel(),
        IncrementBenchmark(),
        device="cpu",
        rollout_steps=3,
        batch_size=2,
        save_results_path=tmp_path / "metrics.json",
    )

    assert metrics["rollout_mse"] == 0.0
    assert metrics["final_step_mse"] == 0.0
    assert metrics["salience_available"] is True


def test_rollout_metrics_exclude_seeded_initial_condition():
    metrics = run(
        HoldStateModel(),
        IncrementBenchmark(),
        device="cpu",
        rollout_steps=2,
        batch_size=1,
    )

    # Forecast errors are 1 at step 1 and 2 at step 2. The seeded t=0
    # condition is intentionally excluded, so MSE=(1^2 + 2^2)/2=2.5.
    assert metrics["rollout_mse"] == 2.5
    assert metrics["final_step_mse"] == 4.0
    assert metrics["rollout_mae"] == 1.5
    assert metrics["final_step_mae"] == 2.0
