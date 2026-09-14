import json
from pathlib import Path

from scripts.analyze_component_ablations import summarize_rows
from scripts.build_component_ablation_paper import manuscript


ROOT = Path(__file__).resolve().parents[1]


def test_generated_paper_preserves_claim_boundaries_and_real_baseline_win() -> None:
    rows = []
    for benchmark in ("lorenz96", "delayed_recall"):
        for variant, values in {
            "full": [1.0, 1.1, 0.9],
            "no_memory": [1.2, 1.3, 1.1],
            "no_retrieval": [0.8, 0.9, 0.7],
            "no_salience_gating": [1.0, 1.0, 1.0],
        }.items():
            for seed, value in zip((11, 23, 37), values, strict=True):
                rows.append({"benchmark": benchmark, "variant": variant, "seed": seed, "rollout_mse": value})
    analysis = {"source_identity": "a" * 64, **summarize_rows(rows)}
    mini = json.loads((ROOT / "results/mini_processed/mini_suite_summary.json").read_text())
    text = manuscript(analysis, mini, {"analysis": "b" * 64, "manifest": "c" * 64, "mini_suite": "d" * 64})

    assert "smallest possible two-sided exact sign p-value is 0.25" in text
    assert "DeepONetFieldSystem has the lowest stored rollout MSE, not FIM" in text
    assert "does not support a blanket superiority claim" in text
    assert "TODO" not in text
    assert "placeholder" not in text.lower()
    assert len(text.split()) > 2_000
