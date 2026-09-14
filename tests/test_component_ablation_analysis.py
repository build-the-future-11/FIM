from scripts.analyze_component_ablations import exact_sign_pvalue, summarize_rows


def _rows():
    rows = []
    values = {
        "full": [1.0, 1.1, 0.9],
        "no_memory": [1.2, 1.3, 1.1],
        "no_retrieval": [0.8, 0.9, 0.7],
        "no_salience_gating": [1.0, 1.0, 1.0],
    }
    for benchmark in ("lorenz96", "delayed_recall"):
        for variant, metrics in values.items():
            for seed, metric in zip((11, 23, 37), metrics, strict=True):
                rows.append({"benchmark": benchmark, "variant": variant, "seed": seed, "rollout_mse": metric})
    return rows


def test_analysis_is_paired_within_benchmark_and_seed() -> None:
    analysis = summarize_rows(_rows())
    target = next(
        row for row in analysis["paired"]
        if row["benchmark"] == "lorenz96" and row["variant"] == "no_memory"
    )
    assert target["n_pairs"] == 3
    assert abs(target["mean_delta_rollout_mse"] - 0.2) < 1e-12
    assert target["seeds_full_better"] == 3
    assert target["exact_sign_pvalue"] == 0.25


def test_three_seed_sign_test_cannot_be_called_confirmatory() -> None:
    assert exact_sign_pvalue([-1.0, -2.0, -3.0]) == 0.25
    assert exact_sign_pvalue([-1.0, 0.0, 1.0]) == 1.0
