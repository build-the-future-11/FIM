#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/current_component_ablations/analysis.json"
MINI_SUITE = ROOT / "results/mini_processed/mini_suite_summary.json"
OUTPUT = ROOT / "paper/fim-current-evidence.md"


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manuscript(analysis: dict[str, Any], mini: list[dict[str, Any]], hashes: dict[str, str]) -> str:
    summaries = analysis["summaries"]
    paired = analysis["paired"]
    full_better = sum(float(row["mean_delta_rollout_mse"]) > 0 for row in paired)
    reduced_better = sum(float(row["mean_delta_rollout_mse"]) < 0 for row in paired)
    zero = len(paired) - full_better - reduced_better
    best_mini = min(mini, key=lambda row: float(row["rollout_mse"]))
    fim_mini = next(row for row in mini if row["model"] == "FIMSystem")

    summary_rows = "\n".join(
        f"| {row['benchmark']} | {row['variant']} | {int(row['n_seeds'])} | {_fmt(row['rollout_mse_mean'])} | {_fmt(row['rollout_mse_std'])} | [{_fmt(row['rollout_mse_min'])}, {_fmt(row['rollout_mse_max'])}] |"
        for row in summaries
    )
    paired_rows = "\n".join(
        f"| {row['benchmark']} | {row['variant']} − full | {_fmt(row['mean_delta_rollout_mse'])} | [{_fmt(row['min_delta_rollout_mse'])}, {_fmt(row['max_delta_rollout_mse'])}] | {int(row['seeds_variant_better'])}/{int(row['n_pairs'])} | {float(row['exact_sign_pvalue']):.3f} |"
        for row in paired
    )
    mini_rows = "\n".join(
        f"| {row['model']} | {int(row['params'])} | {_fmt(row['rollout_mse'])} | {_fmt(row['final_step_mse'])} |"
        for row in sorted(mini, key=lambda item: float(item["rollout_mse"]))
    )
    mechanism_paragraphs = []
    for row in paired:
        delta = float(row["mean_delta_rollout_mse"])
        benchmark = str(row["benchmark"])
        variant = str(row["variant"])
        if delta > 0:
            direction = "the full system has lower mean rollout error"
        elif delta < 0:
            direction = "the reduced variant has lower mean rollout error"
        else:
            direction = "the mean rollout errors are equal"
        mechanism_paragraphs.append(
            f"For **{benchmark}**, `{variant} − full` is {_fmt(delta)} and {direction}. "
            f"The observed seed range is [{_fmt(row['min_delta_rollout_mse'])}, {_fmt(row['max_delta_rollout_mse'])}], "
            f"with the reduced variant better on {int(row['seeds_variant_better'])}/{int(row['n_pairs'])} seeds. "
            f"The two-sided exact sign p-value is {float(row['exact_sign_pvalue']):.3f}. This is a bounded directional result, not confirmatory evidence across tasks."
        )
    mechanism_text = "\n\n".join(mechanism_paragraphs)

    return f"""# When Does a Structured Memory Fabric Matter?

## A fail-closed component study across chaotic dynamics and delayed recall

### Abstract

Structured external memory is often evaluated by comparing a large memory-bearing model with an unrelated baseline, making it difficult to identify whether storage, retrieval, or selective writing caused an observed change. We evaluate the maintained Fabric-Induced Memory (FIM) implementation through explicit current-code interventions. A frozen matrix crosses two benchmarks, three training seeds, and four variants: full memory, no memory, stored-but-not-retrieved memory, and retrieval without salience-gated writing. Every cell retains resolved configuration, log, metrics, raw rollout results, and checkpoint under hash verification. Across the six benchmark-by-removal comparisons, the full system has lower mean rollout MSE in {full_better}, a reduced variant has lower mean error in {reduced_better}, and {zero} are exact mean ties. With only three paired seeds per benchmark, the smallest possible two-sided exact sign p-value is 0.25; results are therefore descriptive mechanism falsification rather than a significance claim. A separate retained delayed-recall mini-suite places {best_mini['model']} first at rollout MSE {_fmt(best_mini['rollout_mse'])}, while FIM reaches {_fmt(fim_mini['rollout_mse'])}. The evidence does not support a blanket superiority claim. It establishes which current components survive direct removal under two fixed tasks, exposes adverse ablations, and provides a reproducible path for larger independent-domain tests.

## 1. Research question

Long-horizon models can forget earlier observations even when the missing information remains useful. One response is to increase context, recurrent width, or state dimension. Another is to preserve selected latent traces and retrieve them when later states appear similar. FIM implements the second idea with a bounded adaptive memory bank, salience-based storage, similarity retrieval, age decay, and gated feedback into the current latent representation.

The scientific question is narrower than whether memory is intuitively attractive: **does the specific implemented storage–retrieval–gating pathway improve rollout error under a fixed training budget, and which component is necessary when it does?** A component is not supported merely because the full system trains. If disabling it matches or improves the full model, the corresponding necessity claim fails for that benchmark and protocol.

This paper deliberately separates three evidence layers. Semantic unit tests establish that switches manipulate the intended code paths. The fresh paired matrix measures current-code effects. An older mini-suite provides baseline context but is not silently reclassified as a reproduction of historical paper claims. These layers answer different questions and remain separate throughout.

## 2. System and interventions

The maintained model maps an input field through an encoder and latent dynamics operator, producing latent state `z`. A salience network scores the state, while a trace compressor emits a key and value. When retrieval is enabled and the memory bank is nonempty, normalized query–key similarity is adjusted by trace age and stored salience. The top-k values are softmax-weighted, projected into latent space, and applied through a learned sigmoid gate and bounded feedback scale. The decoder then produces the next-state prediction.

Writing is distinct from reading. Under the full system, the mean salience score must exceed a fixed threshold before a trace enters the bounded bank. Stored keys, values, scores, and ages are detached from the current graph. When capacity is exhausted, the replacement rule combines stored score with logarithmic age decay. This design is concrete and executable; the present study does not claim it is the only possible memory fabric.

Four interventions are defined directly on these code paths. `full` enables storage, retrieval, and salience gating. `no_memory` disables both storage and retrieval. `no_retrieval` permits storage but prevents retrieved traces from affecting prediction. `no_salience_gating` retains storage and retrieval but admits every produced trace. The historical label `no_spectral_mixing` is rejected because no current component has that semantic identity. Refusing that mapping prevents an obsolete paper label from becoming fabricated current evidence.

## 3. Frozen experiment design

The matrix contains 24 cells: Lorenz-96 and delayed recall, seeds 11, 23, and 37, and all four variants. Each cell uses 12 epochs, batch size 64, 2,048 generated examples, four training rollout steps, and 30 evaluation steps. The primary outcome is rollout mean-squared error. Variant effects are paired against `full` within the same benchmark and seed.

Lorenz-96 represents nonlinear chaotic dynamics in which small state errors can expand over a rollout. Delayed recall requires earlier information to influence later predictions and is therefore the more direct memory stress test. These tasks probe different failure modes, but they are generated benchmarks rather than independent real-world domains. A benefit on one cannot be generalized to scientific simulation, language modeling, or arbitrary long-context tasks.

Three seeds measure optimization variation under the fixed generator and protocol. They do not create three independent datasets. We report means, sample standard deviations, complete ranges, and seed-paired deltas. The exact sign test is included to make its low resolution visible: with three nonzero paired effects, even unanimous direction gives p=0.25. We do not use asymptotic normal tests or relabel a descriptive interval as population-level confidence.

## 4. Integrity and reproducibility

The source directory is not itself a Git checkout, so inventing a commit identifier would be false provenance. The runner instead records a deterministic SHA-256 over the executable `fim/` and `fim_experiments/` Python sources, the runner, the frozen protocol, and project metadata. That identity is {_fmt(0) if False else analysis['source_identity']}. Every run row repeats the identity and exact switch state.

The manifest is written atomically after each completed cell. Resumption is allowed only when the source identity, benchmark list, seeds, variants, and protocol match. Reuse additionally requires byte hashes for the summary, raw rollout result, resolved configuration, checkpoint, and log. A missing or modified artifact invalidates the cell rather than silently converting it into cached evidence.

Semantic tests force trace storage and verify observable behavior. `no_memory` neither writes nor retrieves; `no_retrieval` writes but never reads; `full` stores then retrieves on a later step; `no_salience_gating` writes even when the ordinary threshold would reject the trace; and unsupported labels raise an error. These tests establish intervention fidelity, not empirical usefulness.

## 5. Fresh component results

| Benchmark | Variant | Seeds | Rollout MSE mean | SD | Seed range |
|---|---|---:|---:|---:|---:|
{summary_rows}

The paired table reports reduced variant minus full, so negative values favor removal and positive values favor the complete system.

| Benchmark | Comparison | Mean paired delta | Seed range | Reduced wins | Exact sign p |
|---|---|---:|---:|---:|---:|
{paired_rows}

{mechanism_text}

These comparisons identify behavior only for the retained source identity and frozen settings. A component can be active yet unhelpful, and a removal can win because of optimization, regularization, capacity use, or task mismatch rather than because the general concept is invalid. The direct conclusion is therefore about necessity under the tested intervention, not universal memory value.

## 6. Retained baseline context

The repository also contains a compact delayed-recall mini-suite with persisted checkpoints, logs, configurations, and metrics. Its model sizes and training path differ from the fresh component matrix, so it is contextual evidence rather than a pooled comparison. The stored ranking is:

| Model | Parameters | Rollout MSE | Final-step MSE |
|---|---:|---:|---:|
{mini_rows}

{best_mini['model']} has the lowest stored rollout MSE, not FIM. The shape-aware MLP is also close to FIM in that artifact. This baseline win matters because a component study alone could show that memory helps relative to a reduced FIM while the entire architecture still loses to a simpler alternative. Conversely, a baseline win in one compact run does not prove universal inferiority. A credible successor must rerun strong baselines under identical seeds, data, budgets, and model-selection rules.

## 7. What the evidence establishes

The first contribution is methodological: every mechanism label maps to an executed switch whose semantics are unit-tested. This prevents the common failure in which an ablation command changes a configuration field that the model never reads. The second contribution is fail-closed provenance for a non-Git source bundle, including atomic progress and artifact hashes. The third is scientific contraction: the result is the pattern of mechanisms that survive removal, not a claim that a complex memory system must win.

The design can falsify necessity claims. If `no_memory` is not worse, the benchmark does not demonstrate a benefit from the implemented bank. If `no_retrieval` is not worse, storing traces without reading them is observationally sufficient under that task, and retrieved feedback is unsupported. If `no_salience_gating` is not worse, selective writing has not earned its complexity. Only consistent positive reduced-minus-full deltas would directionally favor the corresponding full component, and even unanimous three-seed direction remains exploratory.

The design cannot establish state of the art, scaling behavior, or domain transfer. Both datasets are generated internally. Hyperparameters were not selected through an independent multi-task development set. The full model and removals share most parameters but can differ in effective optimization and state dynamics. Rollout MSE summarizes prediction error but does not directly measure whether a retrieved trace contains the causally relevant earlier event.

## 8. Failure analysis and next experiments

An adverse memory ablation has several plausible causes. Similarity retrieval can select a superficially close but causally irrelevant trace. Gated feedback can perturb a useful latent trajectory. Salience can correlate with surprise rather than future utility. Capacity replacement can discard low-salience information needed after a long delay. Training with a short rollout can under-optimize components whose benefit appears only at longer horizons.

The next protocol should be frozen before new outcomes are inspected. It should include an oracle-retrieval control, random-retrieval control, parameter-matched no-memory control, and retrieval-precision labels on a benchmark where the causally relevant memory is known. Delay length, bank capacity, query noise, and distractor density should be crossed factorially. Independent public sequence or operator-learning datasets are required before any transfer claim. Multiple seeds remain useful for optimization stability, but uncertainty across domains must cluster by dataset rather than treating seeds as new domains.

A baseline-complete replication should retrain DeepONet, the shape-aware MLP, a selective state-space model, and a Transformer using the same generated samples and evaluation trajectories as every FIM variant. Compute, parameter count, and latency should accompany error. If a simpler model remains best, that is a publishable boundary: elaborate memory is unnecessary in the tested regime. If FIM wins only at long delays or high distractor density, the interaction—not an overall average—becomes the claim.

## 9. Threats to validity

Construct validity is limited by rollout MSE. The metric captures prediction deviation but not memory attribution, calibration, energy conservation, or stability. The delayed-recall task may align strongly with explicit storage, while Lorenz-96 may reward local dynamics more than episodic recall. These are useful stress tests but not a representative sample of all long-horizon modeling.

Internal validity is improved by paired seeds and explicit switches, yet removal changes the computation graph and can alter optimization. Equal epochs do not guarantee equal convergence. The protocol does not perform nested hyperparameter selection. Any threshold, capacity, or rollout change motivated by these outcomes belongs to a versioned successor study, not a repair of the current test set.

External validity is narrow. No natural language, video, industrial telemetry, or contemporary scientific dataset appears in the fresh matrix. The older mini-suite is a single retained package and is not an independent reproduction. Hardware and library differences may affect floating-point trajectories, especially in chaotic rollouts, even when source and seed identities match.

Statistical resolution is deliberately exposed. Three seeds cannot support precise tail probabilities, and seed-level sign tests do not justify cross-domain inference. Six mechanism comparisons also create multiplicity. Because none can attain two-sided p<0.05 with three nonzero pairs, we report all comparisons and avoid selective significance language rather than applying a correction to underpowered tests.

## 10. Responsible-use boundary

The experiments use generated dynamics and recall data; they do not involve people, private records, clinical decisions, or deployed control systems. The code is research software. Performance on Lorenz-96 does not establish safe physical forecasting, and delayed-recall performance does not establish trustworthy memory for agents. Deployment would require domain-specific validation, monitoring, failure containment, and governance that are absent here.

## 11. Reproduction

Install the declared environment, run `MPLBACKEND=Agg pytest -q`, execute `bash scripts/run_ablation.sh`, validate `results/current_component_ablations/manifest.json`, then run `python scripts/analyze_component_ablations.py` and `python scripts/build_component_ablation_paper.py`. Every number in the fresh tables is read from the validated manifest. The retained evidence hashes are:

- component analysis: `{hashes['analysis']}`;
- complete manifest: `{hashes['manifest']}`;
- historical mini-suite context: `{hashes['mini_suite']}`.

Reproduction on another host remains same-protocol computational replication, not independent conceptual validation. A true independent replication should reconstruct the interventions from the written specification, use separately managed code, and retain adverse outcomes.

## 12. Conclusion

FIM’s current evidence supports a disciplined component question, not a blanket architecture claim. The fresh matrix makes storage, retrieval, and salience gating separately falsifiable; preserves every seed and adverse removal; and binds each result to source and artifact hashes. The older delayed-recall comparison remains visible and places {best_mini['model']} ahead of FIM. Across the fresh comparisons, the full system directionally wins {full_better}, removals win {reduced_better}, and exact ties account for {zero}. These bounded results determine which mechanisms merit a larger preregistered study. They do not establish general superiority, real-world transfer, or state of the art.

## References

- Lorenz, E. N. (1996). Predictability: A problem partly solved. Seminar on Predictability, ECMWF.
- Lu, L., Jin, P., Pang, G., Zhang, Z., and Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*, 3, 218–229.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Gu, A. and Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *COLM*.
"""


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    mini = json.loads(MINI_SUITE.read_text(encoding="utf-8"))
    manifest_path = ROOT / analysis["manifest"]
    hashes = {
        "analysis": _sha256(ANALYSIS),
        "manifest": _sha256(manifest_path),
        "mini_suite": _sha256(MINI_SUITE),
    }
    text = manuscript(analysis, mini, hashes)
    if len(text.split()) < 2_000:
        raise RuntimeError("generated evidence paper is unexpectedly short")
    forbidden = ("TODO", "TBD", "placeholder", "results pending")
    if any(marker.lower() in text.lower() for marker in forbidden):
        raise RuntimeError("generated evidence paper contains a forbidden marker")
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"WROTE {OUTPUT} words={len(text.split())} sha256={hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
