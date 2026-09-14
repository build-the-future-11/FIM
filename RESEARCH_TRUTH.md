# Research Truth

This document records what the current public repository artifacts support and what they do not yet support.

## Evidence currently present

- A delayed-recall mini-suite comparing FIM with DeepONet, a selective SSM, a shape-aware MLP, and a Transformer.
- Persisted checkpoints, resolved configs, run logs, metrics, and generated figures for the stored mini-suite runs.
- A `paper_reference_results.json` file whose own metadata identifies its values as paper-reported compact results from `paper/Final FIM.pdf`.
- Core and epistemic test code covering dynamics, memory, benchmark generation, fractional dynamics, Levy sampling, rollout stability, and training execution.
- Maintained train/eval entrypoints repaired and protected by CI.
- A current-code component-ablation system with explicit `full`, `no_memory`, `no_retrieval`, and `no_salience_gating` variants, semantic tests, a multi-seed runner, provenance manifest, and frozen interpretation protocol.

## What the stored mini-suite supports

For the persisted delayed-recall mini-suite, the stored rollout MSE values are:

- DeepONet: 0.1579457372
- FIM: 0.1975817829
- Shape-aware MLP: 0.1965747178
- Transformer: 0.2490188628
- Selective SSM: 0.3046883643

On this artifact, FIM is competitive but is not the best model by rollout MSE. DeepONet is best among the stored runs, and the shape-aware MLP is marginally lower than FIM.

## Current execution-integrity status

The maintained repository execution paths have been repaired and verified. The canonical training path now routes through the maintained experiment implementation, checkpoint evaluation supports the current structured model outputs, and regression tests cover the repaired evaluator behavior. The current local suite passes **37 tests with 8 explicit skips and 0 failures**.

The legacy component-ablation wrapper that depended on unsupported flags was not treated as evidence. It has been replaced by a current-code ablation stack whose switches correspond to mechanisms that actually exist in `FIMSystem`:

- `full`;
- `no_memory`;
- `no_retrieval`;
- `no_salience_gating`.

The historical `no_spectral_mixing` label is intentionally unsupported in the current stack and must not be silently mapped to an unrelated mechanism.

A frozen 24-cell execution matrix was completed on 2026-09-02 as **2 benchmarks × 3 seeds × 4 current-code variants**. The runner atomically checkpointed each cell and bound it to source identity `e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`, switch state, configuration, metrics, raw rollout result, checkpoint, and log hashes. After an MPS allocation deadlock occurred before seed-37 produced any artifact, the process was terminated and resumed from the 20-cell atomic manifest; all four remaining cells then completed on MPS under the unchanged source identity and protocol. The final manifest contains 24 unique cells and passes byte-hash validation.

## Fresh current-component result

The fresh evidence does **not** support a general benefit from the implemented memory fabric:

- delayed recall: removing memory changes mean rollout MSE by -0.000088 (reduced minus full), removing retrieval by -0.000088, and removing salience gating by -0.000106; all mean directions favor removal and none is confirmatory;
- Lorenz-96: removing memory or retrieval changes mean rollout MSE by only +0.000080 with seed ranges spanning both signs; removing salience gating changes the mean by +0.040930, but two of three individual seeds favor removal and one adverse seed drives the mean;
- `no_memory` and `no_retrieval` are numerically identical for every seed in both benchmarks, consistent with stored traces being behaviorally inert when retrieval is disabled;
- across six benchmark-by-removal comparisons, mean direction favors the full model in three and removal in three; every exact sign-test p-value is at least 0.5;
- the retained mini-suite still places DeepONet first (rollout MSE 0.157946) and FIM third (0.197582).

The defensible contribution is therefore a fail-closed component study and a negative/mechanism-boundary result. It is not evidence of state-of-the-art forecasting or universal memory value.

## Historical paper-reference boundary

`paper_reference_results.json` is explicitly labeled as a compact transcription of values reported in the existing PDF. It is not, by itself, proof that those values were freshly reproduced from the current commit/configuration.

The stored Lorenz ablation values in that reference file are:

- full: 1.317169
- no_memory: 1.317169
- no_spectral_mixing: 0.869918
- no_retrieval: 1.317169

Those values do not support a blanket claim that the full model consistently outperforms its ablations. They also must not be reinterpreted as results from the new current-code ablation variants unless a retained current-commit run independently establishes the correspondence.

## Claims that should not currently be made from this public repository alone

- That FIM consistently outperforms all included baselines.
- That every paper-reported number has been freshly reproduced from the current commit.
- That every ablation favors the full model.
- That the historical `no_spectral_mixing` result is equivalent to any new current-code ablation.
- That the completed two-benchmark matrix establishes natural-domain transfer or state of the art.
- That three optimization seeds are independent domains or support a conventional p<0.05 mechanism claim.

## Next evidence gate

Before any stronger architecture claim, run a new preregistered study with oracle and random retrieval, retrieval-precision labels, parameter-matched controls, crossed delay/capacity/query-noise/distractor settings, and public natural sequence or operator-learning datasets. Strong baselines must be retrained on identical trajectories, seeds, budgets, and selection rules. The current negative result must remain immutable rather than being tuned away.

Negative, mixed, baseline-winning, and mechanism-falsifying outcomes are all valid scientific endpoints. Missing execution is not a null result, and historical paper-reference values are not fresh reproduction evidence.
