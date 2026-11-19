# Research Truth

This document records what the current public repository artifacts support and what they do not yet support.

## Current evidence qualification — 19 November 2026

The 24-cell record below is **retired shared-minibatch evidence**, not a current
trajectory-isolated mechanism study. Its mixed/adverse outcomes and artifact
hashes are preserved. The shared memory bank and four-step training horizon
prevent those cells from resolving the intended isolated-memory question.
The separately frozen canonical 40-cell protocol has no completed execution
verified in the main source/Actions records checked for this correction.
Read the [source-bound reconciliation](research/CURRENT_EVIDENCE_RECONCILIATION_20261119.md)
and the [historical evidence boundary](research/HISTORICAL_EVIDENCE_BOUNDARY_20260930.md)
before using any result or completion label below. No experiment was rerun.

## Retained implementation and artifacts

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

## Historical execution-integrity record

The earlier execution record reports repaired training/evaluation paths and a local suite of **37 tests with 8 explicit skips and 0 failures**. Those counts are retained historical receipts; they do not qualify the current checkout and were not rerun for this documentation correction. The recorded repairs route the training path through the maintained experiment implementation and support structured model outputs during checkpoint evaluation.

The legacy component-ablation wrapper that depended on unsupported flags was not treated as evidence. It has been replaced by a current-code ablation stack whose switches correspond to mechanisms that actually exist in `FIMSystem`:

- `full`;
- `no_memory`;
- `no_retrieval`;
- `no_salience_gating`.

The historical `no_spectral_mixing` label is intentionally unsupported in the current stack and must not be silently mapped to an unrelated mechanism.

The historical shared-minibatch 24-cell execution matrix was completed on 2026-09-02 as **2 benchmarks × 3 seeds × 4 current-code variants**. The runner atomically checkpointed each cell and bound it to source identity `e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`, switch state, configuration, metrics, raw rollout result, checkpoint, and log hashes. After an MPS allocation deadlock occurred before seed-37 produced any artifact, the process was terminated and resumed from the 20-cell atomic manifest; all four remaining cells then completed on MPS under the unchanged source identity and protocol. The final manifest contains 24 unique cells; its historical validation receipt authenticates the retained artifacts, not the corrected mechanism claim.

## Historical component outcomes (descriptive)

Under the retained historical implementation and budget, the observed comparisons did **not** establish a consistent memory benefit. Their design limitations prevent a trajectory-isolated mechanism conclusion:

- delayed recall: removing memory changes mean rollout MSE by -0.000088 (reduced minus full), removing retrieval by -0.000088, and removing salience gating by -0.000106; all mean directions favor removal and none is confirmatory;
- Lorenz-96: removing memory or retrieval changes mean rollout MSE by only +0.000080 with seed ranges spanning both signs; removing salience gating changes the mean by +0.040930, but two of three individual seeds favor removal and one adverse seed drives the mean;
- `no_memory` and `no_retrieval` have identical stored error statistics for every seed in both benchmarks; this does not by itself prove bitwise prediction identity, and the duplicate contrasts are not independent predictive evidence;
- across six benchmark-by-removal comparisons, mean direction favors the full model in three and removal in three; every exact sign-test p-value is at least 0.5;
- the retained mini-suite still places DeepONet first (rollout MSE 0.157946) and FIM third (0.197582).

The retained contribution is an archival record of the implementation, descriptive mixed/adverse outcomes and their identified limitations. A valid artifact manifest does not repair shared-trajectory retrieval or inadequate training horizon. This record establishes neither trajectory-isolated memory efficacy nor its absence, and it does not support state-of-the-art forecasting.

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

The maintained source already has a separately frozen [40-cell trajectory-isolated protocol](https://github.com/THE-BU1LD/Fabric-Induced-Memory/blob/e1ec0de48b915002418fde7625bdf96659023815/research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md). Its exact source, execution decision, retained hold, environment and durable output location must be resolved there before a scientific dispatch. This archive does not host that corrected runner. Preserve the closed historical mixed/adverse record and do not rerun its 24 cells for a favorable outcome. Broader architecture or natural-domain claims require separately justified data, strong matched baselines and discriminating retrieval controls after the corrected study; they cannot be borrowed from this archive.

Negative, mixed, baseline-winning, and mechanism-falsifying outcomes are all valid scientific endpoints. Missing execution is not a null result, and historical paper-reference values are not fresh reproduction evidence.
