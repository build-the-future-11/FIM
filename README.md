# Fabric-Induced Memory (FIM)

**Publication:** Internal archival package only. External venue submit not authorized.


FIM is a research framework for neural sequence models with a structured latent memory fabric rather than only a larger context window or a single compressed hidden state.

The implementation explores local information propagation, persistent traces, decay, salience, retrieval, latent geometry, stochastic dynamics, and long-horizon forecasting.

## Evidence boundary

Read [`RESEARCH_TRUTH.md`](RESEARCH_TRUTH.md) before quoting results.

The retained 24-cell matrix is **retired historical shared-minibatch evidence**. Current-status wording was reconciled on 19 November 2026; see the [source-bound correction](research/CURRENT_EVIDENCE_RECONCILIATION_20261119.md). Its byte-hash validation does not establish trajectory isolation or training through the delayed-recall event. Read the [historical evidence boundary](research/HISTORICAL_EVIDENCE_BOUNDARY_20260930.md) before making mechanism claims.

The public repository contains real source code, tests, experiment runners, stored checkpoints/logs/configs, a delayed-recall mini-suite, historical paper-reference values, and an existing PDF manuscript. These artifacts are not all equivalent forms of evidence.

In particular:

- `results/mini_processed/mini_suite_summary.md` is a persisted compact delayed-recall comparison. In that stored artifact, FIM is competitive but does **not** have the lowest rollout MSE; DeepONet does.
- `paper_reference_results.json` explicitly identifies its values as compact results transcribed from the existing paper. Treat those as historical paper-reference values unless they are freshly reproduced from the current commit/configuration.
- The repository must not be described as showing that the full FIM model consistently beats every baseline or ablation. The committed artifacts do not support that blanket statement.

Negative results and baseline wins are part of the scientific record and should be preserved.

This directory is a historical evidence/manuscript line, not an independent replication.
See [Canonical lineage](CANONICAL_LINEAGE.md) and use the sibling
`Fabric-Induced-Memory` repository for new frozen campaigns.

## Historical implementation entry points

These commands document the archived implementation. They do not execute the corrected canonical 40-cell protocol or establish new mechanism evidence. The retired 24-cell campaign must not be restarted to improve its result.

The historical software setup is:

```bash
python -m pip install -e . pytest
python -m compileall -q fim fim_experiments scripts tests
MPLBACKEND=Agg pytest -q
```

Run one maintained training/evaluation experiment:

```bash
DEVICE=auto \
BENCHMARK=lorenz96 \
MODEL=fim_plus \
EXP_NAME=lorenz_fim_seed42 \
SEED=42 \
bash scripts/run_train.sh
```

The experiment runner preserves resolved configuration, logs, metrics, figures and checkpoints under the selected results root. It does not rely on an ad-hoc root-level `model.pt`.

Evaluate an existing checkpoint without retraining:

```bash
CHECKPOINT=results/lorenz_fim_seed42/lorenz96_fim_plus/checkpoints/final_state.pt \
DEVICE=auto \
bash scripts/run_eval.sh
```

Run the supported model-comparison suite:

```bash
python scripts/run_full_suite.py --device auto
```

### Component-ablation warning

The historical `scripts/run_ablation.sh` used flags that were not connected to the maintained experiment CLI. It was replaced rather than allowed to produce misleading paper evidence. Current component ablations use explicit tested switches with frozen semantics.

The historical matrix tooling has artifact-completeness checks. Those checks do not validate trajectory isolation, training horizon or submission readiness. For inspection of already retained artifacts, the documented tools are:

```bash
python scripts/validate_component_ablation_manifest.py
```

`scripts/run_ablation.sh` and `scripts/finalize_current_evidence.sh` remain preserved historical source. They are not a current scientific execution recommendation; do not run them to relabel or extend the retired matrix.

The historical analysis and paper-building scripts are also retained. Any reconstruction must use a disposable copy of its recorded source and preserve the archived outputs; rebuilding them does not validate the retired experiment or create corrected evidence.

The historical runner atomically checkpointed every completed cell. Its reuse checks require the same
frozen protocol, deterministic source-tree identity, and matching hashes for
the summary, raw rollout result, resolved config, checkpoint, and log. The
analysis, manuscript, and PDF commands refuse an incomplete matrix.

## Repository map

- `fim/` — core dynamics, geometry, memory, operators, stochastic modules, training, and utilities.
- `fim_experiments/` — benchmark systems, training/evaluation runners, and experiment orchestration.
- `tests/` and `fim/epistemic_tests_full/tests/` — implementation and scientific sanity checks.
- `scripts/` — maintained train/evaluation/suite entrypoints plus paper/research utilities.
- `results/` — stored run artifacts and processed summaries.
- `docs/` — architecture, mathematics, and evidence-bounded experiment documentation.
- `paper/` — existing manuscript artifact.
- `RESEARCH_TRUTH.md` — current evidence and claim boundary.

## Reproducibility rule

For new paper evidence, preserve the Git commit, dirty-tree state, resolved configuration, seed, benchmark/data identity, raw metrics/predictions, logs/checkpoints, and failure state. Paper tables and figures should be regenerated from those retained artifacts rather than manually copied from historical summaries.

## Current status

**HISTORICAL ARCHIVE / CURRENT MECHANISM EVIDENCE INCOMPLETE.** The retained mixed/adverse results remain unchanged. They cannot answer the corrected trajectory-isolated question. No completed canonical 40-cell execution was verified in the checked source/main Actions records, and the preserved paper/PDF are not a refreshed submission-ready package.
# FIM
