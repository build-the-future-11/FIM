# Project Status

## Current objective

Publish an evidence-bounded component study of the maintained FIM implementation without reviving unsupported historical mechanism labels.

## Working

- maintained train/evaluation paths;
- semantic intervention tests;
- frozen 24-cell component matrix;
- atomic resume and artifact hashes;
- paired descriptive analysis;
- evidence-derived seven-page preprint and checksum sidecar.

## Implemented this run

- completed all 24 benchmark × seed × variant cells;
- recovered one pre-artifact MPS allocator deadlock from the verified 20-cell manifest;
- computed all six reduced-minus-full paired comparisons and exact sign tests;
- deepened related work and clarified the stable-direction failure;
- compiled and visually inspected the final seven-page PDF.

## Tested

- 42 passed, 7 explicit hardware/optional-path skips, 0 failures on 2026-09-14;
- manifest validator: 24 unique cells, one frozen source identity/protocol;
- five-entry SHA-256 sidecar: all checks pass;
- PDF: 7 Letter pages, no observed clipping or overlap.

## Evidence boundary

- the current paper is a rigorous bounded negative result, not a natural-domain or state-of-the-art study;
- the historical mini-suite is contextual evidence and was not pooled with the fresh matrix.

## Negative findings

- no retained experiment demonstrates general necessity of memory, retrieval, or salience gating;
- the historical `no_spectral_mixing` label has no valid current-code mapping and is permanently
  excluded from current-code interpretation; maintained runners reject it before training.

## Conditions for a separate successor study

- stronger claims require new public-domain tasks, stronger matched baselines, and a preregistered mechanism-recovery design.

## Metrics / experimental evidence

- 24/24 frozen cells;
- full model mean-direction wins: 3/6 comparisons;
- removal mean-direction wins: 3/6 comparisons;
- smallest exact sign p-value: 0.5;
- DeepONet mini-suite rollout MSE 0.157946 vs FIM 0.197582.

## Highest-value next actions

1. Freeze an oracle/random-retrieval factorial protocol.
2. Add independent public datasets and dataset-clustered inference.
3. Retrain DeepONet, MLP, SSM, and Transformer controls under identical budgets.

## Reproduction commands

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q
bash scripts/run_ablation.sh
.venv/bin/python scripts/analyze_component_ablations.py
.venv/bin/python scripts/validate_component_ablation_manifest.py
shasum -a 256 -c output/fim-evidence.sha256
```
