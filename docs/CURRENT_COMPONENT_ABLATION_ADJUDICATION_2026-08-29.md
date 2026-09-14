# Current-code FIM component ablation adjudication — 2026-08-29

## Status

**Verdict: mixed / no broad component-superiority claim established.**

This report adjudicates the fresh retained 24-cell current-code ablation run requested in PR #4. It does not reproduce, replace, or relabel historical paper-reference results.

## Provenance

- GitHub Actions run: `33257399564` (`Current FIM component ablations`)
- executed checkout SHA recorded by the retained manifest: `d4fdca0189f071ba4ce38f874301ae8be8e586ba`
- source trigger PR head: `f8455ca2cb15700c09b17745d4687246364a6639`
- retained artifact: `fim-current-component-ablations-d4fdca0189f071ba4ce38f874301ae8be8e586ba`
- artifact ID: `9716689517`
- artifact ZIP SHA-256: `81bb5c79f751cfc9fea9b4cf6e90879e1b7fbe81b48629657bdc30b8d492fd13`
- retained manifest: `results/current_component_ablations/manifest.json`
- manifest creation time: `2026-08-29T14:58:50.451395+00:00`
- matrix: 2 benchmarks × 3 seeds × 4 variants = **24/24 complete cells**
- benchmarks: `lorenz96`, `delayed_recall`
- seeds: `11`, `23`, `37`
- variants: `full`, `no_memory`, `no_retrieval`, `no_salience_gating`
- training budget recorded per cell: 12 epochs, batch size 64, dataset size 2048, train rollout 4, eval rollout 30

The retained manifest explicitly limits these data to fresh current-code component ablations and says they do not reproduce or replace historical paper-reference labels without a separate equivalence audit.

## Interpretation rule

The manifest exposes `rollout_mse` and `rollout_mae`. Lower is better for both. No single primary metric is declared inside the manifest, so both are reported and neither is selectively promoted.

For paired comparisons below:

`delta = full - comparator`

- negative delta: full is better;
- positive delta: comparator is better;
- zero: tie.

Only three paired seeds exist per benchmark. Accordingly, this report gives per-seed deltas, means, and sample standard deviations, but **does not use tiny-n significance tests to manufacture confidence**.

## Aggregate retained metrics

| benchmark | variant | rollout MSE mean | MSE SD | rollout MAE mean | MAE SD |
|---|---|---:|---:|---:|---:|
| delayed_recall | full | 0.838792 | 0.006885 | 0.686347 | 0.004896 |
| delayed_recall | no_memory | 0.838947 | 0.006933 | 0.686396 | 0.004831 |
| delayed_recall | no_retrieval | 0.838947 | 0.006933 | 0.686396 | 0.004831 |
| delayed_recall | no_salience_gating | 0.838819 | 0.006840 | 0.686347 | 0.004895 |
| lorenz96 | full | 0.533549 | 0.076207 | 0.360859 | 0.027175 |
| lorenz96 | no_memory | 0.511963 | 0.067265 | 0.350220 | 0.016196 |
| lorenz96 | no_retrieval | 0.511963 | 0.067265 | 0.350220 | 0.016196 |
| lorenz96 | no_salience_gating | 0.521150 | 0.068244 | 0.360342 | 0.024291 |

## Paired deltas against `full`

| benchmark | metric | comparator | seed 11 | seed 23 | seed 37 | mean delta | delta SD | full wins / ties / losses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lorenz96 | MSE | no_memory | +0.011729 | -0.011837 | +0.064864 | +0.021585 | 0.039289 | 1 / 0 / 2 |
| lorenz96 | MAE | no_memory | -0.002021 | -0.002427 | +0.036367 | +0.010639 | 0.022281 | 2 / 0 / 1 |
| lorenz96 | MSE | no_retrieval | +0.011729 | -0.011837 | +0.064864 | +0.021585 | 0.039289 | 1 / 0 / 2 |
| lorenz96 | MAE | no_retrieval | -0.002021 | -0.002427 | +0.036367 | +0.010639 | 0.022281 | 2 / 0 / 1 |
| lorenz96 | MSE | no_salience_gating | +0.007156 | -0.006842 | +0.036882 | +0.012399 | 0.022328 | 1 / 0 / 2 |
| lorenz96 | MAE | no_salience_gating | -0.016999 | -0.001977 | +0.020528 | +0.000517 | 0.018887 | 2 / 0 / 1 |
| delayed_recall | MSE | no_memory | -0.000008 | -0.000081 | -0.000373 | -0.000154 | 0.000193 | 3 / 0 / 0 |
| delayed_recall | MAE | no_memory | -0.000013 | -0.000003 | -0.000130 | -0.000049 | 0.000071 | 3 / 0 / 0 |
| delayed_recall | MSE | no_retrieval | -0.000008 | -0.000081 | -0.000373 | -0.000154 | 0.000193 | 3 / 0 / 0 |
| delayed_recall | MAE | no_retrieval | -0.000013 | -0.000003 | -0.000130 | -0.000049 | 0.000071 | 3 / 0 / 0 |
| delayed_recall | MSE | no_salience_gating | 0.000000 | -0.000079 | -0.000002 | -0.000027 | 0.000045 | 2 / 1 / 0 |
| delayed_recall | MAE | no_salience_gating | 0.000000 | -0.000001 | -0.000001 | -0.000001 | 0.000001 | 2 / 1 / 0 |

## Benchmark-specific adjudication

### Lorenz-96

The full model does **not** establish superiority over the ablations on this run.

Against `no_memory` / `no_retrieval`, the full model has a **higher** mean rollout MSE by `0.021585` (about 4.22% relative to the comparator mean) and a higher mean rollout MAE by `0.010639` (about 3.04%). The seed-level direction is mixed, with seed 37 strongly adverse to the full condition and enough to reverse the aggregate.

Against `no_salience_gating`, the full model also has a higher mean MSE by `0.012399` (about 2.38%), while mean MAE is almost tied but slightly adverse (`+0.000517`).

**Classification: mixed/adverse.** These cells are incompatible with a blanket statement that the enabled FIM components consistently improve Lorenz-96 rollout error.

### Delayed recall

The full model is numerically lower than `no_memory` / `no_retrieval` on both metrics for all three seeds, but the differences are extremely small:

- mean MSE delta: `-0.000154`, about **0.0184%** relative to the comparator mean;
- mean MAE delta: `-0.000049`, about **0.0071%** relative to the comparator mean.

The `no_salience_gating` comparison is even closer to zero: mean MSE delta `-0.000027`, mean MAE delta `-0.000001`, with one exact tie at seed 11.

**Classification: near-null / weakly favorable numerically, not established as a meaningful component effect.** With n=3 and effect sizes this small, this run alone does not justify a broad causal or practical-improvement claim.

## Identifiability observation

For every retained seed in both benchmarks, `no_memory` and `no_retrieval` have exactly identical `rollout_mse` and `rollout_mae` values at the precision stored in the manifest.

That means this 24-cell execution does **not** separately identify a performance effect of "memory enabled but retrieval disabled" from the `no_memory` condition on the reported metrics. This is an evidence limitation, not by itself proof of an implementation defect.

## Allowed claims after this adjudication

Supported:

- a complete fresh current-code 24-cell component-ablation run exists with retained provenance;
- Lorenz-96 evidence is mixed/adverse to broad component-superiority claims;
- delayed-recall differences are near-null and only weakly favorable numerically to the full condition;
- the two benchmarks do not support one uniform mechanism story;
- `no_memory` and `no_retrieval` are observationally identical on the retained MSE/MAE outputs in this matrix.

Not supported:

- "the full model consistently outperforms its ablations";
- "memory/retrieval/salience gating is necessary";
- statistical significance from three seeds;
- historical-paper equivalence;
- cross-benchmark or general component superiority.

## Next evidence gate

If this mechanism is pursued further, the next run should be frozen **before** seeing outcomes and should focus on identifiability rather than adding favorable metrics: distinguish `no_memory` from `no_retrieval`, increase paired seeds, predeclare a primary metric and minimum meaningful effect size, and retain benchmark-specific conclusions even if they disagree.
