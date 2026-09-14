# FIM Current-Component Ablation Protocol — 2026-08-29

## Why this protocol exists

The legacy ablation wrapper was invalid for current code: it invoked a nonexistent root training entry point and passed flags that the maintained experiment CLI did not implement. This protocol replaces that dead path with explicit switches tied to mechanisms that are present in the maintained `FIMSystem` implementation.

No historical paper result is reclassified as newly reproduced by this change.

## Current-code variants

The fresh matrix supports exactly four variants:

- `full`: memory storage, retrieval feedback, and salience gating enabled;
- `no_memory`: memory storage and retrieval disabled;
- `no_retrieval`: traces may be stored, but retrieved memory is never fed back into the latent state;
- `no_salience_gating`: memory and retrieval remain enabled, but every produced trace is eligible for storage rather than requiring the salience threshold.

`no_spectral_mixing` is intentionally unsupported. The maintained `FIMSystem` does not contain a component with that semantic identity, so mapping the old label onto unrelated current code would create false evidence.

## Frozen default matrix

The default runner executes:

- benchmarks: `lorenz96`, `delayed_recall`;
- seeds: `11`, `23`, `37`;
- variants: all four above;
- epochs: `12`;
- batch size: `64`;
- dataset size: `2048`;
- training rollout steps: `4`;
- evaluation steps: `30`.

The exact command is:

```bash
bash scripts/run_ablation.sh
```

These defaults create a fresh current-code evidence matrix. They are not asserted to be equivalent to the historical paper configuration.

## Required evidence

For every benchmark × seed × variant cell, retain the maintained experiment runner's:

- resolved configuration;
- logs;
- metrics;
- checkpoints;
- figures where generated;
- top-level current-ablation manifest containing the Git commit when available,
  a deterministic SHA-256 of the executable source/protocol tree, and variant switches.

The runner atomically checkpoints the manifest after every completed cell and
resumes only when the source identity, frozen protocol, and retained artifact
paths match. An interrupted matrix therefore preserves completed work without
mixing source versions.

A missing or failed cell makes the matrix incomplete. It must not be silently dropped from aggregation.

## Semantic verification gate

Before paper-facing use, tests must establish that:

1. `no_memory` neither writes to nor reads from the adaptive memory bank;
2. `no_retrieval` can write memory but does not read it back into prediction;
3. `full` stores a trace and can retrieve it on a later step when storage is forced by the test threshold;
4. `no_salience_gating` stores traces even when the configured salience threshold would otherwise reject them;
5. unsupported historical labels fail closed.

These tests verify switch semantics. They do not establish a scientific benefit.

## Interpretation rules

The full model may win, tie, or lose. Report all variants and seeds.

- If `no_memory` matches or beats `full`, do not claim the current benchmark demonstrates a memory benefit.
- If `no_retrieval` matches or beats `full`, do not claim retrieved feedback is necessary under that benchmark/protocol.
- If `no_salience_gating` matches or beats `full`, do not claim salience gating is supported by the fresh matrix.
- A benefit on delayed recall does not automatically generalize to Lorenz96 or other dynamics tasks.
- A benefit on one seed is not a multi-seed effect.

## Relationship to historical paper-reference values

Historical labels and values remain historical reference artifacts until a separate equivalence audit proves that their code, data, hyperparameters, and mechanism semantics match a maintained reproducible implementation.

The fresh current-component matrix must be reported separately if such equivalence is not established.
