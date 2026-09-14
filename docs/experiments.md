# experiments.md

## Experimental Setup

### Benchmarks

The codebase includes multiple dynamical-system experiment paths, including:

- Lorenz-family chaotic ODE tasks
- diffusion and wave PDE tasks
- fractional diffusion systems
- stochastic differential equations
- delayed-recall benchmark variants used by the stored mini-suite

The presence of an implemented benchmark does not mean every benchmark has been freshly reproduced from the current commit.

---

### Data Generation

Each dynamical benchmark generates trajectories

\[
\{x_0, x_1, \dots, x_T\}.
\]

Training pairs are constructed as

\[
(x_t, x_{t+1}).
\]

Evaluation uses autoregressive rollout where the configured benchmark supports it.

---

### Training Details

The repository contains AdamW-based training, configurable learning rates and batch sizes, scheduler support, gradient clipping, checkpointing, and seeded execution. The resolved configuration stored with an executed run is the authoritative source for that run; this document should not be used to infer a single universal hyperparameter setting for every historical artifact.

---

### Loss Function

The FIM training stack can combine prediction, memory, stability, retrieval, and consistency objectives. The exact active terms and weights are configuration-dependent and should be read from the resolved run configuration rather than assumed from a paper-level schematic.

---

### Metrics

Implemented and/or reported diagnostics include:

- mean squared error (MSE)
- long-horizon error growth
- rollout/final-step error
- distributional and spectral diagnostics
- drift/stability diagnostics

Only metrics present in persisted run artifacts should be quoted as executed evidence.

---

### Evaluation Protocol

Where applicable, autoregressive evaluation has the form

\[
\hat{x}_{t+1} = f(\hat{x}_t).
\]

Run-specific metrics, checkpoints, logs, and resolved configs are retained under the result directories for the stored mini-suite.

---

### Ablation Studies

The repository contains full-model and component-removal experiment paths, including memory, retrieval, spectral-mixing, capacity, and noise-related controls depending on the experiment runner/configuration.

Historical paper-reference values must not be treated as freshly reproduced evidence unless their provenance is re-established against the current source/configuration.

---

### Current Evidence Boundary

The previous version of this document stated that the full model "consistently outperforms ablations." The artifacts currently committed to the public repository do not support that blanket statement.

The stored delayed-recall mini-suite reports rollout MSE:

| Model | Rollout MSE |
|---|---:|
| DeepONet | 0.1579457372 |
| Shape-aware MLP | 0.1965747178 |
| FIM | 0.1975817829 |
| Transformer | 0.2490188628 |
| Selective SSM | 0.3046883643 |

On this stored artifact, FIM is competitive but is not the best method by rollout MSE.

Separately, `paper_reference_results.json` labels its contents as paper-reported compact results from the existing PDF. Its stored Lorenz ablation values are:

| Variant | Reference value |
|---|---:|
| full | 1.317169 |
| no_memory | 1.317169 |
| no_spectral_mixing | 0.869918 |
| no_retrieval | 1.317169 |

Those historical reference values also do not support a general claim that the complete model wins every ablation.

See `RESEARCH_TRUTH.md` for the repository-wide evidence boundary.

---

### Reproducibility

For new evidence, preserve at minimum:

- Git commit SHA and dirty-tree state
- resolved configuration
- seed
- dataset/benchmark identity
- raw metrics and predictions where applicable
- checkpoint/log paths
- failure state

Paper-reported reference values should remain clearly separated from freshly reproduced current-commit evidence. Negative results and baseline wins must be retained.
