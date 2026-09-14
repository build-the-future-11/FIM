# Research audit and execution checklist — 2026-09-13

## Verdict and classification

Core FIM modules, evaluation, data registry, component-ablation manifests/analysis, paper compiler, and tests are **complete/real**. Current component ablations are **real mixed/negative evidence**; `docs/experiments.md` correctly withdraws broad superiority. Smoke runs are **engineering fixtures**. General memory superiority, real-domain transfer, and a unique theoretical contribution are **unsupported/missing**. Duplicate implementations relative to `Fabric-Induced-Memory` and QFIM are **maintenance risk/dead divergence** until one source is canonical.

| Priority | WHAT / WHY | HOW / WHERE | VERIFY |
|---|---|---|---|
| P0 | Three copies can attach papers to different code/results. | Provenance and semantics can diverge. | Declare canonical implementation; record commit/file hashes for imported evidence. | Cross-repo manifest rejects nonidentical source. |
| P1 | Mixed ablations do not establish component benefit. | Selection across tasks/seeds invites cherry-picking. | Freeze primary estimand and matched replacements; retain adverse Lorenz96 evidence. | Hierarchical/per-task CIs and corrected family. |
| P1 | Modern sequence/memory baselines and real data are incomplete. | Relative novelty unknown. | Add matched SSM, long-context, retrieval and operator baselines; licensed datasets. | Equal-budget untouched evaluation. |
| P2 | Stability/units/discretization theory is incomplete. | Fractional dynamics may be numerically rather than mechanistically stable. | Derive parameter domains; convergence and boundary-condition tests. | Refinement curves and invariant violations reported. |
| P3 | Consolidate docs, caches, locks, CI. | Release drift. | One environment and paper gate. | Clean-clone reproduction. |

Conference state: **mixed evidence, not ready**.
