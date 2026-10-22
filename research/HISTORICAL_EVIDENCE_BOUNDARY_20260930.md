# Historical component evidence boundary

The retained 24-cell matrix is byte-verifiable historical evidence, not a trajectory-isolated mechanism study. A successful manifest check authenticates its retained artifacts; it does not remove experimental confounding.

## Audited snapshot and executed checks

Snapshot: `build-the-future-11/FIM` commit `49af12e1d6151a0260b1b3e1c1422a56ecad84d5`.

On 2026-09-30, `python scripts/validate_component_ablation_manifest.py` passed against the retained artifacts. An independent call to `summarize_rows` regenerated the six paired descriptive comparisons. The smallest exact sign-test p-value remains 0.5. No neural fit, checkpoint evaluation, new scientific experiment, or outcome replacement was performed.

## Why the scope is historical

The retained manifest records batch size 64 and training rollout 4. The maintained sibling's prospectively frozen trajectory-isolated protocol documents two pre-existing limitations:

1. A model-wide memory bank can mix traces across independent examples in a minibatch. Batch size 1 with episode resets is required by that successor protocol.
2. The default delayed-recall event is at delay 8. A four-step training rollout never reaches the recall event.

The later protocol also aligns validation and evaluation memory semantics. Its 40-cell design is a separate evidence lane, not a reinterpretation of these 24 cells.

Source: [trajectory-isolated protocol at immutable sibling revision](https://github.com/THE-BU1LD/Fabric-Induced-Memory/blob/b6167d481df69fb0b783a06368060a0e15a75d21/research/protocols/FIM_TRAJECTORY_ISOLATED_CONFIRMATORY_V1.md).

## Permitted interpretation

Retain the observed mixed/adverse outcomes and all provenance. They show that this historical implementation and budget did not establish a consistent improvement from the tested switches. They cannot establish that correctly isolated trajectory memory is generally ineffective, nor that the intended long-delay retrieval mechanism was adequately trained.

`no_memory` and `no_retrieval` are mechanically different bank-state controls but prediction-equivalent when retrieval is disabled. Their duplicate outcomes must not be counted as independent predictive evidence. See the sibling's [semantics regression note](https://github.com/THE-BU1LD/Fabric-Induced-Memory/blob/b6167d481df69fb0b783a06368060a0e15a75d21/research/FIM_NO_MEMORY_VS_NO_RETRIEVAL_SEMANTICS_20260927.md).

The negative historical closeout remains closed. This clarification does not authorize rerunning until results improve or making a stronger null claim. Any separately frozen successor must preserve this record and declare its different protocol and implementation identity.
