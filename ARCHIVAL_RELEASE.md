# Archival release classification

## Current interpretation — 19 November 2026

The immutable historical artifacts remain available for inspection. Their
shared-minibatch mixed/adverse outcome must retain its documented limitations;
it is not a validated trajectory-isolated mechanism result. The corrected
canonical study remains separate, with completed execution not verified in the
checked main source/Actions records. The historical release wording below does
not establish a refreshed scientific release or conference readiness.

See the [source-bound reconciliation](research/CURRENT_EVIDENCE_RECONCILIATION_20261119.md).

## Historical status text (superseded interpretation)

This repository is the immutable historical evidence and manuscript line for the completed 24-cell
FIM component study. New research campaigns belong in the maintained sibling
`../Fabric-Induced-Memory` and must not overwrite or pool results from this source lineage.

### Verified release contents

- 24 unique current-code component cells across two benchmarks, three seeds, and four variants;
- six paired comparisons with exact small-sample sign tests;
- seven-page evidence-bounded preprint;
- five-entry SHA-256 evidence sidecar;
- 42 passing tests and 7 explicit hardware/optional-path skips on 2026-09-14;
- fail-closed rejection of unsupported historical ablation labels.

### Final conclusion

The evidence does not establish a general memory, retrieval, salience-gating, or baseline advantage.
That negative conclusion is the finished result. The archive is usable for exact inspection and
reproduction; it is not an active claim-development branch or independent replication.

### Verification

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q
shasum -a 256 -c output/fim-evidence.sha256
.venv/bin/python scripts/validate_component_ablation_manifest.py
```
