#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${FIM_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "FIM Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
export FIM_PYTHON="$PYTHON_BIN"

echo '[1/6] implementation and scientific-integrity tests'
MPLBACKEND=Agg "$PYTHON_BIN" -m pytest -q
echo '[2/6] frozen 24-cell current-component matrix'
bash scripts/run_ablation.sh
echo '[3/6] paired analysis'
"$PYTHON_BIN" scripts/analyze_component_ablations.py
echo '[4/6] evidence-derived manuscript'
"$PYTHON_BIN" scripts/build_component_ablation_paper.py
echo '[5/6] compiled publication artifact'
"$PYTHON_BIN" scripts/compile_component_ablation_paper.py
echo '[6/6] final manifest revalidation'
"$PYTHON_BIN" scripts/validate_component_ablation_manifest.py
echo 'FIM CURRENT-EVIDENCE FINALIZATION COMPLETE'
