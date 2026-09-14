#!/usr/bin/env bash

set -euo pipefail

DEVICE=${DEVICE:-auto}
CHECKPOINT=${CHECKPOINT:-${MODEL_PATH:-}}
CONFIG=${CONFIG:-}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-${STEPS:-50}}
BATCH_SIZE=${BATCH_SIZE:-32}
OUTPUT_DIR=${OUTPUT_DIR:-}

if [[ -z "$CHECKPOINT" ]]; then
    cat >&2 <<'EOF'
ERROR: set CHECKPOINT to an existing experiment checkpoint.

Example:
  CHECKPOINT=results/my_run/lorenz96_fim_plus/checkpoints/final_state.pt \
  bash scripts/run_eval.sh

The old default `runs/latest/model.pt` was removed because the maintained
experiment runner does not create that path.
EOF
    exit 2
fi

args=(
    python scripts/evaluate_checkpoint.py
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --rollout_steps "$ROLLOUT_STEPS"
    --batch_size "$BATCH_SIZE"
)

if [[ -n "$CONFIG" ]]; then
    args+=(--config "$CONFIG")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    args+=(--output_dir "$OUTPUT_DIR")
fi

"${args[@]}"
