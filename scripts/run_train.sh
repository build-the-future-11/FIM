#!/usr/bin/env bash

set -euo pipefail

# Maintained thin wrapper around fim_experiments/main.py.
# This script intentionally does not move ad-hoc root-level model files;
# the experiment runner preserves checkpoints, resolved configs, logs and
# metrics under RESULTS_ROOT/EXP_NAME/<benchmark>_<model>/.

DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-3e-4}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
EVAL_STEPS=${EVAL_STEPS:-50}
DATASET_SIZE=${DATASET_SIZE:-4096}
SEED=${SEED:-42}
BENCHMARK=${BENCHMARK:-lorenz96}
MODEL=${MODEL:-fim_plus}
EXP_NAME=${EXP_NAME:-train_$(date +%Y%m%d_%H%M%S)}
RESULTS_ROOT=${RESULTS_ROOT:-results}

if [[ "${NUM_GPUS:-1}" -gt 1 ]]; then
    echo "ERROR: fim_experiments/main.py does not implement distributed training." >&2
    echo "Run one device per process or add a verified distributed runner before setting NUM_GPUS>1." >&2
    exit 2
fi

echo "=============================="
echo "FIM training"
echo "experiment : $EXP_NAME"
echo "benchmark  : $BENCHMARK"
echo "model      : $MODEL"
echo "device     : $DEVICE"
echo "seed       : $SEED"
echo "results    : $RESULTS_ROOT"
echo "=============================="

python fim_experiments/main.py \
    --exp_name "$EXP_NAME" \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --dataset_size "$DATASET_SIZE" \
    --lr "$LR" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --seed "$SEED" \
    --results_root "$RESULTS_ROOT"

echo "Training/evaluation completed through the maintained experiment runner."
echo "Inspect $RESULTS_ROOT/$EXP_NAME/ for resolved configs, logs, metrics and checkpoints."
