#!/usr/bin/env bash
set -euo pipefail
python3 main.py --benchmark lorenz --model fim --epochs 40 --batch_size 128 --hidden 64 --trace_dim 32 --dataset_size 4096 --lr 3e-4 --device auto --exp_name lorenz_main
