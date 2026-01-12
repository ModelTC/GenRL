#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=18000
CONFIG_PATH=${1:-config/default.yaml}

accelerate launch \
  train.py \
  --config "$CONFIG_PATH"


python ../../../data_lm_data_afs/occupy_gpu.py

