#!/usr/bin/env bash
set -e

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
export XDG_CACHE_HOME=/data/.cache
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=18000
CONFIG_PATH=config/longcat.yaml

NUM_MACHINES=2
MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"10.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

accelerate launch \
  --num_machines ${NUM_MACHINES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  train.py \
  --config "$CONFIG_PATH"

