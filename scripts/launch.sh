#!/usr/bin/env bash
set -e

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
export XDG_CACHE_HOME=/data/.cache
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=18000
CONFIG_PATH=config/default.yaml


NODE_RANK=${NODE_RANK:-0}
RDZV_ID=${RDZV_ID:-5235}

torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=${RDZV_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config "$CONFIG_PATH"

