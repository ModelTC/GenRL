#!/usr/bin/env bash
set -e

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
export XDG_CACHE_HOME=/data/.cache
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=18000
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG_PATH=config/longcat.yaml

accelerate launch \
  train.py \
  --config "$CONFIG_PATH"

