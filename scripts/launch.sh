#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-config/default.yaml}

accelerate launch \
  train.py \
  --config "$CONFIG_PATH"

