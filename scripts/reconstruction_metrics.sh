#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:?usage: bash scripts/reconstruction_metrics.sh model_dir model_name [data_yaml]}"
MODEL_NAME="${2:?provide model_name without .yaml or .ckpt}"
DATA_CFG="${3:-$REPO_ROOT/cfg/data/custom_all.yaml}"
# Resolve caller-supplied paths before switching to the repository root.
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"
DATA_CFG="$(cd -- "$(dirname -- "$DATA_CFG")" && pwd)/$(basename -- "$DATA_CFG")"
cd "$REPO_ROOT"

drumblender test -c "$MODEL_DIR/$MODEL_NAME.yaml" \
  --ckpt_path "$MODEL_DIR/$MODEL_NAME.ckpt" \
  --trainer.logger CSVLogger --trainer.logger.name "${MODEL_NAME}_all" \
  --model.test_metrics "$REPO_ROOT/cfg/metrics/drumblender_metrics.yaml" \
  --data "$DATA_CFG"
