#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?usage: bash scripts/reconstruction_modal.sh modal_checkpoint [data_yaml]}"
DATA_CFG="${2:-$REPO_ROOT/cfg/data/custom_all.yaml}"
CHECKPOINT="$(cd -- "$(dirname -- "$CHECKPOINT")" && pwd)/$(basename -- "$CHECKPOINT")"
DATA_CFG="$(cd -- "$(dirname -- "$DATA_CFG")" && pwd)/$(basename -- "$DATA_CFG")"
cd "$REPO_ROOT"

drumblender test -c "$REPO_ROOT/cfg/06_modal.yaml" \
  --ckpt_path "$CHECKPOINT" \
  --trainer.logger CSVLogger --trainer.logger.name modal_all \
  --model.test_metrics "$REPO_ROOT/cfg/metrics/drumblender_metrics.yaml" \
  --data "$DATA_CFG"
