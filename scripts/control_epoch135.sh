#!/usr/bin/env bash
# Explicit stages only: invoking this file without a stage never trains.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

stage="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${PYTHON:-$HOME/miniconda3/envs/drumblender/bin/python}"
CHECKPOINT="${CHECKPOINT:-$PWD/runs/joint_hybrid64_20261105_154300/joint/version_0/checkpoints/epoch=135-val_loss=0.981610.ckpt}"
DATA_DIR="${DATA_DIR:-$PWD/../dataset/modal_features/processed_modal_window_v2_64_2gpu}"
AUDIO_DIR="${AUDIO_DIR:-$PWD/../dataset/processed}"
NOISE_BANK="${NOISE_BANK:-$PWD/assets/noise_bank_48k.pt}"
CONTROL_ROOT="${CONTROL_ROOT:-$PWD/runs/control_epoch135_clean}"
DEVICE="${DEVICE:-cuda:0}"
common=(--checkpoint "$CHECKPOINT" --noise-bank "$NOISE_BANK"
        --data-dir "$DATA_DIR" --audio-dir "$AUDIO_DIR"
        --cache "$CONTROL_ROOT/cache" --device "$DEVICE")
paired=("${common[@]}" --warmup "$CONTROL_ROOT/warmup/best.pt"
        --output "$CONTROL_ROOT/paired" --minutes 120 --max-minutes 180)

case "$stage" in
  run)
    if [[ $# -ne 0 ]]; then
      echo 'The fresh run uses the fixed protocol; configure paths through environment variables.' >&2
      exit 2
    fi
    mkdir -p "$(dirname "$CONTROL_ROOT")"
    mkdir "$CONTROL_ROOT"
    (
      trap 'status=$?; printf "Pipeline exit=%s at %s\n" "$status" "$(date -Is)"' EXIT
      for phase in cache warmup train; do
        printf '\n=== %s | %s ===\n' "$phase" "$(date -Is)"
        bash scripts/control_epoch135.sh "$phase" || exit "$?"
      done
      printf '\nTraining complete. Audition and held-out evaluation have not been started.\n'
    ) 2>&1 | tee "$CONTROL_ROOT/console.log"
    ;;
  cache)
    "$PYTHON" -m scripts.control_joint cache "${common[@]}" "$@"
    ;;
  warmup)
    "$PYTHON" -m scripts.control_joint train --cache "$CONTROL_ROOT/cache" \
      --output "$CONTROL_ROOT/warmup" --device "$DEVICE" \
      --minutes 25 --max-minutes 30 "$@"
    ;;
  train)
    "$PYTHON" -m scripts.train_joint_control "${paired[@]}" --train-only "$@"
    ;;
  audition)
    "$PYTHON" -m scripts.train_joint_control "${paired[@]}" --render-only "$@"
    ;;
  test)
    if [[ $# -ne 2 ]]; then
      echo 'Provide the two atom IDs chosen from the NEW validation audition.' >&2
      exit 2
    fi
    "$PYTHON" -m scripts.evaluate_selected_controls "${common[@]}" \
      --run "$CONTROL_ROOT/paired" --output "$CONTROL_ROOT/selected" --atoms "$1" "$2"
    ;;
  analyze)
    "$PYTHON" -m scripts.analyze_selected_controls "$CONTROL_ROOT/selected" "$@"
    ;;
  help|-h|--help)
    printf '%s\n' 'run: fresh cache -> common warm-up -> paired training (no audition/test)' \
      'Stages: cache -> warmup -> train -> audition -> test ID1 ID2 -> analyze' \
      'Run on DSP_sjh only. No stage has been started by showing this help.' \
      'Warm-up + paired fit/validation: target 120 min, cap 180 min.'
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
