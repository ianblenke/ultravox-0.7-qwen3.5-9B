#!/usr/bin/env bash
# Evaluation script for Ultravox v0.7 Qwen 3.5 9B
#
# Usage:
#   bash scripts/eval.sh <checkpoint_path>
#   bash scripts/eval.sh runs/exp--2026-03-04--12-00-00/checkpoint-5000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ULTRAVOX_DIR="$PROJECT_DIR/ultravox-upstream"

CHECKPOINT="${1:?Usage: bash scripts/eval.sh <checkpoint_path>}"
shift  # Remove checkpoint from args so "$@" has remaining flags

if [ ! -d "$ULTRAVOX_DIR" ]; then
    echo "ERROR: ultravox-upstream not found. Run scripts/setup.sh first."
    exit 1
fi

echo "=== Ultravox v0.7 Qwen 3.5 9B Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

export PYTHONPATH="${PROJECT_DIR}:${ULTRAVOX_DIR}:${PYTHONPATH:-}"
cd "$ULTRAVOX_DIR"

poetry run python -m ultravox.evaluation.eval \
    --model_path "$CHECKPOINT" \
    --eval_sets \
        covost2-en-de \
        covost2-en-zh \
        covost2-es-en \
        covost2-zh-en \
        librispeech-clean-transcription \
        librispeech-other-transcription \
        commonvoice-en-transcription \
    --eval_batch_size 60 \
    --eval_max_tokens 512 \
    --eval_temperature 0.0 \
    "$@"
