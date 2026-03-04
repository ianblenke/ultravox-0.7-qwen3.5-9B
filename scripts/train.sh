#!/usr/bin/env bash
# Training launch script for Ultravox v0.7 Qwen 3.5 9B
#
# Usage:
#   bash scripts/train.sh                    # Default: auto-detect GPUs
#   bash scripts/train.sh --max_steps 100    # Override any training param
#   NUM_GPUS=4 bash scripts/train.sh         # Specify GPU count
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ULTRAVOX_DIR="$PROJECT_DIR/ultravox-upstream"
CONFIG_PATH="$PROJECT_DIR/configs/v0.7_config_qwen3.5_9b.yaml"

# Auto-detect GPUs
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "=== Ultravox v0.7 Qwen 3.5 9B Training ==="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG_PATH"
echo ""

cd "$ULTRAVOX_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training with torchrun ($NUM_GPUS GPUs)..."
    poetry run torchrun \
        --nproc_per_node="$NUM_GPUS" \
        -m ultravox.training.train \
        --config_path "$CONFIG_PATH" \
        "$@"
else
    echo "Launching single-GPU training..."
    poetry run python \
        -m ultravox.training.train \
        --config_path "$CONFIG_PATH" \
        "$@"
fi
