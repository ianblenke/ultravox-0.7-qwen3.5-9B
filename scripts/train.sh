#!/usr/bin/env bash
# Training launch script for Ultravox v0.7 Qwen 3.5 9B
#
# Usage:
#   bash scripts/train.sh                    # Default: auto-detect GPUs, full config
#   bash scripts/train.sh --max_steps 100    # Override any training param
#   bash scripts/train.sh --config_path configs/v0.7_config_qwen3.5_9b_quick.yaml  # Quick smoke test
#   NUM_GPUS=4 bash scripts/train.sh         # Specify GPU count
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ULTRAVOX_DIR="$PROJECT_DIR/ultravox-upstream"
CONFIG_PATH="$PROJECT_DIR/configs/v0.7_config_qwen3.5_9b.yaml"

if [ ! -d "$ULTRAVOX_DIR" ]; then
    echo "ERROR: ultravox-upstream not found. Run scripts/setup.sh first."
    exit 1
fi

# Auto-detect GPUs
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "=== Ultravox v0.7 Qwen 3.5 9B Training ==="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG_PATH"
echo ""

# Run from ultravox-upstream (where poetry.lock lives) but add project root
# to PYTHONPATH so our patches and train.py are importable.
export PYTHONPATH="${PROJECT_DIR}:${ULTRAVOX_DIR}:${PYTHONPATH:-}"

cd "$ULTRAVOX_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training with torchrun ($NUM_GPUS GPUs)..."
    poetry run torchrun \
        --nproc_per_node="$NUM_GPUS" \
        "$PROJECT_DIR/train.py" \
        --config_path "$CONFIG_PATH" \
        "$@"
else
    echo "Launching single-GPU training..."
    poetry run python "$PROJECT_DIR/train.py" \
        --config_path "$CONFIG_PATH" \
        "$@"
fi
