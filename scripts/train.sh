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
DEFAULT_CONFIG="$PROJECT_DIR/configs/v0.7_config_qwen3.5_9b.yaml"

# Source .env if it exists (for WANDB_MODE, HF_TOKEN, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [ ! -d "$ULTRAVOX_DIR" ]; then
    echo "ERROR: ultravox-upstream not found. Run scripts/setup.sh first."
    exit 1
fi

# Resolve --config_path in user args to absolute paths (relative to original cwd).
# If user provides --config_path, use it instead of the default.
ORIG_CWD="$(pwd)"
CONFIG_PATH="$DEFAULT_CONFIG"
EXTRA_ARGS=()
SKIP_NEXT=false
for i in "$@"; do
    if $SKIP_NEXT; then
        # This is the value after --config_path
        if [[ "$i" != /* ]]; then
            CONFIG_PATH="$ORIG_CWD/$i"
        else
            CONFIG_PATH="$i"
        fi
        SKIP_NEXT=false
        continue
    fi
    if [[ "$i" == --config_path ]]; then
        SKIP_NEXT=true
        continue
    fi
    if [[ "$i" == --config_path=* ]]; then
        val="${i#--config_path=}"
        if [[ "$val" != /* ]]; then
            CONFIG_PATH="$ORIG_CWD/$val"
        else
            CONFIG_PATH="$val"
        fi
        continue
    fi
    EXTRA_ARGS+=("$i")
done

cd "$ULTRAVOX_DIR"

# Auto-detect GPUs (use poetry's python so torch is available)
NUM_GPUS="${NUM_GPUS:-$(poetry run python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "=== Ultravox v0.7 Qwen 3.5 9B Training ==="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG_PATH"
echo ""

# Run from ultravox-upstream (where poetry.lock lives) but add project root
# to PYTHONPATH so our patches and train.py are importable.
export PYTHONPATH="${PROJECT_DIR}:${ULTRAVOX_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training with torchrun ($NUM_GPUS GPUs)..."
    poetry run torchrun \
        --nproc_per_node="$NUM_GPUS" \
        "$PROJECT_DIR/train.py" \
        --config_path "$CONFIG_PATH" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "Launching single-GPU training..."
    poetry run python "$PROJECT_DIR/train.py" \
        --config_path "$CONFIG_PATH" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
