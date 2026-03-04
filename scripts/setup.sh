#!/usr/bin/env bash
# Setup script for Ultravox v0.7 Qwen 3.5 9B finetuning
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ULTRAVOX_DIR="$PROJECT_DIR/ultravox-upstream"

echo "=== Ultravox v0.7 Qwen 3.5 9B Setup ==="

# Check prerequisites
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "WARNING: CUDA not available. Training requires GPU with bf16 support."
fi

# Clone ultravox if not present
if [ ! -d "$ULTRAVOX_DIR" ]; then
    echo "Cloning fixie-ai/ultravox..."
    git clone https://github.com/fixie-ai/ultravox.git "$ULTRAVOX_DIR"
fi

# Install dependencies
cd "$ULTRAVOX_DIR"
if ! command -v poetry &>/dev/null; then
    echo "Installing poetry..."
    pip install poetry==1.7.1
fi

echo "Installing ultravox dependencies..."
poetry install

# Verify key model availability
echo ""
echo "=== Verifying model access ==="
python3 -c "
from transformers import AutoConfig
print('Checking Qwen/Qwen3.5-9B...')
config = AutoConfig.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
print(f'  hidden_size: {config.hidden_size}')
print(f'  vocab_size: {config.vocab_size}')
print(f'  num_hidden_layers: {config.num_hidden_layers}')
print('  OK')
" || echo "WARNING: Could not verify Qwen/Qwen3.5-9B model. Check HuggingFace access."

python3 -c "
from transformers import AutoConfig
print('Checking openai/whisper-large-v3-turbo...')
config = AutoConfig.from_pretrained('openai/whisper-large-v3-turbo')
print(f'  d_model: {config.d_model}')
print('  OK')
" || echo "WARNING: Could not verify whisper-large-v3-turbo model."

echo ""
echo "=== Setup complete ==="
echo "To start training:"
echo "  cd $PROJECT_DIR"
echo "  bash scripts/train.sh"
