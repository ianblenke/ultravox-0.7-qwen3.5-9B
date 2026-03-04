# Ultravox v0.7 — Qwen 3.5 9B

Finetune [Ultravox v0.7](https://github.com/fixie-ai/ultravox) with [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B) as the LLM backbone.

Ultravox is a multimodal speech-language model that processes audio directly into LLM embedding space, bypassing traditional ASR pipelines. The official v0.7 uses GLM-4.6 (0.7B). This project replaces it with Qwen 3.5 9B for significantly stronger reasoning, 262K context, and multilingual support (201 languages).

## Architecture

```
Audio ──► Whisper-large-v3-turbo ──► UltravoxProjector ──► Qwen 3.5 9B ──► Text
          (encoder, trainable)       (1280 → 4096)        (frozen, 9.65B)
```

| Component | Model | Parameters | Training |
|-----------|-------|------------|----------|
| Audio Encoder | whisper-large-v3-turbo | ~800M | LoRA (r=8) |
| Projector | UltravoxProjector (SwiGLU + RMSNorm) | ~50M | Full |
| LLM Backbone | Qwen 3.5 9B | 9.65B | Frozen |

**Training method**: Knowledge distillation — the audio pathway is trained to match the frozen Qwen 3.5 9B's text-only logits.

## Qwen 3.5 9B Specifics

Qwen 3.5 9B (released 2026-03-02) has a novel hybrid architecture:
- **24 Gated DeltaNet (linear attention) + 8 standard GQA layers**
- 262K native context (1M+ with YaRN)
- **Multimodal model** (vision+text) — we use only the text portion
- **bf16 only** — no QLoRA/4-bit (DeltaNet layers have high quantization error)
- Thinking mode (`<think>`/`</think>`) disabled for real-time speech

A compatibility patch (`patches/qwen3_5_support.py`) handles loading the text-only `Qwen3_5ForCausalLM` from the multimodal model weights.

## Quick Start

```bash
# 1. Setup
bash scripts/setup.sh

# 2. Validate compatibility (no GPU needed for config-only)
python scripts/validate_model.py

# 3. Quick smoke test (200 steps)
bash scripts/train.sh --config_path configs/v0.7_config_qwen3.5_9b_quick.yaml

# 4. Full training
bash scripts/train.sh

# 5. Evaluate
bash scripts/eval.sh <checkpoint_path>

# 6. Inference
python scripts/infer.py <checkpoint_path> --audio audio.wav

# 7. Export to HuggingFace
python scripts/export_hf.py <checkpoint_path> ./output --push-to-hub --hub-repo user/model
```

## Project Structure

```
├── configs/
│   ├── v0.7_config_qwen3.5_9b.yaml       # Full training (multilingual, ~250 datasets)
│   ├── v0.7_config_qwen3.5_9b_quick.yaml  # 200-step smoke test
│   └── generation_config.json              # Suppress thinking tokens
├── patches/
│   └── qwen3_5_support.py                 # Multimodal config + model loading patch
├── scripts/
│   ├── setup.sh            # Environment setup
│   ├── train.sh            # Training launcher (auto-detects GPUs)
│   ├── eval.sh             # Benchmark evaluation
│   ├── infer.py            # Single-file inference
│   ├── validate_model.py   # Compatibility validation
│   └── export_hf.py        # HuggingFace export + model card
├── train.py                # Training entry point (applies patch)
├── Dockerfile              # Reproducible CUDA 12.4 environment
└── openspec/               # Spec-first documentation
```

## GPU Requirements

| Mode | VRAM | GPUs |
|------|------|------|
| Smoke test (batch=4) | ~40GB | 1x A100/H100 |
| Full training (batch=8, FSDP) | ~40GB/GPU | 4-8x A100/H100 |
| Inference | ~20GB | 1x A100/RTX 4090 |

## Docker

```bash
docker build -t ultravox-qwen35 .
docker run --gpus all -v ~/.cache/huggingface:/workspace/.cache/huggingface ultravox-qwen35
```

## License

- Qwen 3.5 9B: Apache 2.0
- Ultravox: MIT
- This project: MIT
