# Project Context

## Purpose
Finetune Ultravox v0.7 (a multimodal speech-language model by Fixie AI) to use Qwen 3.5 9B as its LLM backbone. Ultravox processes audio directly into LLM embedding space via a trained multimodal projector, bypassing traditional ASR pipelines. This project creates a community variant pairing the Ultravox v0.7 architecture with Alibaba's newly released Qwen 3.5 9B model.

## Tech Stack
- Python 3.10+
- PyTorch (bf16 mixed precision)
- HuggingFace Transformers
- Ultravox training framework (from fixie-ai/ultravox)
- Audio encoder: OpenAI whisper-large-v3-turbo (encoder-only)
- LLM backbone: Qwen/Qwen3.5-9B (9.65B params, hybrid DeltaNet + GQA architecture)
- Distributed training: torchrun
- Config: SimpleParsing + YAML

## Project Conventions

### Code Style
- Follow upstream Ultravox conventions where possible
- Python: black formatting, type hints
- Config files: YAML with SimpleParsing dataclasses

### Architecture Patterns
- Three-component model: Audio Encoder → Multimodal Projector → LLM
- Knowledge distillation training: match frozen LLM text-only logits from audio input
- LLM backbone frozen during training; projector and optionally audio encoder are trained
- UltravoxProjector: StackAudioFrames → LayerNorm → Linear → SwiGLU → RMSNorm → Linear

### Testing Strategy
- Evaluate on standard speech benchmarks (LibriSpeech WER, Big Bench Audio, VoiceBench)
- Compare against Ultravox v0.7-GLM-4.6 baseline
- Validate audio-text alignment via knowledge distillation loss convergence

### Git Workflow
- Feature branches off main
- Spec-first: proposals approved before implementation
- Conventional commits

## Domain Context
- Ultravox v0.7 official release uses GLM-4.6 (0.7B) as backbone
- Previous versions supported Llama 3.x, Gemma 3, Qwen 3 32B
- Qwen 3.5 9B (released 2026-03-02) uses novel hybrid architecture: Gated DeltaNet (linear attention) + standard GQA, with 262K native context
- The projector must bridge whisper-large-v3-turbo output dimension to Qwen 3.5 9B hidden dimension (4096)
- Qwen 3.5 vocabulary is 248,320 tokens — must ensure no collision with `<|audio|>` placeholder token
- Qwen 3.5 has thinking mode (chain-of-thought) — should be disabled for real-time speech use cases

## Important Constraints
- Use bf16 precision only — Qwen 3.5's hybrid DeltaNet architecture has high quantization error with QLoRA/4-bit
- VRAM: ~22GB minimum for bf16 LoRA; full training requires multi-GPU (8x H100/B200 class)
- Audio encoder (whisper-large-v3-turbo) weights should be initialized from pretrained checkpoint
- Must be compatible with vLLM for production inference

## External Dependencies
- fixie-ai/ultravox GitHub repo (training framework)
- Qwen/Qwen3.5-9B on HuggingFace
- openai/whisper-large-v3-turbo on HuggingFace
- Standard speech evaluation datasets (LibriSpeech, etc.)
