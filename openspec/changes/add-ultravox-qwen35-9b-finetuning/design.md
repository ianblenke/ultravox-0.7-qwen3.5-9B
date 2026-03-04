## Context
Ultravox v0.7 is a three-component multimodal speech model: audio encoder (Whisper) → multimodal projector (MLP adapter) → LLM backbone. The official v0.7 uses GLM-4.6 (0.7B). We are replacing the backbone with Qwen 3.5 9B to create a much more capable speech-language model.

Key stakeholders: open-source community, developers needing real-time speech AI with strong reasoning.

## Goals / Non-Goals
- Goals:
  - Train a production-quality Ultravox v0.7 variant with Qwen 3.5 9B backbone
  - Match or exceed Ultravox v0.7-GLM-4.6 on speech understanding benchmarks
  - Publish model weights to HuggingFace under permissive license
  - Ensure vLLM compatibility for serving
- Non-Goals:
  - Training the Qwen 3.5 9B backbone itself (it stays frozen)
  - Speech generation / TTS (Ultravox is speech understanding only)
  - Vision capabilities (Qwen 3.5's vision encoder is unused)
  - Multi-language speech training in initial release (English first)

## Decisions

### 1. LLM Backbone: Qwen 3.5 9B (frozen)
- **Decision**: Use `Qwen/Qwen3.5-9B` (instruction-tuned) as frozen backbone
- **Why**: 9.65B params with hybrid DeltaNet gives excellent quality/efficiency ratio, 262K context, Apache 2.0 license
- **Alternatives**: Qwen3-8B (dense, no DeltaNet efficiency), Llama 3.1 8B (less capable at reasoning), Gemma 3 9B (restricted license)

### 2. Audio Encoder: Whisper-large-v3-turbo (trainable)
- **Decision**: Initialize from pretrained whisper-large-v3-turbo, finetune during training (following Ultravox v0.5+ approach)
- **Why**: Finetuning the encoder alongside the projector yields better audio-text alignment
- **Alternatives**: Frozen encoder (faster training, slightly worse quality)

### 3. Training Precision: bf16 only
- **Decision**: Use bf16 mixed precision throughout. No QLoRA/4-bit quantization.
- **Why**: Qwen 3.5's Gated DeltaNet layers have high quantization error at 4-bit
- **Impact**: Requires ~40-80GB VRAM depending on batch size (multi-GPU required)

### 4. Projector Dimensions
- **Decision**: UltravoxProjector bridges Whisper output dim (1280 for large-v3-turbo) → Qwen 3.5 hidden dim (4096)
- **Stack factor**: 8 (temporal compression of audio frames)
- **Intermediate dim**: TBD (typically 2x-4x of target dim, ~8192-16384)

### 5. Thinking Mode
- **Decision**: Disable Qwen 3.5's thinking/chain-of-thought mode for real-time speech
- **Why**: Thinking mode adds latency unsuitable for real-time speech applications
- **How**: Set `enable_thinking=False` in generation config / omit `<think>` tokens from training data

### 6. Audio Token Integration
- **Decision**: Register `<|audio|>` as a special token in Qwen 3.5's tokenizer if not already present
- **Why**: Ultravox uses `<|audio|>` placeholder tokens that get replaced with projected audio embeddings during forward pass
- **Risk**: Qwen 3.5's 248K vocabulary is large — must verify token ID doesn't collide with existing special tokens

## Risks / Trade-offs
- **Hybrid architecture compatibility**: Qwen 3.5's DeltaNet layers are novel — the Ultravox training framework may need patches to handle them correctly → Mitigation: test forward pass before full training
- **VRAM requirements**: 9B backbone + Whisper encoder + projector requires significant GPU memory → Mitigation: gradient checkpointing, multi-GPU with torchrun
- **Upstream Ultravox changes**: fixie-ai/ultravox may not have first-class Qwen 3.5 support yet → Mitigation: fork and patch as needed
- **Qwen 3.5 is brand new (2026-03-02)**: Limited community experience → Mitigation: validate basic inference before training

## Migration Plan
N/A — greenfield project, no existing model to migrate from.

## Open Questions
- What intermediate dimension should the projector use? (Need to benchmark)
- Should we also train a LoRA adapter on the Qwen 3.5 backbone for improved speech task performance?
- Which speech training datasets to use? (Follow Ultravox v0.7 data mix or customize?)
- What GPU cluster is available for training? (Determines batch size, training time)
