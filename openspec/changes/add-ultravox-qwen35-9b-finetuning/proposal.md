# Change: Add Ultravox v0.7 Qwen 3.5 9B Finetuning

## Why
Ultravox v0.7 currently only ships with a GLM-4.6 (0.7B) backbone — a very small model. By pairing the Ultravox v0.7 speech architecture with Qwen 3.5 9B (released 2026-03-02), we get a speech-language model with significantly stronger reasoning, 262K native context, multilingual support (201 languages), and an efficient hybrid DeltaNet architecture — all under an Apache 2.0 license.

## What Changes
- Define model architecture spec: Whisper-large-v3-turbo encoder + UltravoxProjector + Qwen 3.5 9B backbone
- Define training pipeline spec: knowledge distillation training with frozen Qwen 3.5 9B, trainable projector + optional encoder finetuning
- Define dataset preparation and configuration
- Define inference serving spec for vLLM compatibility
- Set up training configs (YAML) for the Qwen 3.5 9B backbone
- Handle Qwen 3.5 9B specifics: large vocabulary (248K), hybrid DeltaNet architecture, thinking mode disablement, bf16-only constraint

## Impact
- Affected specs: `model-architecture`, `model-training`, `inference-serving` (all new)
- Affected code: training configs, model initialization, tokenizer integration, projector dimension mapping
- No breaking changes (greenfield project)
