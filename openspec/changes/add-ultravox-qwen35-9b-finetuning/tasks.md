## 1. Environment Setup
- [x] 1.1 Clone and set up fixie-ai/ultravox training framework
- [x] 1.2 Install dependencies (PyTorch, transformers, etc.) with bf16 support
- [ ] 1.3 Verify GPU environment (CUDA, multi-GPU with torchrun)
- [ ] 1.4 Download and validate Qwen/Qwen3.5-9B model weights
- [ ] 1.5 Download and validate openai/whisper-large-v3-turbo weights

## 2. Model Integration
- [x] 2.1 Add Qwen 3.5 9B as a supported text model in Ultravox config
- [x] 2.2 Register `<|audio|>` special token in Qwen 3.5 tokenizer (verify no collision)
- [x] 2.3 Configure UltravoxProjector dimensions (Whisper 1280 → Qwen 4096)
- [ ] 2.4 Validate forward pass: audio input → Whisper → projector → Qwen 3.5 → text output
- [x] 2.5 Disable Qwen 3.5 thinking mode in generation config
- [x] 2.6 Handle any Qwen 3.5 hybrid DeltaNet architecture compatibility issues

## 3. Training Configuration
- [x] 3.1 Create YAML training config for Qwen 3.5 9B backbone
- [x] 3.2 Configure knowledge distillation loss (match Qwen 3.5 text-only logits)
- [x] 3.3 Set up dataset pipeline (ASR datasets + LLM-generated continuations)
- [x] 3.4 Configure gradient checkpointing and bf16 mixed precision
- [x] 3.5 Set up distributed training (torchrun multi-GPU)
- [x] 3.6 Tune hyperparameters (learning rate, batch size, warmup, stack_factor)

## 4. Training Execution
- [ ] 4.1 Run short validation training (few hundred steps) to verify convergence
- [ ] 4.2 Monitor loss curves and gradient norms for stability
- [ ] 4.3 Run full training to convergence
- [ ] 4.4 Save checkpoints and select best model

## 5. Evaluation
- [ ] 5.1 Evaluate LibriSpeech WER
- [ ] 5.2 Evaluate Big Bench Audio
- [ ] 5.3 Evaluate VoiceBench
- [ ] 5.4 Compare against Ultravox v0.7-GLM-4.6 baseline
- [ ] 5.5 Test real-time inference latency

## 6. Publishing & Serving
- [x] 6.1 Export model in HuggingFace format (script created)
- [ ] 6.2 Validate vLLM compatibility for serving
- [x] 6.3 Write model card with architecture details, benchmarks, and usage instructions
- [ ] 6.4 Publish to HuggingFace Hub
