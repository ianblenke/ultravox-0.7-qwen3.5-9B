<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Project: Ultravox v0.7 with Qwen 3.5 9B Backbone

## What This Is
Finetuning Ultravox v0.7 (multimodal speech model) to use Qwen 3.5 9B as the LLM backbone instead of the default GLM-4.6 (0.7B). Architecture: Whisper encoder → MLP projector → Qwen 3.5 9B (frozen).

## Key Files
- `patches/qwen3_5_support.py` — Critical monkey-patch for Qwen 3.5 multimodal config loading (auto-applied on import)
- `train.py` — Training entry point (applies patches, fixes file:// path resolution, calls Ultravox training)
- `configs/v0.7_config_qwen3.5_9b.yaml` — Full training config
- `configs/v0.7_config_qwen3.5_9b_quick.yaml` — 200-step smoke test config
- `configs/chat_template_no_think.jinja` — Custom template that strips Qwen 3.5's thinking mode
- `configs/generation_config.json` — Suppresses `<think>`/`</think>` token IDs during inference
- `scripts/` — setup.sh, train.sh, eval.sh, validate_model.py, infer.py, export_hf.py
- `tests/test_qwen35_compat.py` — 25 tests (21 pass without GPU, 4 skip without `accelerate`)
- `openspec/changes/add-ultravox-qwen35-9b-finetuning/` — Design docs, tasks, specs

## Current Status
All code/config work that can be done without a GPU is **complete and committed**. See `openspec/changes/add-ultravox-qwen35-9b-finetuning/tasks.md` for the full task checklist.

## Next Steps (GPU Required)
Run these in order:

### 1. Environment Setup
```bash
bash scripts/setup.sh   # Clones fixie-ai/ultravox into ultravox-upstream/, installs poetry deps
```

### 2. Validate Config (no GPU needed, but needs full deps)
```bash
cd ultravox-upstream && poetry run python ../scripts/validate_model.py
```

### 3. Validate Forward Pass (needs GPU)
```bash
cd ultravox-upstream && poetry run python ../scripts/validate_model.py --full
```

### 4. Smoke Test Training (200 steps)
```bash
bash scripts/train.sh --config_path configs/v0.7_config_qwen3.5_9b_quick.yaml
```

### 5. Full Training
```bash
bash scripts/train.sh --config_path configs/v0.7_config_qwen3.5_9b.yaml
```

### 6. Evaluation
```bash
bash scripts/eval.sh <checkpoint_path>
```

### 7. Export & Publish
```bash
cd ultravox-upstream && poetry run python ../scripts/export_hf.py --checkpoint <path> --output_dir ./ultravox-v0.7-qwen3.5-9b
```

## Important Technical Details
- Qwen 3.5 is `Qwen3_5ForConditionalGeneration` (multimodal), NOT a CausalLM — the patch in `patches/qwen3_5_support.py` handles this by passing inner `text_config` to load `Qwen3_5ForCausalLM`
- Qwen 3.5's default chat template **always** injects `<think>` tokens even with `enable_thinking=False` — custom template is required
- bf16 only — Qwen 3.5's DeltaNet layers have high quantization error at 4-bit
- `<|audio|>` registers as token ID 248077 with no collisions
- Training runs from `ultravox-upstream/` directory but configs live in project root — `train.py` handles path resolution