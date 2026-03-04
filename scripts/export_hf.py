#!/usr/bin/env python3
"""
Export trained Ultravox v0.7 Qwen 3.5 9B model to HuggingFace format.

Usage:
  python scripts/export_hf.py <checkpoint_path> <output_dir>
  python scripts/export_hf.py runs/exp--2026-03-04/checkpoint-best ./ultravox-v0.7-qwen3.5-9b
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ultravox-upstream'))


def export_model(checkpoint_path: str, output_dir: str, push_to_hub: bool = False, hub_repo: str = ""):
    import torch
    import transformers
    from ultravox.model.ultravox_model import UltravoxModel
    from ultravox.model.ultravox_processing import UltravoxProcessor

    print(f"Loading checkpoint from {checkpoint_path}...")
    model = UltravoxModel.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading processor...")
    processor = UltravoxProcessor.from_pretrained(checkpoint_path)

    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Write model card
    model_card = """---
language:
  - en
  - multilingual
tags:
  - ultravox
  - speech
  - multimodal
  - qwen3.5
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-9B
  - openai/whisper-large-v3-turbo
pipeline_tag: audio-text-to-text
---

# Ultravox v0.7 - Qwen 3.5 9B

A multimodal speech-language model that processes audio directly into language model embedding space.

## Architecture

| Component | Model | Parameters |
|-----------|-------|------------|
| Audio Encoder | whisper-large-v3-turbo | ~800M |
| Projector | UltravoxProjector (SwiGLU + RMSNorm) | ~50M |
| LLM Backbone | Qwen 3.5 9B (frozen) | 9.65B |

## Training

- **Method**: Knowledge distillation (KL divergence against frozen Qwen 3.5 9B text logits)
- **Precision**: bf16
- **Audio compression**: stack_factor=8

## Usage

```python
from transformers import AutoModel, AutoProcessor
import librosa

model = AutoModel.from_pretrained("YOUR_REPO/ultravox-v0.7-qwen3.5-9b", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("YOUR_REPO/ultravox-v0.7-qwen3.5-9b")

audio, sr = librosa.load("audio.wav", sr=16000)
inputs = processor(audio=audio, text="<|audio|>", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## License

Apache 2.0 (inherits from Qwen 3.5 9B)
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    print(f"Export complete: {output_dir}")

    if push_to_hub and hub_repo:
        print(f"Pushing to HuggingFace Hub: {hub_repo}...")
        model.push_to_hub(hub_repo)
        processor.push_to_hub(hub_repo)
        print(f"Pushed to https://huggingface.co/{hub_repo}")


def main():
    parser = argparse.ArgumentParser(description="Export Ultravox model to HuggingFace format")
    parser.add_argument("checkpoint_path", help="Path to training checkpoint")
    parser.add_argument("output_dir", help="Output directory for HF model")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", default="", help="HuggingFace repo ID (e.g., user/model-name)")
    args = parser.parse_args()

    export_model(args.checkpoint_path, args.output_dir, args.push_to_hub, args.hub_repo)


if __name__ == "__main__":
    main()
