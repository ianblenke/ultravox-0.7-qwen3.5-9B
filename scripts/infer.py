#!/usr/bin/env python3
"""
Inference script for Ultravox v0.7 Qwen 3.5 9B.

Usage:
  python scripts/infer.py <model_path> --audio <audio_file>
  python scripts/infer.py <model_path> --audio audio.wav --prompt "Transcribe this audio"
"""
import argparse
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "ultravox-upstream"))

# Apply patch
import patches.qwen3_5_support  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Ultravox v0.7 Qwen 3.5 9B Inference")
    parser.add_argument("model_path", help="Path to model checkpoint or HF model ID")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--prompt", default=None, help="Text prompt (default: transcription)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    import torch
    import librosa
    from ultravox.model.ultravox_model import UltravoxModel
    from ultravox.model.ultravox_processing import UltravoxProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from {args.model_path}...")
    model = UltravoxModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    processor = UltravoxProcessor.from_pretrained(args.model_path)

    # Get thinking token IDs to suppress
    tokenizer = processor.tokenizer
    suppress_ids = []
    for token in ["<think>", "</think>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            suppress_ids.append(tid)

    print(f"Loading audio from {args.audio}...")
    audio, sr = librosa.load(args.audio, sr=16000)
    print(f"  Duration: {len(audio)/sr:.1f}s")

    # Build prompt
    if args.prompt:
        text = f"<|audio|>\n{args.prompt}"
    else:
        text = "<|audio|>"

    inputs = processor(audio=audio, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Generating...")
    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    if suppress_ids:
        gen_kwargs["suppress_tokens"] = suppress_ids

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n--- Output ---")
    print(result)


if __name__ == "__main__":
    main()
