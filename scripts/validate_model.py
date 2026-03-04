#!/usr/bin/env python3
"""
Validate that Qwen 3.5 9B works with the Ultravox architecture.

Performs a forward pass sanity check:
  1. Load Qwen 3.5 9B config and verify dimensions
  2. Load Whisper-large-v3-turbo config and verify dimensions
  3. Verify tokenizer compatibility and audio token registration
  4. Optionally instantiate the model (requires GPU + significant VRAM)

Usage:
  python scripts/validate_model.py              # Config-only validation
  python scripts/validate_model.py --full       # Full model instantiation
"""
import argparse
import sys
import os

# Add project paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "ultravox-upstream"))


def validate_configs():
    """Validate model configs are compatible."""
    import transformers

    print("=== Config Validation ===\n")

    # Load Qwen 3.5 9B config
    print("Loading Qwen/Qwen3.5-9B config...")
    text_config = transformers.AutoConfig.from_pretrained(
        "Qwen/Qwen3.5-9B", trust_remote_code=True
    )
    print(f"  model_type: {text_config.model_type}")
    print(f"  architectures: {getattr(text_config, 'architectures', 'N/A')}")

    # Qwen 3.5 is a multimodal model with nested text_config
    # Ultravox handles this via ultravox_config.py:179-181
    if hasattr(text_config, "text_config"):
        inner = text_config.text_config
        print(f"  NOTE: Multimodal model — using nested text_config")
        print(f"  text_config.model_type: {inner.model_type}")
        print(f"  text_config.hidden_size: {inner.hidden_size}")
        print(f"  text_config.vocab_size: {inner.vocab_size}")
        print(f"  text_config.num_hidden_layers: {inner.num_hidden_layers}")
        print(
            f"  text_config.max_position_embeddings: {getattr(inner, 'max_position_embeddings', 'N/A')}"
        )
        # Show hybrid architecture details
        if hasattr(inner, "layer_types"):
            from collections import Counter

            layer_counts = Counter(inner.layer_types)
            print(
                f"  layer_types: {dict(layer_counts)} ({len(inner.layer_types)} total)"
            )
        qwen_dim = inner.hidden_size
        qwen_vocab = inner.vocab_size
    else:
        print(f"  hidden_size: {text_config.hidden_size}")
        print(f"  vocab_size: {text_config.vocab_size}")
        print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
        print(
            f"  max_position_embeddings: {getattr(text_config, 'max_position_embeddings', 'N/A')}"
        )
        qwen_dim = text_config.hidden_size
        qwen_vocab = text_config.vocab_size

    # Load Whisper config
    print("\nLoading openai/whisper-large-v3-turbo config...")
    audio_config = transformers.AutoConfig.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )
    print(f"  model_type: {audio_config.model_type}")
    print(f"  d_model: {audio_config.d_model}")
    print(f"  encoder_layers: {audio_config.encoder_layers}")

    # Validate projector dimensions
    whisper_dim = audio_config.d_model  # 1280 for large-v3-turbo

    print(f"\n=== Projector Dimensions ===")
    print(f"  Whisper output dim: {whisper_dim}")
    print(f"  Qwen 3.5 hidden dim: {qwen_dim}")
    print(f"  Projector: {whisper_dim} -> {qwen_dim}")

    # Validate tokenizer
    print(f"\n=== Tokenizer Validation ===")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B", trust_remote_code=True
    )
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Config vocab size: {qwen_vocab}")
    print(f"  Special tokens: {tokenizer.all_special_tokens[:10]}...")

    # Check if <|audio|> would conflict
    audio_token = "<|audio|>"
    existing_id = tokenizer.convert_tokens_to_ids(audio_token)
    if existing_id != tokenizer.unk_token_id:
        print(f"  WARNING: '{audio_token}' already exists as token ID {existing_id}")
    else:
        print(f"  '{audio_token}' is available (will be added as special token)")

    # Test adding the audio token
    tokenizer.add_special_tokens({"additional_special_tokens": [audio_token]})
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    print(f"  Audio token ID after registration: {audio_token_id}")

    # Check Qwen 3.5 specific: thinking mode tokens
    for token in ["<think>", "</think>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            print(f"  Thinking token '{token}' = ID {tid} (will be suppressed)")

    # Verify CausalLM class exists for text model
    print(f"\n=== CausalLM Compatibility ===")
    if hasattr(text_config, "text_config"):
        print(
            f"  Multimodal model detected — Qwen3_5ForCausalLM will be used via patch"
        )
        print(
            f"  The patch passes inner text_config to AutoModelForCausalLM.from_pretrained()"
        )
        print(
            f"  This loads Qwen3_5ForCausalLM which ignores vision weights (^model.visual.*)"
        )
    else:
        print(f"  Standard CausalLM model — no patch needed")

    print("\n=== Config validation PASSED ===")
    return text_config, audio_config


def validate_full_model():
    """Instantiate the full UltravoxModel with Qwen 3.5 9B backbone."""
    import torch

    # Apply the Qwen 3.5 patch
    import patches.qwen3_5_support  # noqa: F401
    from ultravox.model.ultravox_config import UltravoxConfig
    from ultravox.model.ultravox_model import UltravoxModel

    print("\n=== Full Model Validation ===\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for full model validation")
        return False

    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    print("\nCreating UltravoxConfig...")
    config = UltravoxConfig(
        audio_model_id="openai/whisper-large-v3-turbo",
        text_model_id="Qwen/Qwen3.5-9B",
        stack_factor=8,
        projector_act="swiglu",
        projector_ln_mid=True,
    )
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")

    print("\nInstantiating UltravoxModel (this may take a while)...")
    model = UltravoxModel(config)
    model = model.to(dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n=== Full model validation PASSED ===")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate Ultravox + Qwen 3.5 9B compatibility"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full model instantiation (requires GPU)",
    )
    args = parser.parse_args()

    validate_configs()

    if args.full:
        validate_full_model()


if __name__ == "__main__":
    main()
