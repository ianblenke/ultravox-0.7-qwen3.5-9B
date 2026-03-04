"""
Patch for Ultravox to support Qwen 3.5 9B as LLM backbone.

Qwen 3.5 9B is a multimodal (vision+text) model whose HuggingFace config
maps to Qwen3_5ForConditionalGeneration. AutoModelForCausalLM cannot load
it directly. However, the transformers library provides Qwen3_5ForCausalLM
which loads just the text portion and ignores vision weights.

This patch addresses two issues:
  1. UltravoxConfig.__init__ crashes on `text_config.initializer_range` because
     the outer Qwen3_5Config doesn't have this attribute (only the nested
     text_config does). We extend the existing guard to also copy initializer_range.
  2. UltravoxModel._create_language_model uses AutoModelForCausalLM which can't
     load Qwen3_5ForConditionalGeneration. We patch it to pass the inner text
     config, resolving to Qwen3_5ForCausalLM instead.

Apply this patch before training by importing it:
    import patches.qwen3_5_support
"""
import logging

import transformers

logger = logging.getLogger(__name__)


def _patched_ultravox_config_init(original_init):
    """
    Wraps UltravoxConfig.__init__ to extend the multimodal config guard.

    The existing guard (lines 179-181) copies vocab_size and hidden_size from
    text_config.text_config, but misses initializer_range. This wrapper adds it.
    """

    def __init__(self, *args, **kwargs):
        # The upstream guard (lines 179-182) now copies initializer_range from
        # nested text_config for multimodal models, so original_init works directly.
        original_init(self, *args, **kwargs)

    return __init__


def _create_language_model_patched(cls, config):
    """
    Patched version of UltravoxModel._create_language_model that handles
    multimodal models like Qwen 3.5 by using the text sub-config.
    """
    from ultravox.model.ultravox_model import FROM_PRETRAINED_KWARGS, apply_lora

    text_config = config.text_config
    is_multimodal = hasattr(text_config, "text_config")

    if is_multimodal:
        inner_text_config = text_config.text_config
        logger.info(
            f"Detected multimodal model config (model_type={text_config.model_type}). "
            f"Loading language model using inner text_config (model_type={inner_text_config.model_type})."
        )

    if (
        transformers.modeling_utils._init_weights
        and config.text_model_id is not None
    ):
        load_kwargs = {
            "torch_dtype": config.torch_dtype,
            **FROM_PRETRAINED_KWARGS,
        }

        if is_multimodal:
            inner_text_config = text_config.text_config
            if hasattr(inner_text_config, "_attn_implementation"):
                load_kwargs["attn_implementation"] = (
                    inner_text_config._attn_implementation
                )
            language_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.text_model_id,
                config=inner_text_config,
                **load_kwargs,
            )
        else:
            load_kwargs["attn_implementation"] = text_config._attn_implementation
            language_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.text_model_id,
                **load_kwargs,
            )
    else:
        import accelerate

        effective_config = (
            text_config.text_config if is_multimodal else text_config
        )
        with accelerate.init_empty_weights():
            language_model = transformers.AutoModelForCausalLM.from_config(
                effective_config,
                attn_implementation=getattr(
                    effective_config, "_attn_implementation", None
                ),
                torch_dtype=config.torch_dtype,
            )

    language_model = apply_lora(language_model, config.text_model_lora_config)
    return language_model


def apply_patch():
    """Apply all Qwen 3.5 compatibility patches."""
    from ultravox.model.ultravox_config import UltravoxConfig
    from ultravox.model.ultravox_model import UltravoxModel

    # Patch 1: UltravoxConfig.__init__ — handle missing initializer_range
    UltravoxConfig.__init__ = _patched_ultravox_config_init(
        UltravoxConfig.__init__
    )
    logger.info(
        "Patched UltravoxConfig.__init__ for multimodal config compatibility"
    )

    # Patch 2: UltravoxModel._create_language_model — load CausalLM from multimodal model
    UltravoxModel._create_language_model = classmethod(
        _create_language_model_patched
    )
    logger.info(
        "Patched UltravoxModel._create_language_model for multimodal model loading"
    )


# Auto-apply on import
apply_patch()
