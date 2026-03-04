"""
Patch for Ultravox to support Qwen 3.5 9B as LLM backbone.

Qwen 3.5 9B is a multimodal (vision+text) model whose HuggingFace config
maps to Qwen3_5ForConditionalGeneration. AutoModelForCausalLM cannot load
it directly. However, the transformers library provides Qwen3_5ForCausalLM
which loads just the text portion and ignores vision weights.

This patch monkey-patches UltravoxModel._create_language_model to handle
multimodal models by loading with the text sub-config.

Apply this patch before training by importing it:
    import patches.qwen3_5_support
"""
import logging

import transformers

logger = logging.getLogger(__name__)


def _create_language_model_patched(cls, config):
    """
    Patched version of UltravoxModel._create_language_model that handles
    multimodal models like Qwen 3.5 by using the text sub-config.
    """
    from ultravox.model.ultravox_model import FROM_PRETRAINED_KWARGS, apply_lora

    text_config = config.text_config
    is_multimodal = hasattr(text_config, "text_config")

    if is_multimodal:
        # For multimodal models (e.g., Qwen 3.5), the outer config is the
        # full multimodal config. We need to use the inner text_config to
        # load just the language model via AutoModelForCausalLM.
        inner_text_config = text_config.text_config if is_multimodal else text_config
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
            # Pass the inner text config so AutoModelForCausalLM resolves to
            # the correct text-only CausalLM class (e.g., Qwen3_5ForCausalLM)
            inner_text_config = text_config.text_config
            # Set attn_implementation from inner config if available
            if hasattr(inner_text_config, "_attn_implementation"):
                load_kwargs["attn_implementation"] = inner_text_config._attn_implementation
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

        effective_config = text_config.text_config if is_multimodal else text_config
        with accelerate.init_empty_weights():
            language_model = transformers.AutoModelForCausalLM.from_config(
                effective_config,
                attn_implementation=getattr(effective_config, "_attn_implementation", None),
                torch_dtype=config.torch_dtype,
            )

    language_model = apply_lora(language_model, config.text_model_lora_config)
    return language_model


def apply_patch():
    """Apply the Qwen 3.5 compatibility patch to UltravoxModel."""
    from ultravox.model.ultravox_model import UltravoxModel

    original_method = UltravoxModel._create_language_model
    UltravoxModel._create_language_model = classmethod(
        _create_language_model_patched
    )
    logger.info("Applied Qwen 3.5 multimodal model loading patch to UltravoxModel")
    return original_method


# Auto-apply on import
apply_patch()
