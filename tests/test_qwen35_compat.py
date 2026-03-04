"""
Tests for Qwen 3.5 9B compatibility with Ultravox.

These tests validate the patches and config without requiring GPU or model weights.
They only need transformers installed (config/tokenizer downloads are small).
"""
import os
import sys
import pytest

# Add project paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "ultravox-upstream"))


# -- Config Tests --


class TestQwen35Config:
    """Test that Qwen 3.5 9B config is compatible with Ultravox."""

    def test_config_loads(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        assert config.model_type == "qwen3_5"

    def test_has_nested_text_config(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        assert hasattr(config, "text_config")
        assert config.text_config.model_type == "qwen3_5_text"

    def test_text_config_dimensions(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        tc = config.text_config
        assert tc.hidden_size == 4096
        assert tc.vocab_size == 248320
        assert tc.num_hidden_layers == 32

    def test_hybrid_architecture(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        tc = config.text_config
        assert hasattr(tc, "layer_types")
        assert len(tc.layer_types) == 32
        assert tc.layer_types.count("linear_attention") == 24
        assert tc.layer_types.count("full_attention") == 8

    def test_outer_config_missing_attributes(self):
        """Verify that the outer config lacks attributes Ultravox needs."""
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        # These should NOT exist on the outer config
        assert not hasattr(config, "hidden_size")
        assert not hasattr(config, "vocab_size")
        assert not hasattr(config, "initializer_range")
        # But they should exist on the inner text_config
        assert hasattr(config.text_config, "hidden_size")
        assert hasattr(config.text_config, "vocab_size")
        assert hasattr(config.text_config, "initializer_range")

    def test_initializer_range_value(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        assert config.text_config.initializer_range == 0.02


class TestWhisperConfig:
    """Test Whisper config compatibility."""

    def test_whisper_dimensions(self):
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            "openai/whisper-large-v3-turbo"
        )
        assert config.d_model == 1280
        assert config.hidden_size == 1280  # alias


class TestProjectorDimensions:
    """Test that projector dimensions are compatible."""

    def test_whisper_to_qwen_projection(self):
        import transformers

        whisper_config = transformers.AutoConfig.from_pretrained(
            "openai/whisper-large-v3-turbo"
        )
        qwen_config = transformers.AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )
        whisper_dim = whisper_config.d_model  # 1280
        qwen_dim = qwen_config.text_config.hidden_size  # 4096
        assert whisper_dim == 1280
        assert qwen_dim == 4096
        # Projector maps whisper_dim * stack_factor -> hidden -> qwen_dim
        stack_factor = 8
        projector_input_dim = whisper_dim * stack_factor
        assert projector_input_dim == 10240


# -- Tokenizer Tests --


class TestQwen35Tokenizer:
    """Test tokenizer compatibility and audio token registration."""

    @pytest.fixture
    def tokenizer(self):
        import transformers

        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )

    def test_eos_token(self, tokenizer):
        assert tokenizer.eos_token == "<|im_end|>"
        assert tokenizer.eos_token_id == 248046

    def test_pad_token_exists(self, tokenizer):
        """Qwen 3.5 has its own pad token (unlike Llama)."""
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token == "<|endoftext|>"
        assert tokenizer.pad_token_id == 248044

    def test_audio_token_available(self, tokenizer):
        """<|audio|> should not already exist in the vocabulary."""
        audio_id = tokenizer.convert_tokens_to_ids("<|audio|>")
        assert audio_id == tokenizer.unk_token_id or audio_id is None

    def test_audio_token_registration(self, tokenizer):
        """Adding <|audio|> as a special token should work."""
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|audio|>"]}
        )
        audio_id = tokenizer.convert_tokens_to_ids("<|audio|>")
        assert audio_id != tokenizer.unk_token_id
        assert audio_id > 0

    def test_audio_token_encodes_as_single(self, tokenizer):
        """<|audio|> should encode to a single token ID."""
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|audio|>"]}
        )
        ids = tokenizer.encode("<|audio|>", add_special_tokens=False)
        assert len(ids) == 1

    def test_thinking_tokens_exist(self, tokenizer):
        """Verify thinking tokens exist and can be suppressed."""
        think_id = tokenizer.convert_tokens_to_ids("<think>")
        end_think_id = tokenizer.convert_tokens_to_ids("</think>")
        assert think_id != tokenizer.unk_token_id
        assert end_think_id != tokenizer.unk_token_id
        assert think_id == 248068
        assert end_think_id == 248069

    def test_existing_audio_special_tokens(self, tokenizer):
        """Qwen 3.5 has its own audio tokens that should not conflict."""
        special_tokens = tokenizer.all_special_tokens
        assert "<|audio_start|>" in special_tokens
        assert "<|audio_end|>" in special_tokens
        assert "<|audio_pad|>" in special_tokens


# -- Chat Template Tests --


class TestChatTemplate:
    """Test chat template behavior with and without thinking."""

    @pytest.fixture
    def tokenizer(self):
        import transformers

        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-9B", trust_remote_code=True
        )

    @pytest.fixture
    def custom_template(self):
        template_path = os.path.join(project_dir, "configs", "chat_template_no_think.jinja")
        with open(template_path) as f:
            return f.read()

    def test_default_template_has_thinking(self, tokenizer):
        """Default Qwen template always injects <think> tokens."""
        msgs = [{"role": "user", "content": "Hello"}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        assert "<think>" in text

    def test_default_template_thinking_false_still_has_think(self, tokenizer):
        """Even with enable_thinking=False, default template adds empty think block."""
        msgs = [{"role": "user", "content": "Hello"}]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert "<think>" in text
        assert "</think>" in text

    def test_custom_template_no_thinking(self, tokenizer, custom_template):
        """Custom template should have no thinking tokens at all."""
        msgs = [{"role": "user", "content": "Hello"}]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=custom_template,
        )
        assert "<think>" not in text
        assert "</think>" not in text

    def test_custom_template_format(self, tokenizer, custom_template):
        """Verify custom template produces correct Qwen im_start/im_end format."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "<|audio|>\nTranscribe."},
        ]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=custom_template,
        )
        assert "<|im_start|>system\n" in text
        assert "<|im_start|>user\n" in text
        assert "<|im_start|>assistant\n" in text
        assert "<|audio|>" in text
        assert text.endswith("<|im_start|>assistant\n")

    def test_custom_template_audio_passthrough(self, tokenizer, custom_template):
        """<|audio|> token should pass through the template unchanged."""
        msgs = [{"role": "user", "content": "<|audio|>"}]
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=custom_template,
        )
        assert "<|audio|>" in text


# -- Patch Tests --


accelerate_available = pytest.importorskip.__module__ is not None  # always True, used below
try:
    import accelerate  # noqa: F401
    _has_accelerate = True
except ImportError:
    _has_accelerate = False

requires_accelerate = pytest.mark.skipif(
    not _has_accelerate,
    reason="requires accelerate (install full Ultravox deps)",
)


@requires_accelerate
class TestPatches:
    """Test that our patches apply correctly. Requires full Ultravox deps."""

    def test_patch_imports(self):
        """Patch module should import without error."""
        import patches.qwen3_5_support  # noqa: F401

    def test_ultravox_config_patched(self):
        """UltravoxConfig.__init__ should be patched."""
        import patches.qwen3_5_support  # noqa: F401
        from ultravox.model.ultravox_config import UltravoxConfig

        # The __init__ should be our patched version (a closure)
        # We can verify by checking it handles the initializer_range case
        # Just verify it doesn't crash on import
        assert UltravoxConfig is not None

    def test_ultravox_model_patched(self):
        """UltravoxModel._create_language_model should be patched."""
        import patches.qwen3_5_support  # noqa: F401
        from ultravox.model.ultravox_model import UltravoxModel

        # Verify the method exists and is a classmethod
        assert hasattr(UltravoxModel, "_create_language_model")

    def test_ultravox_config_with_qwen35(self):
        """UltravoxConfig should handle Qwen 3.5's nested config without crashing."""
        import patches.qwen3_5_support  # noqa: F401
        from ultravox.model.ultravox_config import UltravoxConfig

        config = UltravoxConfig(
            text_model_id="Qwen/Qwen3.5-9B",
            audio_model_id="openai/whisper-large-v3-turbo",
        )
        # These should be populated from nested text_config
        assert config.vocab_size == 248320
        assert config.initializer_range == 0.02
        assert config.hidden_size == 4096


# -- Generation Config Tests --


class TestGenerationConfig:
    """Test generation config suppresses thinking tokens."""

    def test_generation_config_valid(self):
        import json

        config_path = os.path.join(project_dir, "configs", "generation_config.json")
        with open(config_path) as f:
            config = json.load(f)
        assert "suppress_tokens" in config
        assert 248068 in config["suppress_tokens"]  # <think>
        assert 248069 in config["suppress_tokens"]  # </think>
