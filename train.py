#!/usr/bin/env python3
"""
Training entry point for Ultravox v0.7 Qwen 3.5 9B.

This wrapper applies the Qwen 3.5 multimodal model loading patch
before invoking the standard Ultravox training pipeline.

Usage:
  python train.py --config_path configs/v0.7_config_qwen3.5_9b.yaml
  torchrun --nproc_per_node=8 train.py --config_path configs/v0.7_config_qwen3.5_9b.yaml
"""
import sys
import os

# Add project root and ultravox-upstream to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "ultravox-upstream"))

# Resolve file:// paths in chat_template relative to project root.
# Ultravox's config_base.py resolves file:// paths relative to cwd,
# but training runs from ultravox-upstream/. We fix chat_template paths
# in sys.argv before Ultravox parses them.
for i, arg in enumerate(sys.argv):
    if arg.startswith("--chat_template") and "file://" in arg:
        # Handle --chat_template=file://path
        prefix, path = arg.split("file://", 1)
        if not os.path.isabs(path):
            sys.argv[i] = f"{prefix}file://{os.path.join(project_dir, path)}"

# Also patch file:// paths in YAML configs by monkey-patching the config loader.
# The YAML config sets chat_template: "file://configs/chat_template_no_think.jinja"
# which needs to resolve relative to project_dir, not cwd.
_original_post_init = None


def _patch_config_post_init():
    """Patch TrainConfig.__post_init__ to resolve file:// paths relative to project root."""
    from ultravox.training.config_base import TrainConfig

    global _original_post_init
    _original_post_init = TrainConfig.__post_init__

    def patched_post_init(self):
        # Resolve file:// chat_template paths relative to project root before original __post_init__
        if (
            self.chat_template
            and self.chat_template.startswith("file://")
        ):
            path = self.chat_template[7:].strip()
            if not os.path.isabs(path):
                self.chat_template = f"file://{os.path.join(project_dir, path)}"

        _original_post_init(self)

    TrainConfig.__post_init__ = patched_post_init


# Apply Qwen 3.5 compatibility patch BEFORE importing ultravox training
import patches.qwen3_5_support  # noqa: F401

# Patch config path resolution
_patch_config_post_init()

# Now run the standard Ultravox training
from ultravox.training import train

if __name__ == "__main__":
    train.main()
