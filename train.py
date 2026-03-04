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

# Apply Qwen 3.5 compatibility patch BEFORE importing ultravox training
import patches.qwen3_5_support  # noqa: F401

# Now run the standard Ultravox training
from ultravox.training import train

if __name__ == "__main__":
    train.main()
