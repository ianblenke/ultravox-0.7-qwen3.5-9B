FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.cache/huggingface

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git git-lfs wget curl ffmpeg libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Clone Ultravox
RUN git clone https://github.com/fixie-ai/ultravox.git ultravox-upstream

# Install dependencies
WORKDIR /workspace/ultravox-upstream
RUN pip install poetry==1.7.1 && poetry install --no-interaction

# Copy project configs and scripts
WORKDIR /workspace
COPY configs/ configs/
COPY scripts/ scripts/
RUN chmod +x scripts/*.sh

# Prefetch model configs (not full weights — too large for build)
RUN python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True); print('Qwen config OK')"
RUN python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('openai/whisper-large-v3-turbo'); print('Whisper config OK')"

ENV PYTHONPATH=/workspace/ultravox-upstream:$PYTHONPATH

ENTRYPOINT ["bash"]
CMD ["scripts/train.sh"]
