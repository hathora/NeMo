# Minimal ASR-only image
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# Performance Optimization Environment Variables
# ============================================================================
# Enable TF32 for faster computation on Ampere+ GPUs
ENV NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA memory allocation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache Hugging Face models
ENV HF_HOME=/root/.cache/huggingface

# Disable tokenizer parallelism warnings
ENV TOKENIZERS_PARALLELISM=false

# Optimize NCCL for single-GPU inference
ENV NCCL_DEBUG=WARN

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox libsox-fmt-all \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install "nemo_toolkit[asr]" fastapi uvicorn[standard] python-multipart

WORKDIR /workspace

COPY .hathora_build/app/serve_asr.py /workspace/serve_asr.py

ENV PORT=8080

EXPOSE 8080

# ============================================================================
# Recommended docker run flags for optimal performance:
# docker run --gpus all --ipc=host --shm-size=8g \
#   --ulimit memlock=-1 --ulimit stack=67108864 \
#   -p 8080:8080 -e MODEL_ID=<model> <image>
# ============================================================================
CMD ["sh", "-c", "echo Starting uvicorn on 0.0.0.0:${PORT} && uvicorn serve_asr:app --host 0.0.0.0 --port ${PORT} --log-level info"]