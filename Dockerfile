ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_TF32_OVERRIDE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/root/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

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
CMD ["sh", "-c", "echo Starting uvicorn on 0.0.0.0:${PORT} && uvicorn serve_asr:app --host 0.0.0.0 --port ${PORT} --log-level info"]