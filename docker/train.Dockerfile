FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.13 -m ensurepip && python3.13 -m pip install --upgrade pip

RUN python3.13 -m pip install --no-cache-dir \
    "torch>=2.5" \
    "trl>=0.15" \
    "peft>=0.14" \
    "transformers>=4.48" \
    "bitsandbytes>=0.45" \
    "datasets>=3.2" \
    "accelerate>=1.3" \
    "wandb>=0.19"

COPY pyproject.toml /app/
COPY nothing_gpt/ /app/nothing_gpt/
WORKDIR /app
RUN python3.13 -m pip install --no-cache-dir .

CMD ["python3.13", "-m", "nothing_gpt.sft.train"]
