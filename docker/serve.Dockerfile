FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.13 -m ensurepip && python3.13 -m pip install --upgrade pip

RUN python3.13 -m pip install --no-cache-dir "vllm>=0.7"

COPY pyproject.toml /app/
COPY nothing_gpt/ /app/nothing_gpt/
WORKDIR /app
RUN python3.13 -m pip install --no-cache-dir .

EXPOSE 8000
CMD ["python3.13", "-m", "nothing_gpt.serve.server"]
