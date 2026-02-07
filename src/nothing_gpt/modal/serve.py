"""Serve fine-tuned Seinfeld model via vLLM on Modal."""

import json
import subprocess

import modal

from .config import ADAPTER_PATH, BASE_MODEL, app, hf_cache, serve_image, vol

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}

LORA_CONFIG = json.dumps({
    "name": "seinfeld",
    "path": ADAPTER_PATH,
    "local_path": ADAPTER_PATH,
})


@app.function(
    image=serve_image,
    gpu="T4",
    volumes=VOLUMES,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve() -> None:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "2048",
        "--enable-lora",
        "--lora-modules", LORA_CONFIG,
        "--dtype", "half",
    ]
    subprocess.Popen(cmd)
