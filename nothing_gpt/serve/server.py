"""Serve fine-tuned Seinfeld model via vLLM."""

import json
import subprocess
import sys

from nothing_gpt.constants import ADAPTER_PATH, BASE_MODEL


def serve(background: bool = False) -> None:
    lora_module = json.dumps({"name": "seinfeld", "path": ADAPTER_PATH})
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "2048",
        "--enable-lora",
        "--max-lora-rank", "32",
        "--lora-modules", lora_module,
        "--dtype", "half",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
    ]
    if background:
        subprocess.Popen(cmd)
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    serve()
