"""Serve fine-tuned Seinfeld model via vLLM."""

import json
import os
import subprocess
import sys

from nothing_gpt.constants import ADAPTER_PATH, BASE_MODEL, DPO_ADAPTER_PATH


def _lora_modules() -> str:
    """Build LoRA modules JSON, conditionally including the DPO adapter."""
    modules = [{"name": "seinfeld", "path": ADAPTER_PATH}]
    if os.path.isdir(DPO_ADAPTER_PATH):
        modules.append({"name": "seinfeld-dpo", "path": DPO_ADAPTER_PATH})
    return json.dumps(modules)


def serve(background: bool = False) -> None:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "2048",
        "--enable-lora",
        "--max-lora-rank", "32",
        "--lora-modules", _lora_modules(),
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
