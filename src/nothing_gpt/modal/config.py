"""Modal app configuration, volumes, images, and constants."""

import modal

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/vol/adapters/seinfeld"
DATA_PATH = "/vol/data"

app = modal.App("nothing-gpt")

vol = modal.Volume.from_name("nothing-gpt-vol", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch>=2.5",
        "trl>=0.15",
        "peft>=0.14",
        "transformers>=4.48",
        "bitsandbytes>=0.45",
        "datasets>=3.2",
        "accelerate>=1.3",
        "wandb>=0.19",
    )
)

serve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .pip_install("vllm>=0.7")
)

ui_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "gradio>=5.0",
    "openai>=1.0",
)
