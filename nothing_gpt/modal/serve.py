"""Modal entry point for vLLM serving."""

import modal

from nothing_gpt.modal.config import hf_cache, serve_image, vol

app = modal.App("nothing-gpt-serve")

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}


@app.function(
    image=serve_image,
    gpu="L4",
    volumes=VOLUMES,
    timeout=600,
    scaledown_window=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve() -> None:
    from nothing_gpt.serve.server import serve as core_serve

    core_serve(background=True)
