"""Modal entry point for Gradio UI."""

import modal

from nothing_gpt.modal.config import ui_image

app = modal.App("nothing-gpt")


@app.function(image=ui_image, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=120)
def web() -> None:
    import os

    os.environ.setdefault("SERVE_URL", "https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1")

    from nothing_gpt.ui.app import web as core_web

    core_web(prevent_thread_lock=True)
