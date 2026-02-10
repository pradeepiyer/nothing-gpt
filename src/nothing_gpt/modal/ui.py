"""Gradio scene generator UI served on Modal."""

import re

import modal

from nothing_gpt.characters import SCRIPT_PROMPT

from .config import ui_image

app = modal.App("nothing-gpt")

SERVE_URL = "https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1"
CONTEXT_LINES = 10  # Lines of scene to send as context for each API call
NUM_ROUNDS = 12  # Number of API calls to generate a full scene


def _parse_line(text: str) -> str | None:
    """Extract a [CHARACTER] dialogue line from model output."""
    match = re.match(r"\[([A-Z]+)\] .+", text.strip())
    return text.strip() if match else None


@app.function(image=ui_image, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=120)
def web() -> None:
    import gradio as gr  # type: ignore[import-not-found]
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(base_url=SERVE_URL, api_key="not-needed", timeout=300)

    def generate_scene(premise: str):  # type: ignore[no-untyped-def]
        if not premise.strip():
            yield "Enter a premise to generate a scene."
            return

        scene_lines: list[str] = []

        for _ in range(NUM_ROUNDS):
            context = "\n".join(scene_lines[-CONTEXT_LINES:]) if scene_lines else premise
            messages = [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": context},
            ]

            try:
                response = client.chat.completions.create(
                    model="seinfeld", messages=messages, max_tokens=256,
                )
                text = response.choices[0].message.content or ""
                for line in text.strip().split("\n"):
                    parsed = _parse_line(line)
                    if parsed:
                        scene_lines.append(parsed)
            except Exception as e:
                scene_lines.append(f"[Error: {e}]")
                break

            yield "\n".join(scene_lines)

        yield "\n".join(scene_lines)

    with gr.Blocks(theme=gr.themes.Monochrome(), title="Nothing-GPT") as demo:
        gr.Markdown("# Nothing-GPT\nGenerate a Seinfeld scene from a premise.")
        premise_input = gr.Textbox(
            label="Premise",
            placeholder="Jerry discovers his dry cleaner has been wearing his clothes...",
            lines=2,
        )
        generate_btn = gr.Button("Generate Scene", variant="primary")
        output = gr.Textbox(label="Scene", lines=20, interactive=False)

        generate_btn.click(fn=generate_scene, inputs=premise_input, outputs=output)

    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=True)
