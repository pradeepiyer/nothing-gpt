"""Gradio scene generator UI for Nothing-GPT."""

import os
import re

import gradio as gr
from openai import OpenAI

from nothing_gpt.constants import SCRIPT_PROMPT

SERVE_URL = os.environ.get("SERVE_URL", "http://nothing-gpt-serve:8000/v1")
MODEL = "seinfeld"
CONTEXT_LINES = 10  # Lines of scene to send as context for each API call
NUM_ROUNDS = 4  # Number of API calls to generate a full scene


def _parse_line(text: str) -> str | None:
    """Extract a [CHARACTER] dialogue line from model output."""
    match = re.match(r"\[([A-Z]+)\] .+", text.strip())
    return text.strip() if match else None


def _generate_scene(client: OpenAI, premise: str) -> str:
    """Generate a complete scene via multiple rounds of API calls."""
    scene_lines: list[str] = []

    for _ in range(NUM_ROUNDS):
        context = "\n".join(scene_lines[-CONTEXT_LINES:]) if scene_lines else premise
        messages = [
            {"role": "system", "content": f"{SCRIPT_PROMPT}\n\nScene premise: {premise}"},
            {"role": "user", "content": context},
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=256,
                frequency_penalty=0.5, temperature=0.7,
            )
            text = response.choices[0].message.content or ""
            for line in text.strip().split("\n"):
                parsed = _parse_line(line)
                if parsed:
                    scene_lines.append(parsed)
        except Exception as e:
            scene_lines.append(f"[Error: {e}]")
            break

    return "\n".join(scene_lines)


def web(prevent_thread_lock: bool = False) -> None:
    client = OpenAI(base_url=SERVE_URL, api_key="not-needed", timeout=300)

    def generate(premise: str):  # type: ignore[no-untyped-def]
        if not premise.strip():
            yield "Enter a premise."
            return

        yield "Generating..."
        yield _generate_scene(client, premise)

    with gr.Blocks(theme=gr.themes.Monochrome(), title="Nothing-GPT") as demo:
        gr.Markdown("# Nothing-GPT\nGenerate a Seinfeld scene from a premise.")
        premise_input = gr.Textbox(
            label="Premise",
            placeholder="Jerry discovers his dry cleaner has been wearing his clothes...",
            lines=2,
        )
        generate_btn = gr.Button("Generate Scene", variant="primary")
        output = gr.Textbox(label="Scene", lines=20, interactive=False)

        generate_btn.click(
            fn=generate, inputs=premise_input, outputs=output,
        )

    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=prevent_thread_lock)


if __name__ == "__main__":
    web()
