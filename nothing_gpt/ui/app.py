"""Gradio scene generator UI for Nothing-GPT."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
from openai import OpenAI

from nothing_gpt.constants import SCRIPT_PROMPT

SERVE_URL = os.environ.get("SERVE_URL", "http://nothing-gpt-serve:8000/v1")
CONTEXT_LINES = 10  # Lines of scene to send as context for each API call
NUM_ROUNDS = 4  # Number of API calls to generate a full scene


def _parse_line(text: str) -> str | None:
    """Extract a [CHARACTER] dialogue line from model output."""
    match = re.match(r"\[([A-Z]+)\] .+", text.strip())
    return text.strip() if match else None


def _generate_scene(client: OpenAI, premise: str, model: str) -> str:
    """Generate a complete scene for a single model."""
    scene_lines: list[str] = []

    for _ in range(NUM_ROUNDS):
        context = "\n".join(scene_lines[-CONTEXT_LINES:]) if scene_lines else premise
        messages = [
            {"role": "system", "content": f"{SCRIPT_PROMPT}\n\nScene premise: {premise}"},
            {"role": "user", "content": context},
        ]

        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=256,
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

    def generate_comparison(premise: str):  # type: ignore[no-untyped-def]
        if not premise.strip():
            yield "Enter a premise.", "Enter a premise."
            return

        yield "Generating...", "Generating..."
        results = {"seinfeld": "", "seinfeld-dpo": ""}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_generate_scene, client, premise, model): model
                for model in results
            }
            for future in as_completed(futures):
                model = futures[future]
                results[model] = future.result()
                yield results["seinfeld"] or "Generating...", results["seinfeld-dpo"] or "Generating..."

    with gr.Blocks(theme=gr.themes.Monochrome(), title="Nothing-GPT") as demo:
        gr.Markdown("# Nothing-GPT\nGenerate a Seinfeld scene from a premise.")
        premise_input = gr.Textbox(
            label="Premise",
            placeholder="Jerry discovers his dry cleaner has been wearing his clothes...",
            lines=2,
        )
        generate_btn = gr.Button("Generate Scene", variant="primary")
        with gr.Row():
            with gr.Column():
                sft_output = gr.Textbox(label="Model A", lines=20, interactive=False)
            with gr.Column():
                dpo_output = gr.Textbox(label="Model B", lines=20, interactive=False)

        generate_btn.click(
            fn=generate_comparison, inputs=premise_input, outputs=[sft_output, dpo_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=prevent_thread_lock)


if __name__ == "__main__":
    web()
