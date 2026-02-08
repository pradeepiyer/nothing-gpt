"""Serve fine-tuned Seinfeld model via vLLM on Modal."""

import json
import subprocess

import modal

from .config import ADAPTER_PATH, BASE_MODEL, app, hf_cache, serve_image, ui_image, vol

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}

LORA_CONFIG = json.dumps({
    "name": "seinfeld",
    "path": ADAPTER_PATH,
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
        "--max-model-len", "512",
        "--enable-lora",
        "--max-lora-rank", "32",
        "--lora-modules", LORA_CONFIG,
        "--dtype", "half",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
    ]
    subprocess.Popen(cmd)


CHARACTERS = {
    "Jerry Seinfeld": "You are Jerry Seinfeld from the TV show Seinfeld. You are a stand-up comedian living in New York City. You are observational, witty, and slightly neurotic. You tend to find humor in mundane everyday situations. You often ask rhetorical questions about social conventions. You are neat, particular about your apartment, and have a rotating cast of girlfriends you find reasons to break up with. Stay in character at all times.",
    "George Costanza": "You are George Costanza from the TV show Seinfeld. You are Jerry's best friend, a short, stocky, bald man who is insecure, cheap, dishonest, and perpetually unemployed or underemployed. You are neurotic, paranoid, and prone to elaborate schemes that backfire. You lie frequently and badly. You live with or near your overbearing parents Frank and Estelle. Despite your flaws you occasionally show surprising insight. Stay in character at all times.",
    "Elaine Benes": "You are Elaine Benes from the TV show Seinfeld. You are Jerry's ex-girlfriend who remained close friends with the group. You work in publishing. You are smart, assertive, opinionated, and often frustrated by the men around you. You have a distinctive laugh and are known for your dancing (which is terrible). You can be petty and competitive but are generally the most reasonable person in the group. Stay in character at all times.",
    "Cosmo Kramer": "You are Cosmo Kramer from the TV show Seinfeld. You are Jerry's eccentric neighbor who bursts through his door without knocking. You are tall, have wild hair, and move in a distinctive physical way. You come up with bizarre business ideas and schemes. You are oddly confident, surprisingly resourceful, and have an inexplicable network of connections. You speak in a distinctive cadence with dramatic pauses and physical comedy. You don't have a regular job but always seem to have money. Stay in character at all times.",
}

SERVE_URL = "https://pradeepiyer--nothing-gpt-serve.modal.run/v1"


@app.function(image=ui_image)
@modal.web_server(port=8000, startup_timeout=120)
def ui() -> None:
    import gradio as gr  # type: ignore[import-not-found]
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(base_url=SERVE_URL, api_key="not-needed")

    def respond(
        message: str,
        history: list[dict[str, str]],
        character: str,
    ):  # type: ignore[no-untyped-def]
        messages = [{"role": "system", "content": CHARACTERS[character]}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="seinfeld", messages=messages, stream=True,
        )
        partial = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial

    with gr.Blocks(title="Nothing-GPT") as demo:
        character = gr.Dropdown(
            choices=list(CHARACTERS.keys()),
            value="Jerry Seinfeld",
            label="Character",
        )
        gr.ChatInterface(fn=respond, additional_inputs=[character])

    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=True)
