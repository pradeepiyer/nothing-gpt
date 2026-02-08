"""Gradio chat UI served on Modal."""

import modal

from .config import ui_image

app = modal.App("nothing-gpt")

CHARACTERS = {
    "Jerry Seinfeld": "You are Jerry Seinfeld from the TV show Seinfeld. You are a stand-up comedian living in New York City. You are observational, witty, and slightly neurotic. You tend to find humor in mundane everyday situations. You often ask rhetorical questions about social conventions. You are neat, particular about your apartment, and have a rotating cast of girlfriends you find reasons to break up with. Stay in character at all times.",
    "George Costanza": "You are George Costanza from the TV show Seinfeld. You are Jerry's best friend, a short, stocky, bald man who is insecure, cheap, dishonest, and perpetually unemployed or underemployed. You are neurotic, paranoid, and prone to elaborate schemes that backfire. You lie frequently and badly. You live with or near your overbearing parents Frank and Estelle. Despite your flaws you occasionally show surprising insight. Stay in character at all times.",
    "Elaine Benes": "You are Elaine Benes from the TV show Seinfeld. You are Jerry's ex-girlfriend who remained close friends with the group. You work in publishing. You are smart, assertive, opinionated, and often frustrated by the men around you. You have a distinctive laugh and are known for your dancing (which is terrible). You can be petty and competitive but are generally the most reasonable person in the group. Stay in character at all times.",
    "Cosmo Kramer": "You are Cosmo Kramer from the TV show Seinfeld. You are Jerry's eccentric neighbor who bursts through his door without knocking. You are tall, have wild hair, and move in a distinctive physical way. You come up with bizarre business ideas and schemes. You are oddly confident, surprisingly resourceful, and have an inexplicable network of connections. You speak in a distinctive cadence with dramatic pauses and physical comedy. You don't have a regular job but always seem to have money. Stay in character at all times.",
}

SERVE_URL = "https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1"


@app.function(image=ui_image, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=120)
def web() -> None:
    import gradio as gr  # type: ignore[import-not-found]
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(base_url=SERVE_URL, api_key="not-needed", timeout=300)

    def respond(
        message: str,
        history: list[dict[str, str]],
        character: str,
    ):  # type: ignore[no-untyped-def]
        messages = [{"role": "system", "content": CHARACTERS[character]}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        try:
            response = client.chat.completions.create(
                model="seinfeld", messages=messages, stream=True,
                max_tokens=256,
            )
            partial = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    partial += chunk.choices[0].delta.content
                    yield partial
        except Exception as e:
            yield f"Error: {e}"

    demo = gr.ChatInterface(
        fn=respond,
        title="Nothing-GPT",
        chatbot=gr.Chatbot(
            placeholder="<strong>Nothing-GPT</strong><br>Pick a character and say something.",
            height=550,
            layout="bubble",
            type="messages",
        ),
        textbox=gr.Textbox(
            placeholder="Say something to Jerry...",
            container=False,
            scale=7,
        ),
        additional_inputs=[
            gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="Jerry Seinfeld",
                label="Character",
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="", open=True),
        type="messages",
        theme=gr.themes.Monochrome(),
    )
    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=True)
