"""Generate multiple completions per prompt for DPO preference judging."""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

from nothing_gpt.constants import DATA_PATH, DPO_DATA_PATH

SERVE_URL = os.environ.get("SERVE_URL", "http://nothing-gpt-serve:8000/v1")
NUM_COMPLETIONS = 5
TEMPERATURE = 0.9
BATCH_SIZE = 32


def _parse_line(text: str) -> str | None:
    """Extract a [CHARACTER] dialogue line from model output."""
    match = re.match(r"\[([A-Z]+)\] .+", text.strip())
    return text.strip() if match else None


def load_prompts(data_dir: str | None = None) -> list[list[dict]]:
    """Load prompt message lists from train.jsonl and val.jsonl in data_dir."""
    directory = data_dir or DATA_PATH
    prompts: list[list[dict]] = []
    for filename in ("train.jsonl", "val.jsonl"):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                prompts.append(row["prompt"])
    return prompts


async def generate_completion(client: AsyncOpenAI, prompt: list[dict]) -> str:
    """Generate a single completion, filtering to valid [CHARACTER] lines."""
    response = await client.chat.completions.create(
        model="seinfeld",
        messages=prompt,
        max_tokens=256,
        temperature=TEMPERATURE,
        frequency_penalty=0.5,
    )
    text = response.choices[0].message.content or ""
    lines = [_parse_line(line) for line in text.strip().split("\n")]
    return "\n".join(line for line in lines if line)


async def generate_for_prompt(client: AsyncOpenAI, prompt: list[dict]) -> list[str]:
    """Generate NUM_COMPLETIONS completions for a single prompt."""
    tasks = [generate_completion(client, prompt) for _ in range(NUM_COMPLETIONS)]
    return list(await asyncio.gather(*tasks))


async def _run(data_dir: str | None = None, output_dir: str | None = None) -> None:
    out_dir = output_dir or DPO_DATA_PATH
    os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(base_url=SERVE_URL, api_key="not-needed", timeout=300)
    prompts = load_prompts(data_dir)
    total = len(prompts)

    # Resume from existing output
    output_path = os.path.join(out_dir, "completions.jsonl")
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            already_done = sum(1 for _ in f)
    if already_done:
        print(f"Resuming from {already_done}/{total}")

    with open(output_path, "a") as f:
        for batch_start in range(already_done, total, BATCH_SIZE):
            batch = prompts[batch_start : batch_start + BATCH_SIZE]
            tasks = [generate_for_prompt(client, prompt) for prompt in batch]
            results = await asyncio.gather(*tasks)

            for prompt, completions in zip(batch, results):
                row = {"prompt": prompt, "completions": completions}
                f.write(json.dumps(row) + "\n")
            f.flush()

            done = min(batch_start + BATCH_SIZE, total)
            print(f"[{done}/{total}] prompts completed")

    print(f"Wrote {total} rows to {output_path}")
    await client.close()


def generate_pairs(data_dir: str | None = None, output_dir: str | None = None) -> None:
    """Generate NUM_COMPLETIONS completions per prompt and write completions.jsonl."""
    asyncio.run(_run(data_dir, output_dir))


if __name__ == "__main__":
    generate_pairs()
