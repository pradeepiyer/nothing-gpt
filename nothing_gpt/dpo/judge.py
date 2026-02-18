"""Preference judging for DPO training pairs."""

import asyncio
import json
import os
import random
import re

from openai import AsyncOpenAI

from nothing_gpt.constants import DPO_DATA_PATH

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-pro")
VAL_RATIO = 0.1
SEED = 42
BATCH_SIZE = 16
MIN_CONFIDENCE = 3


def build_judge_prompt(completions: list[str]) -> str:
    """Build a judge prompt for N completions."""
    n = len(completions)
    completion_blocks = "\n\n".join(
        f"Completion {i + 1}:\n{c}" for i, c in enumerate(completions)
    )
    range_str = "|".join(str(i + 1) for i in range(n))
    return f"""\
You are evaluating Seinfeld script completions. Below are {n} completions for the same prompt.

Rate them on faithfulness to Seinfeld scripts and characters:
- Does the dialogue sound like it belongs in an actual Seinfeld episode?
- Does each character speak in their distinctive voice (Jerry's observational style, \
George's neurotic complaints, Elaine's sarcasm, Kramer's physicality and wild ideas)?
- Is the dialogue funny, coherent, and relevant to the scene context?

Respond with ONLY a JSON object: {{"best": <{range_str}>, "worst": <{range_str}>, "confidence": <1-5>}}
"confidence" rates how distinguishable the best and worst completions are \
(1 = nearly identical quality, 5 = clear quality difference).
Do not include any other text.

{completion_blocks}"""


async def rank_completions(
    client: AsyncOpenAI, completions: list[str]
) -> tuple[int, int, int]:
    """Rank completions via LLM judge. Returns (best_idx, worst_idx, confidence)."""
    prompt = build_judge_prompt(completions)
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    text = response.choices[0].message.content or ""
    # Strip markdown code fences if present (e.g. ```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    ranking = json.loads(text)
    return int(ranking["best"]) - 1, int(ranking["worst"]) - 1, int(ranking["confidence"])


def build_preference_pair(
    prompt: list[dict], completions: list[str], best_idx: int, worst_idx: int
) -> dict:
    """Build a TRL DPOTrainer preference pair in conversational format."""
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": completions[best_idx]}],
        "rejected": [{"role": "assistant", "content": completions[worst_idx]}],
    }


async def _run(input_dir: str | None = None, output_dir: str | None = None) -> None:
    data_dir = input_dir or DPO_DATA_PATH
    out_dir = output_dir or DPO_DATA_PATH
    os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(timeout=120)

    completions_path = os.path.join(data_dir, "completions.jsonl")
    rows: list[dict] = []
    with open(completions_path) as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    pairs: list[dict] = []
    filtered = 0
    print(f"Judging {total} rows with {JUDGE_MODEL}")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        tasks = [rank_completions(client, row["completions"]) for row in batch]
        rankings = await asyncio.gather(*tasks)

        for row, (best_idx, worst_idx, confidence) in zip(batch, rankings):
            if confidence < MIN_CONFIDENCE:
                filtered += 1
                continue
            pair = build_preference_pair(row["prompt"], row["completions"], best_idx, worst_idx)
            pairs.append(pair)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"[{done}/{total}] judged")

    print(f"Filtered {filtered}/{total} low-confidence pairs (confidence < {MIN_CONFIDENCE})")

    # Shuffle and split
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")

    with open(val_path, "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    print(f"Saved to {train_path} and {val_path}")
    await client.close()


def judge(input_dir: str | None = None, output_dir: str | None = None) -> None:
    """Read completions, rank with LLM judge, write train/val preference pairs."""
    asyncio.run(_run(input_dir, output_dir))


if __name__ == "__main__":
    judge()
