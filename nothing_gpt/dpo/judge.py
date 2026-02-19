"""Preference judging for DPO training pairs."""

import asyncio
import json
import os
import random
import re

from openai import AsyncOpenAI, RateLimitError

from nothing_gpt.constants import DPO_DATA_PATH

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-pro")
VAL_RATIO = 0.1
SEED = 42
MAX_CONCURRENT = 50
FLUSH_INTERVAL = 32
MIN_CONFIDENCE = 3
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


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
    client: AsyncOpenAI, sem: asyncio.Semaphore, completions: list[str]
) -> tuple[int, int, int] | None:
    """Rank completions via LLM judge. Returns (best_idx, worst_idx, confidence) or None on failure."""
    prompt = build_judge_prompt(completions)
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                )
            choice = response.choices[0]
            text = (choice.message.content if choice.message else None) or ""
            if not text.strip():
                raise ValueError(f"Empty response from model (finish_reason={choice.finish_reason})")
            # Strip markdown code fences if present (e.g. ```json ... ```)
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1)
            ranking = json.loads(text)
            best_idx = int(ranking["best"]) - 1
            worst_idx = int(ranking["worst"]) - 1
            n = len(completions)
            if not (0 <= best_idx < n and 0 <= worst_idx < n):
                raise ValueError(f"Index out of range: best={best_idx}, worst={worst_idx}, n={n}")
            if best_idx == worst_idx:
                raise ValueError(f"best and worst are the same index: {best_idx}")
            return best_idx, worst_idx, int(ranking["confidence"])
        except RateLimitError as e:
            delay = 30 * (attempt + 1)
            print(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES}): {e}", flush=True)
            await asyncio.sleep(delay)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to parse judge response after {MAX_RETRIES} attempts: {e}", flush=True)
                return None
            delay = RETRY_BASE_DELAY * (2**attempt)
            print(f"Parse error, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES}): {e}", flush=True)
            await asyncio.sleep(delay)
    print(f"Failed after {MAX_RETRIES} attempts (rate limited)", flush=True)
    return None


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

    client = AsyncOpenAI(timeout=300)

    completions_path = os.path.join(data_dir, "completions.jsonl")
    rows: list[dict] = []
    with open(completions_path) as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Resume from existing output
    judged_path = os.path.join(out_dir, "judged.jsonl")
    already_done = 0
    if os.path.exists(judged_path):
        with open(judged_path) as f:
            already_done = sum(1 for _ in f)
    if already_done:
        print(f"Resuming from {already_done}/{total}", flush=True)

    print(f"Judging {total} rows with {JUDGE_MODEL}")

    completed = 0
    buffer: list[str] = []

    async def process_row(row: dict) -> None:
        nonlocal completed
        ranking = await rank_completions(client, sem, row["completions"])
        if ranking is None:
            buffer.append(json.dumps({"filtered": True}))
        else:
            best_idx, worst_idx, confidence = ranking
            if confidence < MIN_CONFIDENCE:
                buffer.append(json.dumps({"filtered": True}))
            else:
                pair = build_preference_pair(row["prompt"], row["completions"], best_idx, worst_idx)
                buffer.append(json.dumps(pair))
        completed += 1
        if completed % FLUSH_INTERVAL == 0:
            with open(judged_path, "a") as f:
                f.write("\n".join(buffer) + "\n")
            buffer.clear()
            print(f"[{already_done + completed}/{total}] judged", flush=True)

    await asyncio.gather(*(process_row(row) for row in rows[already_done:]))
    if buffer:
        with open(judged_path, "a") as f:
            f.write("\n".join(buffer) + "\n")
    print(f"[{total}/{total}] judged", flush=True)

    # Load all judged pairs, filter, shuffle, and split
    pairs: list[dict] = []
    filtered_total = 0
    with open(judged_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("filtered"):
                filtered_total += 1
            else:
                pairs.append(row)

    print(f"Filtered {filtered_total}/{total} low-confidence pairs (confidence < {MIN_CONFIDENCE})")

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
