"""Preference judging for DPO training pairs."""

import asyncio
import json
import os
import random
import re

from openai import AsyncOpenAI, RateLimitError

from nothing_gpt.constants import DPO_DATA_PATH

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
VAL_RATIO = 0.1
SEED = 42
MAX_CONCURRENT = 10
FLUSH_INTERVAL = 32
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
                    max_tokens=8192,
                )
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise ValueError(f"Empty response from model (finish_reason={response.choices[0].finish_reason})")
            # Strip markdown code fences if present (e.g. ```json ... ```)
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1)
            ranking = json.loads(text)
            return int(ranking["best"]) - 1, int(ranking["worst"]) - 1, int(ranking["confidence"])
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

    client = AsyncOpenAI(timeout=120)

    completions_path = os.path.join(data_dir, "completions.jsonl")
    rows: list[dict] = []
    with open(completions_path) as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    pairs: list[dict] = []
    filtered = 0
    completed = 0
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    print(f"Judging {total} rows with {JUDGE_MODEL}")

    async def process_row(row: dict) -> None:
        nonlocal completed, filtered
        ranking = await rank_completions(client, sem, row["completions"])
        if ranking is None:
            filtered += 1
        else:
            best_idx, worst_idx, confidence = ranking
            if confidence < MIN_CONFIDENCE:
                filtered += 1
            else:
                pair = build_preference_pair(row["prompt"], row["completions"], best_idx, worst_idx)
                pairs.append(pair)
        completed += 1
        if completed % FLUSH_INTERVAL == 0:
            print(f"[{completed}/{total}] judged", flush=True)

    await asyncio.gather(*(process_row(row) for row in rows))
    print(f"[{completed}/{total}] judged", flush=True)
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
