"""Format parsed dialogue into script-continuation training data for TRL SFTTrainer."""

import json
import random
import statistics
from pathlib import Path

import tiktoken

from nothing_gpt.characters import SCRIPT_PROMPT
from nothing_gpt.data.parse import DialogueTurn, Episode

PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
TRAINING_DIR = Path(__file__).parent.parent.parent.parent / "data" / "training"

CONTEXT_TURNS = 5  # Turns given as user prompt (context)
WINDOW_SIZE = 35  # Total turns per window (context + completion)
MIN_COMPLETION_TURNS = 10  # Minimum turns in the assistant completion
STRIDE = 5  # Sliding window stride
VAL_RATIO = 0.1  # 10% of episodes for validation
MAX_LENGTH = 2048  # Training max_length — examples beyond this get truncated

# Approximate overhead from chat template role markers (system/user/assistant tokens)
CHAT_TEMPLATE_OVERHEAD = 20


def print_token_stats(examples: list[dict]) -> None:
    """Print token length statistics for training examples using tiktoken as a proxy.

    Uses cl100k_base (GPT-4 tokenizer) which tracks within ~5-10% of Llama's tokenizer
    for English text. Adds a fixed overhead estimate for chat template role markers.
    """
    if not examples:
        print("No examples to analyze.")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    token_counts: list[int] = []
    for ex in examples:
        text = "".join(msg["content"] for msg in ex["prompt"] + ex["completion"])
        token_counts.append(len(enc.encode(text)) + CHAT_TEMPLATE_OVERHEAD)

    token_counts.sort()
    n = len(token_counts)
    exceeding = sum(1 for c in token_counts if c > MAX_LENGTH)

    print(f"\nToken length statistics ({n} examples, tiktoken cl100k proxy):")
    print(f"  Min: {token_counts[0]}, Max: {token_counts[-1]}, "
          f"Mean: {statistics.mean(token_counts):.0f}")
    if n >= 2:
        quantiles = statistics.quantiles(token_counts, n=100)
        print(f"  P50: {quantiles[49]}, P90: {quantiles[89]}, "
              f"P95: {quantiles[94]}, P99: {quantiles[98]}")
    print(f"  Exceeding {MAX_LENGTH} tokens: {exceeding} ({exceeding / n * 100:.1f}%)")


def format_context(turns: list[DialogueTurn]) -> str:
    """Format turns as [CharName] Dialogue lines."""
    return "\n".join(f"[{t.character}] {t.dialogue}" for t in turns)


def merge_consecutive_turns(turns: list[DialogueTurn]) -> list[DialogueTurn]:
    """Merge consecutive turns by the same character into single turns."""
    if not turns:
        return []

    merged: list[DialogueTurn] = [
        DialogueTurn(character=turns[0].character, dialogue=turns[0].dialogue)
    ]
    for turn in turns[1:]:
        if turn.character == merged[-1].character:
            merged[-1].dialogue += " " + turn.dialogue
        else:
            merged.append(DialogueTurn(character=turn.character, dialogue=turn.dialogue))
    return merged


def episode_to_examples(episode: Episode) -> list[dict]:
    """Generate script-continuation examples from an episode.

    Each example has a system prompt, a user message with context lines,
    and an assistant message with continuation lines.
    """
    turns = merge_consecutive_turns(episode.turns)

    examples: list[dict] = []
    start = 0
    while start + CONTEXT_TURNS + MIN_COMPLETION_TURNS <= len(turns):
        window = turns[start : start + WINDOW_SIZE]
        context = window[:CONTEXT_TURNS]
        completion = window[CONTEXT_TURNS:]

        if len(completion) >= MIN_COMPLETION_TURNS:
            examples.append({
                "prompt": [
                    {"role": "system", "content": SCRIPT_PROMPT},
                    {"role": "user", "content": format_context(context)},
                ],
                "completion": [
                    {"role": "assistant", "content": format_context(completion)},
                ],
            })

        start += STRIDE

    return examples


def split_episodes_by_id(
    episodes: list[Episode],
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode]]:
    """Split episodes into train/val sets. Split by episode to avoid data leakage."""
    rng = random.Random(seed)
    episode_ids = sorted({ep.episode_id for ep in episodes})
    rng.shuffle(episode_ids)

    n_val = max(1, int(len(episode_ids) * val_ratio))
    val_ids = set(episode_ids[:n_val])

    train = [ep for ep in episodes if ep.episode_id not in val_ids]
    val = [ep for ep in episodes if ep.episode_id in val_ids]
    return train, val


def format_dataset(
    dialogues_path: Path = PROCESSED_DIR / "dialogues.jsonl",
    output_dir: Path = TRAINING_DIR,
    val_ratio: float = VAL_RATIO,
) -> tuple[Path, Path]:
    """Load parsed dialogues and produce train/val JSONL files."""
    # Load episodes
    episodes: list[Episode] = []
    with open(dialogues_path) as f:
        for line in f:
            record = json.loads(line)
            turns = [
                DialogueTurn(character=t["character"], dialogue=t["dialogue"])
                for t in record["turns"]
            ]
            episodes.append(Episode(
                episode_id=record["episode_id"],
                season=record["season"],
                episode_no=record["episode_no"],
                turns=turns,
            ))

    train_eps, val_eps = split_episodes_by_id(episodes, val_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    all_examples: list[dict] = []

    train_count = 0
    with open(train_path, "w") as f:
        for ep in train_eps:
            for example in episode_to_examples(ep):
                f.write(json.dumps(example) + "\n")
                all_examples.append(example)
                train_count += 1

    val_count = 0
    with open(val_path, "w") as f:
        for ep in val_eps:
            for example in episode_to_examples(ep):
                f.write(json.dumps(example) + "\n")
                all_examples.append(example)
                val_count += 1

    print(f"Training examples: {train_count}")
    print(f"Validation examples: {val_count}")
    print(f"Saved to {train_path} and {val_path}")
    print_token_stats(all_examples)
    return train_path, val_path


if __name__ == "__main__":
    format_dataset()
