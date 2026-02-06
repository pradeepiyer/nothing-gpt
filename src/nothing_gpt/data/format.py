"""Format parsed dialogue into prompt-completion training data for TRL SFTTrainer."""

import json
import random
from pathlib import Path

from nothing_gpt.characters import MAIN_CHARACTERS, get_system_prompt
from nothing_gpt.data.parse import DialogueTurn, Episode

PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
TRAINING_DIR = Path(__file__).parent.parent.parent.parent / "data" / "training"

CONTEXT_WINDOW = 8  # Max preceding turns to include as context
VAL_RATIO = 0.1  # 10% of episodes for validation


def format_context(turns: list[DialogueTurn]) -> str:
    """Format preceding turns as [CharName] Dialogue lines."""
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
    """Generate prompt-completion examples from an episode.

    For each main character's line, collect preceding turns as context
    and format as a prompt-completion pair.
    """
    turns = merge_consecutive_turns(episode.turns)
    examples: list[dict] = []

    for i, turn in enumerate(turns):
        if turn.character not in MAIN_CHARACTERS:
            continue

        # Collect preceding context (up to CONTEXT_WINDOW turns)
        start = max(0, i - CONTEXT_WINDOW)
        context_turns = turns[start:i]

        # Skip if no context (first line in episode)
        if not context_turns:
            continue

        context_text = format_context(context_turns)
        system_prompt = get_system_prompt(turn.character)

        example = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_text},
            ],
            "completion": [
                {"role": "assistant", "content": turn.dialogue},
            ],
        }
        examples.append(example)

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

    train_count = 0
    with open(train_path, "w") as f:
        for ep in train_eps:
            for example in episode_to_examples(ep):
                f.write(json.dumps(example) + "\n")
                train_count += 1

    val_count = 0
    with open(val_path, "w") as f:
        for ep in val_eps:
            for example in episode_to_examples(ep):
                f.write(json.dumps(example) + "\n")
                val_count += 1

    print(f"Training examples: {train_count}")
    print(f"Validation examples: {val_count}")
    print(f"Saved to {train_path} and {val_path}")
    return train_path, val_path


if __name__ == "__main__":
    format_dataset()
