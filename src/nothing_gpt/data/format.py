"""Format parsed dialogue into multi-turn conversation training data for TRL SFTTrainer."""

import json
import random
from pathlib import Path

from nothing_gpt.characters import MAIN_CHARACTERS, get_system_prompt
from nothing_gpt.data.parse import DialogueTurn, Episode

PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
TRAINING_DIR = Path(__file__).parent.parent.parent.parent / "data" / "training"

MAX_MESSAGES = 20  # Max messages (user + assistant) per training example
MIN_ASSISTANT_TURNS = 2  # Minimum assistant turns per example
STRIDE = 5  # Sliding window stride
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


def _turns_to_messages(
    turns: list[DialogueTurn], character: str
) -> list[dict[str, str]]:
    """Convert dialogue turns to user/assistant messages for a target character.

    Consecutive non-target turns are merged into a single user message.
    Target character turns become assistant messages.
    """
    messages: list[dict[str, str]] = []
    pending_user_lines: list[str] = []

    for turn in turns:
        if turn.character == character:
            if pending_user_lines:
                messages.append({"role": "user", "content": "\n".join(pending_user_lines)})
                pending_user_lines = []
            messages.append({"role": "assistant", "content": turn.dialogue})
        else:
            pending_user_lines.append(f"[{turn.character}] {turn.dialogue}")

    # Trailing user lines are dropped (no assistant response follows)
    return messages


def episode_to_examples(episode: Episode) -> list[dict]:
    """Generate multi-turn conversation examples from an episode.

    For each main character, convert the episode into alternating user/assistant
    messages, then extract sliding windows.
    """
    turns = merge_consecutive_turns(episode.turns)
    examples: list[dict] = []

    for character in MAIN_CHARACTERS:
        messages = _turns_to_messages(turns, character)

        if len(messages) < 2:
            continue

        system_prompt = get_system_prompt(character)

        # Sliding window over messages
        start = 0
        while start < len(messages):
            window = messages[start : start + MAX_MESSAGES]

            # Trim to start with user, end with assistant
            while window and window[0]["role"] != "user":
                window = window[1:]
            while window and window[-1]["role"] != "assistant":
                window = window[:-1]

            assistant_count = sum(1 for m in window if m["role"] == "assistant")
            if len(window) >= 2 and assistant_count >= MIN_ASSISTANT_TURNS:
                examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *window,
                    ]
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
