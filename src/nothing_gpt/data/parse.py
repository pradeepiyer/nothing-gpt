"""Parse scripts CSV into structured dialogue sequences per episode."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"

# Canonical names for main characters
CHARACTER_ALIASES: dict[str, str] = {
    "GEORGE COSTANZA": "GEORGE",
    "ESTELLE COSTANZA": "ESTELLE",
    "FRANK COSTANZA": "FRANK",
    "MORTY SEINFELD": "MORTY",
    "HELEN SEINFELD": "HELEN",
    "UNCLE LEO": "UNCLE LEO",
    "MR. STEINBRENNER": "STEINBRENNER",
    "MR. KRUGER": "KRUGER",
    "MR. PETERMAN": "PETERMAN",
    "J. PETERMAN": "PETERMAN",
    "MR. PITT": "PITT",
    "MR. LIPPMAN": "LIPPMAN",
    "JACOPO PETERMAN": "PETERMAN",
    "SUSAN ROSS": "SUSAN",
    "SUSAN": "SUSAN",
    "PUDDY": "PUDDY",
    "DAVID PUDDY": "PUDDY",
    "NEWMAN": "NEWMAN",
}

# Stage direction pattern: text enclosed in parentheses (possibly spanning content)
STAGE_DIRECTION_RE = re.compile(r"\([^)]*\)")

# Allowlist for valid character names: uppercase letters, digits, spaces, periods,
# apostrophes, hyphens, and # (for "MAN #1" style names)
VALID_NAME_RE = re.compile(r"^[A-Z][A-Z0-9 .'\-#]*$")

# Words that pass the regex but are stage directions or script metadata, not characters
STAGE_DIRECTION_WORDS: frozenset[str] = frozenset({
    "ANNOYED", "CHUCKLING", "DELIGHTEDLY", "EXASPERATED", "EXITS",
    "FURIOUS", "GASPS", "HANDS", "LAUGHING", "MIND", "MUTTERING",
    "PANTING", "PAUSE", "PERKILY", "SARCASTICALLY", "SHRUGS", "SIGHS",
    "WHISPERED", "WHISPERING", "NO REACTION", "YEAH",
    "DEDICATION", "HTTP", "PUBLISHED", "SCENE",
    "OPENING SCENE", "PERFORMED BY", "SO FAR",
    "SONG OVER THE END CREDITS", "SUNG BY",
})


@dataclass
class DialogueTurn:
    character: str
    dialogue: str


@dataclass
class Episode:
    episode_id: str
    season: int
    episode_no: int
    turns: list[DialogueTurn]


def normalize_character(name: str) -> str:
    """Normalize character name to canonical form."""
    name = name.strip().upper()
    name = STAGE_DIRECTION_RE.sub("", name).strip()
    return CHARACTER_ALIASES.get(name, name)


def clean_dialogue(text: str) -> str:
    """Strip stage directions and clean up whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove stage directions in parentheses
    cleaned = STAGE_DIRECTION_RE.sub("", text)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_csv(csv_path: Path = RAW_DIR / "scripts.csv") -> list[Episode]:
    """Parse the scripts CSV into episode dialogue sequences."""
    df = pd.read_csv(csv_path)

    # Expected columns: Character, Dialogue, EpisodeNo, SEID, Season
    required_cols = {"Character", "Dialogue", "EpisodeNo", "SEID", "Season"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    episodes: list[Episode] = []

    for seid, group in df.groupby("SEID", sort=False):
        group = group.sort_index()  # Preserve original row order
        turns: list[DialogueTurn] = []

        for _, row in group.iterrows():
            character = row["Character"]
            dialogue = row["Dialogue"]

            if not isinstance(character, str) or not isinstance(dialogue, str):
                continue

            character = normalize_character(character)

            if (
                not VALID_NAME_RE.match(character)
                or len(character) > 25
                or character in STAGE_DIRECTION_WORDS
            ):
                continue

            dialogue = clean_dialogue(dialogue)

            if not dialogue:
                continue

            turns.append(DialogueTurn(character=character, dialogue=dialogue))

        if turns:
            season = int(group["Season"].iloc[0])
            episode_no = int(group["EpisodeNo"].iloc[0])
            episodes.append(Episode(
                episode_id=str(seid),
                season=season,
                episode_no=episode_no,
                turns=turns,
            ))

    return episodes


def save_dialogues(episodes: list[Episode], output_dir: Path = PROCESSED_DIR) -> Path:
    """Save parsed episodes to JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dialogues.jsonl"

    with open(output_path, "w") as f:
        for ep in episodes:
            record = {
                "episode_id": ep.episode_id,
                "season": ep.season,
                "episode_no": ep.episode_no,
                "turns": [
                    {"character": t.character, "dialogue": t.dialogue}
                    for t in ep.turns
                ],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(episodes)} episodes to {output_path}")
    return output_path


if __name__ == "__main__":
    episodes = parse_csv()
    save_dialogues(episodes)
