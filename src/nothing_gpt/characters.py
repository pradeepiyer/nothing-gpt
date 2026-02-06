import json
from pathlib import Path

CHARACTERS_FILE = Path(__file__).parent.parent.parent / "configs" / "characters.json"

MAIN_CHARACTERS = frozenset({"JERRY", "GEORGE", "ELAINE", "KRAMER"})


def load_characters() -> dict[str, dict[str, str]]:
    with open(CHARACTERS_FILE) as f:
        return json.load(f)


def get_system_prompt(character: str) -> str:
    characters = load_characters()
    key = character.upper()
    if key not in characters:
        raise ValueError(f"Unknown character: {character}. Choose from: {sorted(characters)}")
    return characters[key]["system_prompt"]
