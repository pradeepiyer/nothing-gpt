import json
from pathlib import Path

import pandas as pd
import pytest

from nothing_gpt.data.parse import (
    DialogueTurn,
    Episode,
    clean_dialogue,
    normalize_character,
    parse_csv,
    save_dialogues,
)


class TestNormalizeCharacter:
    def test_main_characters_unchanged(self):
        assert normalize_character("JERRY") == "JERRY"
        assert normalize_character("GEORGE") == "GEORGE"
        assert normalize_character("ELAINE") == "ELAINE"
        assert normalize_character("KRAMER") == "KRAMER"

    def test_george_costanza_alias(self):
        assert normalize_character("GEORGE COSTANZA") == "GEORGE"

    def test_peterman_aliases(self):
        assert normalize_character("MR. PETERMAN") == "PETERMAN"
        assert normalize_character("J. PETERMAN") == "PETERMAN"

    def test_case_insensitive(self):
        assert normalize_character("jerry") == "JERRY"
        assert normalize_character("George Costanza") == "GEORGE"

    def test_whitespace_stripped(self):
        assert normalize_character("  JERRY  ") == "JERRY"

    def test_unknown_character_passthrough(self):
        assert normalize_character("LLOYD BRAUN") == "LLOYD BRAUN"


class TestCleanDialogue:
    def test_removes_stage_directions(self):
        assert clean_dialogue("Hello (waves) there") == "Hello there"

    def test_removes_multiple_stage_directions(self):
        assert clean_dialogue("(entering) Hey (pause) buddy") == "Hey buddy"

    def test_preserves_normal_dialogue(self):
        assert clean_dialogue("What is the deal with airline food?") == (
            "What is the deal with airline food?"
        )

    def test_collapses_whitespace(self):
        assert clean_dialogue("Hello   there   buddy") == "Hello there buddy"

    def test_empty_after_cleaning(self):
        assert clean_dialogue("(exits)") == ""

    def test_non_string_returns_empty(self):
        assert clean_dialogue(None) == ""  # type: ignore[arg-type]
        assert clean_dialogue(42) == ""  # type: ignore[arg-type]


class TestParseCSV:
    def _make_csv(self, rows: list[dict], tmp_path: Path) -> Path:
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "scripts.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def _row(
        self, char: str, dialogue: str, ep: int = 1, seid: str = "S01E01", season: int = 1
    ) -> dict:
        return {
            "Character": char,
            "Dialogue": dialogue,
            "EpisodeNo": ep,
            "SEID": seid,
            "Season": season,
        }

    def test_basic_parsing(self, tmp_path: Path):
        rows = [self._row("JERRY", "Hello"), self._row("GEORGE", "Hey")]
        csv_path = self._make_csv(rows, tmp_path)
        episodes = parse_csv(csv_path)

        assert len(episodes) == 1
        assert episodes[0].episode_id == "S01E01"
        assert len(episodes[0].turns) == 2
        assert episodes[0].turns[0].character == "JERRY"
        assert episodes[0].turns[0].dialogue == "Hello"

    def test_character_normalization(self, tmp_path: Path):
        rows = [self._row("GEORGE COSTANZA", "Hey")]
        csv_path = self._make_csv(rows, tmp_path)
        episodes = parse_csv(csv_path)
        assert episodes[0].turns[0].character == "GEORGE"

    def test_stage_directions_stripped(self, tmp_path: Path):
        rows = [self._row("KRAMER", "(bursts in) Hey buddy!")]
        csv_path = self._make_csv(rows, tmp_path)
        episodes = parse_csv(csv_path)
        assert episodes[0].turns[0].dialogue == "Hey buddy!"

    def test_multiple_episodes(self, tmp_path: Path):
        rows = [
            self._row("JERRY", "Hello", ep=1, seid="S01E01"),
            self._row("JERRY", "Goodbye", ep=2, seid="S01E02"),
        ]
        csv_path = self._make_csv(rows, tmp_path)
        episodes = parse_csv(csv_path)
        assert len(episodes) == 2

    def test_empty_dialogue_skipped(self, tmp_path: Path):
        rows = [self._row("JERRY", "(exits)"), self._row("GEORGE", "Hey")]
        csv_path = self._make_csv(rows, tmp_path)
        episodes = parse_csv(csv_path)
        assert len(episodes[0].turns) == 1
        assert episodes[0].turns[0].character == "GEORGE"

    def test_missing_columns_raises(self, tmp_path: Path):
        df = pd.DataFrame({"Character": ["JERRY"], "Dialogue": ["Hi"]})
        csv_path = tmp_path / "scripts.csv"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            parse_csv(csv_path)


class TestSaveDialogues:
    def test_roundtrip(self, tmp_path: Path):
        episodes = [
            Episode(
                episode_id="S01E01",
                season=1,
                episode_no=1,
                turns=[
                    DialogueTurn(character="JERRY", dialogue="Hello"),
                    DialogueTurn(character="GEORGE", dialogue="Hey"),
                ],
            ),
        ]
        output_path = save_dialogues(episodes, tmp_path)

        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 1
        assert records[0]["episode_id"] == "S01E01"
        assert len(records[0]["turns"]) == 2
        assert records[0]["turns"][0]["character"] == "JERRY"
