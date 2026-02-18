import json
import os
import random
import tempfile

from nothing_gpt.constants import SCRIPT_PROMPT
from nothing_gpt.dpo.generate_pairs import (  # type: ignore[import-not-found]
    _parse_line,
    load_prompts,
)
from nothing_gpt.dpo.judge import build_preference_pair  # type: ignore[import-not-found]


class TestParseLine:
    def test_valid_character_line(self):
        assert _parse_line("[JERRY] Hello there!") == "[JERRY] Hello there!"

    def test_valid_with_leading_whitespace(self):
        assert _parse_line("  [GEORGE] What is this?  ") == "[GEORGE] What is this?"

    def test_rejects_lowercase_character(self):
        assert _parse_line("[jerry] Hello") is None

    def test_rejects_no_brackets(self):
        assert _parse_line("JERRY: Hello") is None

    def test_rejects_empty_dialogue(self):
        assert _parse_line("[JERRY] ") is None

    def test_rejects_stage_direction(self):
        assert _parse_line("(Jerry enters)") is None

    def test_rejects_empty_string(self):
        assert _parse_line("") is None


class TestLoadPrompts:
    def test_loads_prompts_from_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            prompt = [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": "[JERRY] Hello"},
            ]
            completion = [{"role": "assistant", "content": "[GEORGE] Hey"}]
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
            path = f.name

        try:
            prompts = load_prompts(path)
            assert len(prompts) == 2
            assert prompts[0] == prompt
            assert prompts[0][0]["role"] == "system"
            assert prompts[0][1]["role"] == "user"
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            assert load_prompts(path) == []
        finally:
            os.unlink(path)


class TestBuildPreferencePair:
    def test_correct_format(self):
        prompt = [
            {"role": "system", "content": SCRIPT_PROMPT},
            {"role": "user", "content": "[JERRY] Hello"},
        ]
        completions = [
            "[JERRY] Line A\n[GEORGE] Response A",
            "[JERRY] Line B\n[GEORGE] Response B",
            "[JERRY] Line C\n[GEORGE] Response C",
        ]
        pair = build_preference_pair(prompt, completions, best_idx=0, worst_idx=2)
        assert pair["prompt"] == prompt
        assert pair["chosen"] == [{"role": "assistant", "content": completions[0]}]
        assert pair["rejected"] == [{"role": "assistant", "content": completions[2]}]

    def test_all_fields_present(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c"]
        pair = build_preference_pair(prompt, completions, best_idx=1, worst_idx=0)
        assert set(pair.keys()) == {"prompt", "chosen", "rejected"}

    def test_chosen_rejected_are_message_lists(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c"]
        pair = build_preference_pair(prompt, completions, best_idx=2, worst_idx=1)
        assert isinstance(pair["chosen"], list)
        assert isinstance(pair["rejected"], list)
        assert pair["chosen"][0]["role"] == "assistant"
        assert pair["rejected"][0]["role"] == "assistant"


class TestJudgeSplitLogic:
    """Test the shuffle + split logic used in judge.py."""

    def test_split_ratio(self):
        pairs = [{"prompt": [], "chosen": [], "rejected": []} for _ in range(100)]
        rng = random.Random(42)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * 0.1))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        assert len(val_pairs) == 10
        assert len(train_pairs) == 90

    def test_split_small_dataset(self):
        pairs = [{"prompt": [], "chosen": [], "rejected": []} for _ in range(5)]
        rng = random.Random(42)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * 0.1))
        # max(1, 0) = 1 — always at least 1 val example
        assert n_val == 1

    def test_deterministic_with_seed(self):
        pairs = list(range(20))
        rng1 = random.Random(42)
        rng1.shuffle(pairs)
        order1 = pairs[:]

        pairs = list(range(20))
        rng2 = random.Random(42)
        rng2.shuffle(pairs)
        order2 = pairs[:]

        assert order1 == order2


class TestCompletionsJsonlSchema:
    """Validate the schema that generate_pairs.py writes."""

    def test_roundtrip_schema(self):
        prompt = [
            {"role": "system", "content": SCRIPT_PROMPT},
            {"role": "user", "content": "[JERRY] Hello"},
        ]
        row = {
            "prompt": prompt,
            "completions": [
                "[JERRY] Line 1",
                "[GEORGE] Line 2",
                "[ELAINE] Line 3",
            ],
        }

        serialized = json.dumps(row)
        deserialized = json.loads(serialized)
        assert deserialized["prompt"] == prompt
        assert len(deserialized["completions"]) == 3
        assert all(isinstance(c, str) for c in deserialized["completions"])


class TestPreferencePairJsonlSchema:
    """Validate the schema that judge.py writes for TRL DPOTrainer."""

    def test_roundtrip_schema(self):
        pair = {
            "prompt": [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": "[JERRY] Hello"},
            ],
            "chosen": [{"role": "assistant", "content": "[JERRY] Great line"}],
            "rejected": [{"role": "assistant", "content": "[JERRY] Bad line"}],
        }

        serialized = json.dumps(pair)
        deserialized = json.loads(serialized)
        assert set(deserialized.keys()) == {"prompt", "chosen", "rejected"}
        assert deserialized["prompt"][0]["role"] == "system"
        assert deserialized["prompt"][1]["role"] == "user"
        assert deserialized["chosen"][0]["role"] == "assistant"
        assert deserialized["rejected"][0]["role"] == "assistant"
