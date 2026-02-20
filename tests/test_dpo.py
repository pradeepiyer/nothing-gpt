import json
import os
import random
import tempfile

from nothing_gpt.constants import SCRIPT_PROMPT
from nothing_gpt.dpo.generate_pairs import (  # type: ignore[import-not-found]
    _parse_line,
    load_prompts,
)
from nothing_gpt.dpo.judge import (  # type: ignore[import-not-found]
    MIN_CONFIDENCE,
    build_judge_prompt,
    build_preference_pair,
)


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
    def test_loads_prompts_from_both_files(self):
        with tempfile.TemporaryDirectory() as data_dir:
            prompt_a = [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": "[JERRY] Hello"},
            ]
            prompt_b = [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": "[GEORGE] Hey"},
            ]
            completion = [{"role": "assistant", "content": "[ELAINE] Hi"}]

            with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
                f.write(json.dumps({"prompt": prompt_a, "completion": completion}) + "\n")
                f.write(json.dumps({"prompt": prompt_a, "completion": completion}) + "\n")

            with open(os.path.join(data_dir, "val.jsonl"), "w") as f:
                f.write(json.dumps({"prompt": prompt_b, "completion": completion}) + "\n")

            prompts = load_prompts(data_dir)
            assert len(prompts) == 3
            assert prompts[0] == prompt_a
            assert prompts[2] == prompt_b
            assert prompts[0][0]["role"] == "system"

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as data_dir:
            assert load_prompts(data_dir) == []


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
            "[JERRY] Line D\n[GEORGE] Response D",
            "[JERRY] Line E\n[GEORGE] Response E",
        ]
        pair = build_preference_pair(prompt, completions, best_idx=0, worst_idx=4)
        assert pair["prompt"] == prompt
        assert pair["chosen"] == [{"role": "assistant", "content": completions[0]}]
        assert pair["rejected"] == [{"role": "assistant", "content": completions[4]}]

    def test_all_fields_present(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c", "d", "e"]
        pair = build_preference_pair(prompt, completions, best_idx=1, worst_idx=0)
        assert set(pair.keys()) == {"prompt", "chosen", "rejected"}

    def test_chosen_rejected_are_message_lists(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c", "d", "e"]
        pair = build_preference_pair(prompt, completions, best_idx=4, worst_idx=1)
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
                "[KRAMER] Line 4",
                "[JERRY] Line 5",
            ],
        }

        serialized = json.dumps(row)
        deserialized = json.loads(serialized)
        assert deserialized["prompt"] == prompt
        assert len(deserialized["completions"]) == 5
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


class TestBuildJudgePrompt:
    """Test dynamic judge prompt construction."""

    def test_includes_all_completions(self):
        completions = ["Line A", "Line B", "Line C", "Line D", "Line E"]
        prompt = build_judge_prompt(completions)
        for i, c in enumerate(completions):
            assert f"Completion {i + 1}:\n{c}" in prompt

    def test_correct_range_string(self):
        prompt = build_judge_prompt(["a", "b", "c", "d", "e"])
        assert '"best": <1|2|3|4|5>' in prompt
        assert '"worst": <1|2|3|4|5>' in prompt

    def test_includes_confidence_field(self):
        prompt = build_judge_prompt(["a", "b", "c"])
        assert '"confidence": <1-5>' in prompt

    def test_handles_3_completions(self):
        prompt = build_judge_prompt(["x", "y", "z"])
        assert "Below are 3 completions" in prompt
        assert "1|2|3" in prompt

    def test_handles_7_completions(self):
        completions = [f"c{i}" for i in range(7)]
        prompt = build_judge_prompt(completions)
        assert "Below are 7 completions" in prompt
        assert "1|2|3|4|5|6|7" in prompt


class TestConfidenceFiltering:
    """Test that low-confidence pairs are excluded."""

    def test_min_confidence_value(self):
        assert MIN_CONFIDENCE == 3

    def test_high_confidence_pair_included(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c", "d", "e"]
        confidence = 4
        pairs = []
        if confidence >= MIN_CONFIDENCE:
            pairs.append(build_preference_pair(prompt, completions, 0, 4))
        assert len(pairs) == 1

    def test_low_confidence_pair_excluded(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c", "d", "e"]
        confidence = 2
        pairs = []
        if confidence >= MIN_CONFIDENCE:
            pairs.append(build_preference_pair(prompt, completions, 0, 4))
        assert len(pairs) == 0

    def test_boundary_confidence_included(self):
        prompt = [{"role": "system", "content": "test"}]
        completions = ["a", "b", "c", "d", "e"]
        confidence = MIN_CONFIDENCE
        pairs = []
        if confidence >= MIN_CONFIDENCE:
            pairs.append(build_preference_pair(prompt, completions, 0, 4))
        assert len(pairs) == 1
