import json
import tempfile
from pathlib import Path

from nothing_gpt.characters import MAIN_CHARACTERS, get_system_prompt
from nothing_gpt.eval.generate import (
    EvalPrompt,
    GeneratedResponse,
    _balanced_sample,
    _identify_character,
    load_eval_prompts,
    load_responses,
    save_responses,
)


def _make_val_example(character: str, turns: list[tuple[str, str]]) -> dict:
    """Build a val.jsonl example in the training data format."""
    messages = [{"role": "system", "content": get_system_prompt(character)}]
    for role, content in turns:
        messages.append({"role": role, "content": content})
    return {"messages": messages}


class TestIdentifyCharacter:
    def test_identifies_all_main_characters(self):
        for char in MAIN_CHARACTERS:
            prompt = get_system_prompt(char)
            assert _identify_character(prompt) == char

    def test_raises_on_unknown_prompt(self):
        try:
            _identify_character("You are Newman.")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestLoadEvalPrompts:
    def test_extracts_context_and_reference(self):
        example = _make_val_example("JERRY", [
            ("user", "[GEORGE] What's the deal?"),
            ("assistant", "The deal with what?"),
            ("user", "[GEORGE] Airline peanuts."),
            ("assistant", "I know, right?"),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(example) + "\n")
            path = Path(f.name)

        prompts = load_eval_prompts(path)
        assert len(prompts) == 1

        p = prompts[0]
        assert p.character == "JERRY"
        assert p.reference == "I know, right?"
        # Context should be everything except the last assistant turn
        assert len(p.context) == 3
        assert p.context[-1] == {"role": "user", "content": "[GEORGE] Airline peanuts."}

    def test_skips_examples_not_ending_with_assistant(self):
        messages = [
            {"role": "system", "content": get_system_prompt("JERRY")},
            {"role": "user", "content": "[GEORGE] Hey."},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": messages}) + "\n")
            path = Path(f.name)

        prompts = load_eval_prompts(path)
        assert len(prompts) == 0

    def test_balanced_sampling(self):
        examples = []
        for char in sorted(MAIN_CHARACTERS):
            for i in range(20):
                examples.append(_make_val_example(char, [
                    ("user", f"[OTHER] Line {i}"),
                    ("assistant", f"Response {i}"),
                ]))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            path = Path(f.name)

        prompts = load_eval_prompts(path, max_prompts=20)
        assert len(prompts) == 20

        char_counts = {}
        for p in prompts:
            char_counts[p.character] = char_counts.get(p.character, 0) + 1

        # Each character should get 5 (20 / 4 characters)
        for char in MAIN_CHARACTERS:
            assert char_counts[char] == 5


class TestBalancedSample:
    def test_equal_distribution(self):
        prompts = [
            EvalPrompt(character=char, system_prompt="", context=[], reference=f"ref-{i}")
            for char in sorted(MAIN_CHARACTERS)
            for i in range(10)
        ]
        sampled = _balanced_sample(prompts, max_prompts=12, seed=42)
        assert len(sampled) == 12

        char_counts = {}
        for p in sampled:
            char_counts[p.character] = char_counts.get(p.character, 0) + 1

        # 12 / 4 = 3 per character
        for char in MAIN_CHARACTERS:
            assert char_counts[char] == 3

    def test_deterministic_with_same_seed(self):
        prompts = [
            EvalPrompt(character=char, system_prompt="", context=[], reference=f"ref-{i}")
            for char in sorted(MAIN_CHARACTERS)
            for i in range(10)
        ]
        sample1 = _balanced_sample(prompts, max_prompts=8, seed=99)
        sample2 = _balanced_sample(prompts, max_prompts=8, seed=99)
        assert [p.reference for p in sample1] == [p.reference for p in sample2]


class TestSaveLoadRoundtrip:
    def test_responses_roundtrip(self):
        responses = [
            GeneratedResponse(
                character="JERRY",
                system_prompt="You are Jerry.",
                context=[{"role": "user", "content": "Hello"}],
                reference="Hey there.",
                generated="What's the deal?",
                generation_character="JERRY",
            ),
            GeneratedResponse(
                character="GEORGE",
                system_prompt="You are George.",
                context=[{"role": "user", "content": "Hi"}],
                reference="Hey.",
                generated="I'm very uncomfortable.",
                generation_character="GEORGE",
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        save_responses(responses, path)
        loaded = load_responses(path)

        assert len(loaded) == 2
        assert loaded[0].character == "JERRY"
        assert loaded[0].generated == "What's the deal?"
        assert loaded[0].context == [{"role": "user", "content": "Hello"}]
        assert loaded[1].character == "GEORGE"
        assert loaded[1].generation_character == "GEORGE"
