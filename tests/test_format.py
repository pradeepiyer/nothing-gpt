from nothing_gpt.constants import SCRIPT_PROMPT
from nothing_gpt.data.format import (
    CONTEXT_TURNS,
    MIN_COMPLETION_TURNS,
    WINDOW_SIZE,
    episode_to_examples,
    format_context,
    merge_consecutive_turns,
    print_token_stats,
    split_episodes_by_id,
)
from nothing_gpt.data.parse import DialogueTurn, Episode


class TestFormatContext:
    def test_formats_turns_with_brackets(self):
        turns = [
            DialogueTurn(character="JERRY", dialogue="Hello"),
            DialogueTurn(character="GEORGE", dialogue="Hey"),
        ]
        result = format_context(turns)
        assert result == "[JERRY] Hello\n[GEORGE] Hey"

    def test_empty_turns(self):
        assert format_context([]) == ""


class TestMergeConsecutiveTurns:
    def test_merges_same_character(self):
        turns = [
            DialogueTurn(character="JERRY", dialogue="Hello."),
            DialogueTurn(character="JERRY", dialogue="How are you?"),
        ]
        merged = merge_consecutive_turns(turns)
        assert len(merged) == 1
        assert merged[0].dialogue == "Hello. How are you?"

    def test_different_characters_not_merged(self):
        turns = [
            DialogueTurn(character="JERRY", dialogue="Hello"),
            DialogueTurn(character="GEORGE", dialogue="Hey"),
        ]
        merged = merge_consecutive_turns(turns)
        assert len(merged) == 2

    def test_empty_input(self):
        assert merge_consecutive_turns([]) == []

    def test_three_consecutive_same_character(self):
        turns = [
            DialogueTurn(character="KRAMER", dialogue="Hey."),
            DialogueTurn(character="KRAMER", dialogue="Listen."),
            DialogueTurn(character="KRAMER", dialogue="I got an idea."),
        ]
        merged = merge_consecutive_turns(turns)
        assert len(merged) == 1
        assert merged[0].dialogue == "Hey. Listen. I got an idea."


def _make_episode(turns: list[tuple[str, str]], episode_id: str = "S01E01") -> Episode:
    return Episode(
        episode_id=episode_id,
        season=1,
        episode_no=1,
        turns=[DialogueTurn(character=c, dialogue=d) for c, d in turns],
    )


def _alternating_turns(n: int) -> list[tuple[str, str]]:
    """Generate n alternating JERRY/GEORGE turns."""
    chars = ["JERRY", "GEORGE"]
    return [(chars[i % 2], f"Line {i}") for i in range(n)]


class TestEpisodeToExamples:
    def test_each_example_has_prompt_and_completion(self):
        episode = _make_episode(_alternating_turns(20))
        examples = episode_to_examples(episode)
        assert len(examples) >= 1
        for ex in examples:
            assert len(ex["prompt"]) == 2
            assert ex["prompt"][0]["role"] == "system"
            assert ex["prompt"][1]["role"] == "user"
            assert len(ex["completion"]) == 1
            assert ex["completion"][0]["role"] == "assistant"

    def test_system_prompt_is_script_prompt(self):
        episode = _make_episode(_alternating_turns(20))
        examples = episode_to_examples(episode)
        for ex in examples:
            assert ex["prompt"][0]["content"] == SCRIPT_PROMPT

    def test_user_and_assistant_use_bracket_format(self):
        episode = _make_episode(_alternating_turns(20))
        examples = episode_to_examples(episode)
        for ex in examples:
            user_content = ex["prompt"][1]["content"]
            assistant_content = ex["completion"][0]["content"]
            for line in user_content.split("\n"):
                assert line.startswith("["), f"User line missing bracket format: {line}"
            for line in assistant_content.split("\n"):
                assert line.startswith("["), f"Assistant line missing bracket format: {line}"

    def test_user_has_context_turns_lines(self):
        episode = _make_episode(_alternating_turns(20))
        examples = episode_to_examples(episode)
        for ex in examples:
            user_lines = ex["prompt"][1]["content"].split("\n")
            assert len(user_lines) == CONTEXT_TURNS

    def test_assistant_has_at_least_min_completion_turns(self):
        episode = _make_episode(_alternating_turns(20))
        examples = episode_to_examples(episode)
        for ex in examples:
            assistant_lines = ex["completion"][0]["content"].split("\n")
            assert len(assistant_lines) >= MIN_COMPLETION_TURNS

    def test_sliding_window_produces_overlapping_examples(self):
        episode = _make_episode(_alternating_turns(50))
        examples = episode_to_examples(episode)
        assert len(examples) > 1

    def test_short_episode_produces_no_examples(self):
        episode = _make_episode(_alternating_turns(10))
        examples = episode_to_examples(episode)
        assert len(examples) == 0

    def test_all_characters_appear_in_output(self):
        turns = [
            ("JERRY", "Hey."), ("GEORGE", "What?"), ("ELAINE", "Nothing."),
            ("KRAMER", "Giddy up!"), ("JERRY", "So."), ("GEORGE", "Yeah."),
            ("ELAINE", "Right."), ("KRAMER", "Oh!"), ("JERRY", "Ok."),
            ("GEORGE", "Fine."), ("ELAINE", "Sure."), ("KRAMER", "Alright."),
            ("JERRY", "Done."), ("GEORGE", "Done."), ("ELAINE", "Done."),
            ("KRAMER", "Done."), ("JERRY", "A."), ("GEORGE", "B."),
            ("ELAINE", "C."), ("KRAMER", "D."),
        ]
        episode = _make_episode(turns)
        examples = episode_to_examples(episode)
        assert len(examples) >= 1
        all_text = " ".join(
            ex["prompt"][1]["content"] + ex["completion"][0]["content"]
            for ex in examples
        )
        for char in ["JERRY", "GEORGE", "ELAINE", "KRAMER"]:
            assert f"[{char}]" in all_text

    def test_consecutive_turns_merged_before_windowing(self):
        turns = [
            ("JERRY", "Hey."), ("JERRY", "Listen."),
            ("GEORGE", "What?"), ("JERRY", "Nothing."),
            ("GEORGE", "Ok."), ("JERRY", "Yeah."),
            ("GEORGE", "Fine."), ("JERRY", "Sure."),
            ("GEORGE", "Right."), ("JERRY", "Yep."),
            ("GEORGE", "Alright."), ("JERRY", "Done."),
            ("GEORGE", "Done."), ("JERRY", "A."),
            ("GEORGE", "B."), ("JERRY", "C."),
        ]
        episode = _make_episode(turns)
        examples = episode_to_examples(episode)
        assert len(examples) >= 1
        # "Hey. Listen." should be merged into one turn
        all_text = " ".join(
            ex["prompt"][1]["content"] + ex["completion"][0]["content"]
            for ex in examples
        )
        assert "Hey. Listen." in all_text

    def test_assistant_completion_capped_at_window_size(self):
        episode = _make_episode(_alternating_turns(100))
        examples = episode_to_examples(episode)
        for ex in examples:
            user_lines = ex["prompt"][1]["content"].split("\n")
            assistant_lines = ex["completion"][0]["content"].split("\n")
            total = len(user_lines) + len(assistant_lines)
            assert total <= WINDOW_SIZE


class TestSplitEpisodes:
    def _make_episodes(self, n: int) -> list[Episode]:
        return [
            Episode(
                episode_id=f"S01E{i:02d}",
                season=1,
                episode_no=i,
                turns=[DialogueTurn(character="JERRY", dialogue=f"Line {i}")],
            )
            for i in range(n)
        ]

    def test_split_ratio(self):
        episodes = self._make_episodes(100)
        train, val = split_episodes_by_id(episodes, val_ratio=0.1)
        assert len(val) == 10
        assert len(train) == 90

    def test_no_overlap(self):
        episodes = self._make_episodes(50)
        train, val = split_episodes_by_id(episodes, val_ratio=0.2)
        train_ids = {ep.episode_id for ep in train}
        val_ids = {ep.episode_id for ep in val}
        assert train_ids.isdisjoint(val_ids)

    def test_all_episodes_present(self):
        episodes = self._make_episodes(20)
        train, val = split_episodes_by_id(episodes, val_ratio=0.1)
        all_ids = {ep.episode_id for ep in train} | {ep.episode_id for ep in val}
        assert all_ids == {ep.episode_id for ep in episodes}

    def test_deterministic(self):
        episodes = self._make_episodes(30)
        train1, val1 = split_episodes_by_id(episodes, seed=42)
        train2, val2 = split_episodes_by_id(episodes, seed=42)
        assert [e.episode_id for e in train1] == [e.episode_id for e in train2]
        assert [e.episode_id for e in val1] == [e.episode_id for e in val2]

    def test_at_least_one_val(self):
        episodes = self._make_episodes(3)
        train, val = split_episodes_by_id(episodes, val_ratio=0.01)
        assert len(val) >= 1


class TestPrintTokenStats:
    def _make_examples(self, n: int) -> list[dict]:
        return [
            {
                "prompt": [
                    {"role": "system", "content": SCRIPT_PROMPT},
                    {"role": "user", "content": f"[JERRY] Line {i}"},
                ],
                "completion": [
                    {"role": "assistant", "content": f"[GEORGE] Response {i}"},
                ],
            }
            for i in range(n)
        ]

    def test_prints_stats_for_multiple_examples(self, capsys):
        print_token_stats(self._make_examples(10))
        output = capsys.readouterr().out
        assert "Token length statistics" in output
        assert "P50:" in output
        assert "P90:" in output
        assert "Exceeding" in output

    def test_handles_single_example(self, capsys):
        print_token_stats(self._make_examples(1))
        output = capsys.readouterr().out
        assert "Token length statistics (1 examples" in output
        assert "P50:" not in output  # quantiles need >= 2 data points

    def test_handles_empty_input(self, capsys):
        print_token_stats([])
        output = capsys.readouterr().out
        assert "No examples to analyze." in output
