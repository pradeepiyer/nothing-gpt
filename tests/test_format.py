from nothing_gpt.data.format import (
    episode_to_examples,
    format_context,
    merge_consecutive_turns,
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


class TestEpisodeToExamples:
    def _make_episode(self, turns: list[tuple[str, str]]) -> Episode:
        return Episode(
            episode_id="S01E01",
            season=1,
            episode_no=1,
            turns=[DialogueTurn(character=c, dialogue=d) for c, d in turns],
        )

    def test_generates_examples_for_main_characters(self):
        episode = self._make_episode([
            ("JERRY", "So what happened?"),
            ("GEORGE", "I didn't get the job."),
            ("JERRY", "Why not?"),
        ])
        examples = episode_to_examples(episode)

        # George's line (has Jerry's context) and Jerry's second line (has 2 turns context)
        assert len(examples) == 2

    def test_skips_non_main_characters(self):
        episode = self._make_episode([
            ("JERRY", "Hello"),
            ("NEWMAN", "Hello Jerry."),
            ("JERRY", "Newman."),
        ])
        examples = episode_to_examples(episode)
        # Only Jerry's second line should be an example (has context)
        assert len(examples) == 1
        assert examples[0]["completion"][0]["content"] == "Newman."

    def test_skips_first_line_no_context(self):
        episode = self._make_episode([
            ("JERRY", "Hello"),
        ])
        examples = episode_to_examples(episode)
        assert len(examples) == 0

    def test_prompt_completion_structure(self):
        episode = self._make_episode([
            ("GEORGE", "I'm thinking about quitting."),
            ("JERRY", "You should."),
        ])
        examples = episode_to_examples(episode)
        assert len(examples) == 1

        ex = examples[0]
        assert len(ex["prompt"]) == 2
        assert ex["prompt"][0]["role"] == "system"
        assert ex["prompt"][1]["role"] == "user"
        assert "[GEORGE]" in ex["prompt"][1]["content"]
        assert len(ex["completion"]) == 1
        assert ex["completion"][0]["role"] == "assistant"
        assert ex["completion"][0]["content"] == "You should."

    def test_context_window_limit(self):
        # Create episode with 12 turns, last one from JERRY
        turns = [(f"CHAR{i}", f"Line {i}") for i in range(11)]
        turns.append(("JERRY", "Final line"))
        episode = self._make_episode(turns)
        examples = episode_to_examples(episode)

        # Find Jerry's example
        jerry_examples = [
            e for e in examples
            if e["completion"][0]["content"] == "Final line"
        ]
        assert len(jerry_examples) == 1

        # Context should have at most CONTEXT_WINDOW (8) turns
        context = jerry_examples[0]["prompt"][1]["content"]
        context_lines = context.strip().split("\n")
        assert len(context_lines) <= 8

    def test_consecutive_turns_merged(self):
        episode = self._make_episode([
            ("JERRY", "So."),
            ("JERRY", "What happened?"),
            ("GEORGE", "Nothing."),
        ])
        examples = episode_to_examples(episode)

        # George's response should have merged Jerry context
        assert len(examples) == 1
        context = examples[0]["prompt"][1]["content"]
        assert "[JERRY] So. What happened?" in context


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
