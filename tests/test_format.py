from nothing_gpt.data.format import (
    MAX_MESSAGES,
    MIN_ASSISTANT_TURNS,
    STRIDE,
    _turns_to_messages,
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


class TestTurnsToMessages:
    def test_alternates_user_assistant(self):
        turns = [
            DialogueTurn(character="GEORGE", dialogue="What happened?"),
            DialogueTurn(character="JERRY", dialogue="Nothing."),
            DialogueTurn(character="GEORGE", dialogue="Nothing?"),
            DialogueTurn(character="JERRY", dialogue="Absolutely nothing."),
        ]
        messages = _turns_to_messages(turns, "JERRY")
        assert messages == [
            {"role": "user", "content": "[GEORGE] What happened?"},
            {"role": "assistant", "content": "Nothing."},
            {"role": "user", "content": "[GEORGE] Nothing?"},
            {"role": "assistant", "content": "Absolutely nothing."},
        ]

    def test_merges_consecutive_non_target_turns(self):
        turns = [
            DialogueTurn(character="GEORGE", dialogue="Hey."),
            DialogueTurn(character="ELAINE", dialogue="Hi."),
            DialogueTurn(character="JERRY", dialogue="Hello."),
        ]
        messages = _turns_to_messages(turns, "JERRY")
        assert messages == [
            {"role": "user", "content": "[GEORGE] Hey.\n[ELAINE] Hi."},
            {"role": "assistant", "content": "Hello."},
        ]

    def test_drops_trailing_user_lines(self):
        turns = [
            DialogueTurn(character="GEORGE", dialogue="Hey."),
            DialogueTurn(character="JERRY", dialogue="Hi."),
            DialogueTurn(character="GEORGE", dialogue="Bye."),
        ]
        messages = _turns_to_messages(turns, "JERRY")
        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"

    def test_target_starts_episode(self):
        turns = [
            DialogueTurn(character="JERRY", dialogue="So."),
            DialogueTurn(character="GEORGE", dialogue="What?"),
            DialogueTurn(character="JERRY", dialogue="Nothing."),
        ]
        messages = _turns_to_messages(turns, "JERRY")
        # First assistant message has no preceding user, so starts with assistant
        assert messages[0] == {"role": "assistant", "content": "So."}
        assert messages[1] == {"role": "user", "content": "[GEORGE] What?"}
        assert messages[2] == {"role": "assistant", "content": "Nothing."}

    def test_includes_non_main_characters_in_user(self):
        turns = [
            DialogueTurn(character="NEWMAN", dialogue="Hello Jerry."),
            DialogueTurn(character="JERRY", dialogue="Newman."),
        ]
        messages = _turns_to_messages(turns, "JERRY")
        assert messages[0]["content"] == "[NEWMAN] Hello Jerry."
        assert messages[1]["content"] == "Newman."


class TestEpisodeToExamples:
    def _make_episode(self, turns: list[tuple[str, str]]) -> Episode:
        return Episode(
            episode_id="S01E01",
            season=1,
            episode_no=1,
            turns=[DialogueTurn(character=c, dialogue=d) for c, d in turns],
        )

    def test_messages_structure(self):
        episode = self._make_episode([
            ("GEORGE", "I'm thinking about quitting."),
            ("JERRY", "You should."),
            ("GEORGE", "Really?"),
            ("JERRY", "Absolutely."),
        ])
        examples = episode_to_examples(episode)

        # Find Jerry's example
        jerry_examples = [
            e for e in examples
            if e["messages"][0]["content"].startswith("You are Jerry")
        ]
        assert len(jerry_examples) >= 1
        ex = jerry_examples[0]

        assert "messages" in ex
        assert "prompt" not in ex
        assert "completion" not in ex
        assert ex["messages"][0]["role"] == "system"
        assert ex["messages"][1]["role"] == "user"
        # Must alternate user/assistant after system
        roles = [m["role"] for m in ex["messages"][1:]]
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1], "Consecutive same roles found"

    def test_generates_examples_for_multiple_characters(self):
        episode = self._make_episode([
            ("JERRY", "Hey."),
            ("GEORGE", "Hey."),
            ("JERRY", "What's up?"),
            ("GEORGE", "Nothing."),
            ("JERRY", "Nothing?"),
            ("GEORGE", "Absolutely nothing."),
        ])
        examples = episode_to_examples(episode)

        characters_seen = set()
        for ex in examples:
            system_content = ex["messages"][0]["content"]
            if "Jerry" in system_content:
                characters_seen.add("JERRY")
            if "George" in system_content:
                characters_seen.add("GEORGE")

        assert "JERRY" in characters_seen
        assert "GEORGE" in characters_seen

    def test_skips_character_with_too_few_assistant_turns(self):
        # Elaine only has 1 line — should not produce an example (MIN_ASSISTANT_TURNS=2)
        episode = self._make_episode([
            ("JERRY", "Hey."),
            ("ELAINE", "Hi."),
            ("JERRY", "What's new?"),
        ])
        examples = episode_to_examples(episode)
        elaine_examples = [
            e for e in examples
            if "Elaine" in e["messages"][0]["content"]
        ]
        assert len(elaine_examples) == 0

    def test_min_assistant_turns_enforced(self):
        episode = self._make_episode([
            ("GEORGE", "Hey."),
            ("JERRY", "Hi."),
            ("GEORGE", "Bye."),
            ("JERRY", "See ya."),
        ])
        examples = episode_to_examples(episode)
        for ex in examples:
            assistant_count = sum(1 for m in ex["messages"] if m["role"] == "assistant")
            assert assistant_count >= MIN_ASSISTANT_TURNS

    def test_starts_with_user_ends_with_assistant(self):
        episode = self._make_episode([
            ("JERRY", "So."),
            ("GEORGE", "What?"),
            ("JERRY", "Nothing."),
            ("GEORGE", "Nothing?"),
            ("JERRY", "Yeah."),
            ("GEORGE", "Huh."),
        ])
        examples = episode_to_examples(episode)
        assert len(examples) > 0
        for ex in examples:
            # After system message, should start with user
            assert ex["messages"][1]["role"] == "user"
            assert ex["messages"][-1]["role"] == "assistant"

    def test_consecutive_turns_merged_before_conversion(self):
        episode = self._make_episode([
            ("GEORGE", "Hey."),
            ("GEORGE", "Listen."),
            ("JERRY", "What?"),
            ("GEORGE", "Nothing."),
            ("JERRY", "Oh."),
        ])
        examples = episode_to_examples(episode)
        jerry_examples = [
            e for e in examples
            if "Jerry" in e["messages"][0]["content"]
        ]
        assert len(jerry_examples) >= 1
        # The merged "Hey. Listen." should appear in user content
        user_msgs = [m for m in jerry_examples[0]["messages"] if m["role"] == "user"]
        assert any("Hey. Listen." in m["content"] for m in user_msgs)

    def test_windowing_limits_message_count(self):
        # Create a long episode that would exceed MAX_MESSAGES
        turns: list[tuple[str, str]] = []
        for i in range(30):
            if i % 2 == 0:
                turns.append(("GEORGE", f"Line {i}"))
            else:
                turns.append(("JERRY", f"Line {i}"))
        episode = self._make_episode(turns)
        examples = episode_to_examples(episode)

        for ex in examples:
            # messages includes system + user/assistant, so at most MAX_MESSAGES + 1
            non_system = [m for m in ex["messages"] if m["role"] != "system"]
            assert len(non_system) <= MAX_MESSAGES

    def test_windowing_produces_overlapping_examples(self):
        # With enough turns, sliding window with STRIDE should produce multiple examples
        turns: list[tuple[str, str]] = []
        for i in range(30):
            if i % 2 == 0:
                turns.append(("GEORGE", f"Line {i}"))
            else:
                turns.append(("JERRY", f"Line {i}"))
        episode = self._make_episode(turns)
        examples = episode_to_examples(episode)

        jerry_examples = [
            e for e in examples
            if "Jerry" in e["messages"][0]["content"]
        ]
        assert len(jerry_examples) > 1

    def test_no_examples_from_single_turn(self):
        episode = self._make_episode([("JERRY", "Hello")])
        examples = episode_to_examples(episode)
        assert len(examples) == 0


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
