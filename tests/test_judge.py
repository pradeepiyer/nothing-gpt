import json

from nothing_gpt.eval.judge import (
    JudgeScores,
    ParsedScores,
    build_judge_prompt,
    compute_metrics,
    parse_judge_response,
)


class TestBuildJudgePrompt:
    def test_includes_character_name(self):
        prompt = build_judge_prompt(
            character="JERRY",
            system_prompt="You are Jerry Seinfeld...",
            context_messages=[{"role": "user", "content": "[GEORGE] Hey Jerry."}],
            generated_response="What's the deal?",
        )
        assert "JERRY" in prompt

    def test_includes_context(self):
        context = [
            {"role": "user", "content": "[GEORGE] I'm thinking about quitting."},
            {"role": "assistant", "content": "You should."},
            {"role": "user", "content": "[GEORGE] Really?"},
        ]
        prompt = build_judge_prompt(
            character="JERRY",
            system_prompt="You are Jerry.",
            context_messages=context,
            generated_response="Absolutely.",
        )
        assert "I'm thinking about quitting" in prompt
        assert "You should." in prompt
        assert "Really?" in prompt

    def test_includes_generated_response(self):
        prompt = build_judge_prompt(
            character="ELAINE",
            system_prompt="You are Elaine.",
            context_messages=[],
            generated_response="Get out!",
        )
        assert "Get out!" in prompt

    def test_includes_scoring_dimensions(self):
        prompt = build_judge_prompt(
            character="KRAMER",
            system_prompt="You are Kramer.",
            context_messages=[],
            generated_response="Giddy up!",
        )
        assert "character_consistency" in prompt
        assert "humor" in prompt
        assert "coherence" in prompt
        assert "overall" in prompt


class TestParseJudgeResponse:
    def test_valid_json(self):
        response = json.dumps({
            "character_consistency": 4,
            "humor": 3,
            "coherence": 5,
            "overall": 4,
            "reasoning": "Good response.",
        })
        result = parse_judge_response(response)
        assert isinstance(result, ParsedScores)
        assert result.character_consistency == 4
        assert result.humor == 3
        assert result.coherence == 5
        assert result.overall == 4
        assert result.reasoning == "Good response."

    def test_markdown_fenced_json(self):
        inner = json.dumps({
            "character_consistency": 3, "humor": 2,
            "coherence": 4, "overall": 3, "reasoning": "Okay.",
        })
        response = f"```json\n{inner}\n```"
        result = parse_judge_response(response)
        assert result.character_consistency == 3
        assert result.humor == 2

    def test_markdown_fenced_without_language(self):
        inner = json.dumps({
            "character_consistency": 5, "humor": 5,
            "coherence": 5, "overall": 5, "reasoning": "Perfect.",
        })
        response = f"```\n{inner}\n```"
        result = parse_judge_response(response)
        assert result.overall == 5

    def test_missing_field_raises(self):
        response = json.dumps({
            "character_consistency": 4,
            "humor": 3,
            # missing coherence
            "overall": 4,
            "reasoning": "Incomplete.",
        })
        try:
            parse_judge_response(response)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "coherence" in str(e)

    def test_missing_reasoning_raises(self):
        response = json.dumps({
            "character_consistency": 4,
            "humor": 3,
            "coherence": 4,
            "overall": 4,
        })
        try:
            parse_judge_response(response)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "reasoning" in str(e)

    def test_out_of_range_score_raises(self):
        response = json.dumps({
            "character_consistency": 6,
            "humor": 3,
            "coherence": 4,
            "overall": 4,
            "reasoning": "Too high.",
        })
        try:
            parse_judge_response(response)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "character_consistency" in str(e)

    def test_zero_score_raises(self):
        response = json.dumps({
            "character_consistency": 0,
            "humor": 3,
            "coherence": 4,
            "overall": 4,
            "reasoning": "Zero is out of range.",
        })
        try:
            parse_judge_response(response)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "1-5" in str(e)

    def test_invalid_json_raises(self):
        try:
            parse_judge_response("not json at all")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid JSON" in str(e)


class TestComputeMetrics:
    def _make_scores(self, values: list[tuple[str, int, int, int, int]]) -> list[JudgeScores]:
        return [
            JudgeScores(
                character=char,
                generation_character=char,
                character_consistency=cc,
                humor=h,
                coherence=co,
                overall=o,
                reasoning="test",
            )
            for char, cc, h, co, o in values
        ]

    def test_mean_computation(self):
        scores = self._make_scores([
            ("JERRY", 4, 3, 5, 4),
            ("JERRY", 2, 1, 3, 2),
        ])
        metrics = compute_metrics(scores)
        assert metrics.mean_character_consistency == 3.0
        assert metrics.mean_humor == 2.0
        assert metrics.mean_coherence == 4.0
        assert metrics.mean_overall == 3.0

    def test_per_character_breakdown(self):
        scores = self._make_scores([
            ("JERRY", 4, 4, 4, 4),
            ("JERRY", 2, 2, 2, 2),
            ("GEORGE", 5, 5, 5, 5),
        ])
        metrics = compute_metrics(scores)
        assert metrics.per_character["JERRY"]["character_consistency"] == 3.0
        assert metrics.per_character["GEORGE"]["character_consistency"] == 5.0

    def test_empty_scores(self):
        metrics = compute_metrics([])
        assert metrics.mean_overall == 0
        assert metrics.per_character == {}
