"""End-to-end evaluation orchestration."""

import argparse
import json
from pathlib import Path

from nothing_gpt.eval.generate import (
    SERVE_URL,
    generate_cross_character,
    generate_responses,
    load_eval_prompts,
    load_responses,
    save_responses,
)
from nothing_gpt.eval.judge import (
    compute_metrics,
    judge_responses,
    load_scores,
    save_scores,
)

DEFAULT_OUTPUT_DIR = Path("data/eval")
DEFAULT_VAL_PATH = Path("data/training/val.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Nothing-GPT model quality")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for eval outputs",
    )
    parser.add_argument(
        "--val-path", type=Path, default=DEFAULT_VAL_PATH,
        help="Path to val.jsonl",
    )
    parser.add_argument(
        "--base-url", type=str, default=SERVE_URL,
        help="vLLM endpoint URL",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=200,
        help="Number of eval prompts to use",
    )
    parser.add_argument(
        "--cross-character-count", type=int, default=50,
        help="Number of prompts for cross-character eval",
    )
    parser.add_argument(
        "--judge-count", type=int, default=100,
        help="Max responses to judge",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Load saved responses instead of generating",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge scoring",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "responses.jsonl"
    cross_char_path = output_dir / "cross_character_responses.jsonl"
    judge_scores_path = output_dir / "judge_scores.jsonl"
    cross_char_scores_path = output_dir / "cross_character_scores.jsonl"
    summary_path = output_dir / "summary.json"

    # Step 1: Load prompts
    print(f"Loading eval prompts from {args.val_path}...")
    prompts = load_eval_prompts(args.val_path, max_prompts=args.max_prompts)
    print(f"  Loaded {len(prompts)} prompts")
    char_counts = {}
    for p in prompts:
        char_counts[p.character] = char_counts.get(p.character, 0) + 1
    for char, count in sorted(char_counts.items()):
        print(f"    {char}: {count}")

    # Step 2: Generate responses
    if args.skip_generation:
        print(f"Loading saved responses from {responses_path}...")
        responses = load_responses(responses_path)
        print(f"  Loaded {len(responses)} responses")

        print(f"Loading saved cross-character responses from {cross_char_path}...")
        cross_char_responses = load_responses(cross_char_path)
        print(f"  Loaded {len(cross_char_responses)} cross-character responses")
    else:
        print(f"Generating responses via {args.base_url}...")
        responses = generate_responses(prompts, base_url=args.base_url)
        save_responses(responses, responses_path)
        print(f"  Generated and saved {len(responses)} responses")

        cross_prompts = prompts[:args.cross_character_count]
        print(
            f"Generating cross-character responses "
            f"({len(cross_prompts)} prompts × 4 characters)..."
        )
        cross_char_responses = generate_cross_character(
            cross_prompts, base_url=args.base_url,
        )
        save_responses(cross_char_responses, cross_char_path)
        print(f"  Generated and saved {len(cross_char_responses)} cross-character responses")

    if args.skip_judge:
        print("Skipping judge scoring (--skip-judge)")
        # Try loading existing scores for summary
        if judge_scores_path.exists():
            judge_scores = load_scores(judge_scores_path)
            cross_char_scores = load_scores(cross_char_scores_path)
        else:
            print("No saved scores found. Run without --skip-judge to score.")
            return
    else:
        # Step 3: Judge responses
        print(f"Judging responses (up to {args.judge_count})...")
        judge_scores = judge_responses(responses, max_responses=args.judge_count)
        save_scores(judge_scores, judge_scores_path)
        print(f"  Scored {len(judge_scores)} responses")

        print("Judging cross-character responses...")
        cross_char_scores = judge_responses(cross_char_responses)
        save_scores(cross_char_scores, cross_char_scores_path)
        print(f"  Scored {len(cross_char_scores)} cross-character responses")

    # Step 4: Compute metrics
    print("\n=== Quality Metrics ===")
    quality = compute_metrics(judge_scores)
    print(f"  Character consistency: {quality.mean_character_consistency:.2f}")
    print(f"  Humor:                 {quality.mean_humor:.2f}")
    print(f"  Coherence:             {quality.mean_coherence:.2f}")
    print(f"  Overall:               {quality.mean_overall:.2f}")
    print("\n  Per character:")
    for char, means in sorted(quality.per_character.items()):
        print(
            f"    {char}: consistency={means['character_consistency']:.2f} "
            f"humor={means['humor']:.2f} coherence={means['coherence']:.2f} "
            f"overall={means['overall']:.2f}"
        )

    # Step 5: Compute distinguishability
    print("\n=== Character Distinguishability ===")
    distinguishability = _compute_distinguishability(cross_char_scores)
    print(f"  Correct character scored highest: {distinguishability['accuracy']:.1%}")
    print(f"    ({distinguishability['correct']}/{distinguishability['total']} prompts)")
    for char, acc in sorted(distinguishability["per_character"].items()):
        print(f"    {char}: {acc:.1%}")

    # Step 6: Save summary
    summary = {
        "quality": {
            "mean_character_consistency": quality.mean_character_consistency,
            "mean_humor": quality.mean_humor,
            "mean_coherence": quality.mean_coherence,
            "mean_overall": quality.mean_overall,
            "per_character": quality.per_character,
        },
        "distinguishability": distinguishability,
        "counts": {
            "prompts": len(prompts),
            "responses": len(responses),
            "cross_character_responses": len(cross_char_responses),
            "judged_responses": len(judge_scores),
            "judged_cross_character": len(cross_char_scores),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def _compute_distinguishability(
    scores: list,
) -> dict:
    """Check if the correct character gets the highest character_consistency score.

    Groups scores by (character, context) to compare the 4 generations per prompt.
    """
    # Scores arrive in groups of 4 (one per generation_character),
    # since generate_cross_character iterates characters in sorted order per prompt
    correct = 0
    total = 0
    per_char_correct: dict[str, int] = {}
    per_char_total: dict[str, int] = {}

    for i in range(0, len(scores), 4):
        group = scores[i : i + 4]
        if len(group) < 4:
            break

        original_char = group[0].character
        best = max(group, key=lambda s: s.character_consistency)

        per_char_total[original_char] = per_char_total.get(original_char, 0) + 1
        total += 1

        if best.generation_character == original_char:
            correct += 1
            per_char_correct[original_char] = per_char_correct.get(original_char, 0) + 1

    per_character_acc = {
        char: per_char_correct.get(char, 0) / per_char_total[char]
        for char in per_char_total
    }

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_character": per_character_acc,
    }
