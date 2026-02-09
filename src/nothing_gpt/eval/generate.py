"""Extract eval prompts from val.jsonl and generate responses via vLLM endpoint."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from nothing_gpt.characters import MAIN_CHARACTERS, get_system_prompt

SERVE_URL = "https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1"


@dataclass
class EvalPrompt:
    character: str
    system_prompt: str
    context: list[dict[str, str]]
    reference: str


@dataclass
class GeneratedResponse:
    character: str
    system_prompt: str
    context: list[dict[str, str]]
    reference: str
    generated: str
    generation_character: str = ""


def _identify_character(system_content: str) -> str:
    """Identify which character a system prompt belongs to by matching against known prompts."""
    for char in MAIN_CHARACTERS:
        if system_content == get_system_prompt(char):
            return char
    raise ValueError(f"System prompt does not match any known character: {system_content[:80]}...")


def load_eval_prompts(
    val_path: Path,
    max_prompts: int = 0,
    seed: int = 42,
) -> list[EvalPrompt]:
    """Load eval prompts from val.jsonl.

    Each example's messages are split into: system prompt, context (all messages
    except the last assistant turn), and reference (the last assistant turn).

    When max_prompts is set, samples equally across the 4 main characters.
    """
    prompts: list[EvalPrompt] = []

    with open(val_path) as f:
        for line in f:
            example = json.loads(line)
            messages = example["messages"]

            system_msg = messages[0]
            assert system_msg["role"] == "system"

            character = _identify_character(system_msg["content"])

            # Last message should be assistant (the reference response)
            conversation = messages[1:]
            if not conversation or conversation[-1]["role"] != "assistant":
                continue

            reference = conversation[-1]["content"]
            context = conversation[:-1]

            prompts.append(EvalPrompt(
                character=character,
                system_prompt=system_msg["content"],
                context=context,
                reference=reference,
            ))

    if max_prompts > 0 and max_prompts < len(prompts):
        prompts = _balanced_sample(prompts, max_prompts, seed)

    return prompts


def _balanced_sample(
    prompts: list[EvalPrompt],
    max_prompts: int,
    seed: int,
) -> list[EvalPrompt]:
    """Sample equally across characters, filling remainder from random pool."""
    rng = random.Random(seed)

    by_character: dict[str, list[EvalPrompt]] = {}
    for p in prompts:
        by_character.setdefault(p.character, []).append(p)

    per_character = max_prompts // len(by_character)
    sampled: list[EvalPrompt] = []

    for char in sorted(by_character):
        pool = by_character[char]
        rng.shuffle(pool)
        sampled.extend(pool[:per_character])

    # Fill remainder from all remaining prompts
    already_selected = set(id(p) for p in sampled)
    remainder = [p for p in prompts if id(p) not in already_selected]
    rng.shuffle(remainder)
    sampled.extend(remainder[: max_prompts - len(sampled)])

    return sampled


def generate_responses(
    prompts: list[EvalPrompt],
    base_url: str = SERVE_URL,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> list[GeneratedResponse]:
    """Generate responses for each prompt via the vLLM endpoint."""
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=300)
    responses: list[GeneratedResponse] = []

    for prompt in prompts:
        messages: list[Any] = [{"role": "system", "content": prompt.system_prompt}]
        messages.extend(prompt.context)

        response = client.chat.completions.create(
            model="seinfeld",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generated = response.choices[0].message.content or ""

        responses.append(GeneratedResponse(
            character=prompt.character,
            system_prompt=prompt.system_prompt,
            context=prompt.context,
            reference=prompt.reference,
            generated=generated,
            generation_character=prompt.character,
        ))

    return responses


def generate_cross_character(
    prompts: list[EvalPrompt],
    base_url: str = SERVE_URL,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> list[GeneratedResponse]:
    """For each prompt, generate with all 4 characters' system prompts."""
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=300)
    responses: list[GeneratedResponse] = []

    for prompt in prompts:
        for char in sorted(MAIN_CHARACTERS):
            system_prompt = get_system_prompt(char)
            messages: list[Any] = [{"role": "system", "content": system_prompt}]
            messages.extend(prompt.context)

            response = client.chat.completions.create(
                model="seinfeld",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generated = response.choices[0].message.content or ""

            responses.append(GeneratedResponse(
                character=prompt.character,
                system_prompt=system_prompt,
                context=prompt.context,
                reference=prompt.reference,
                generated=generated,
                generation_character=char,
            ))

    return responses


def save_responses(responses: list[GeneratedResponse], path: Path) -> None:
    """Save responses to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in responses:
            f.write(json.dumps(asdict(r)) + "\n")


def load_responses(path: Path) -> list[GeneratedResponse]:
    """Load responses from JSONL."""
    responses: list[GeneratedResponse] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            responses.append(GeneratedResponse(**data))
    return responses
