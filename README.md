# Nothing-GPT

Chat with Jerry, George, Elaine, and Kramer powered by a fine-tuned Llama 3.2 3B Instruct model.

## How it works

1. **Data**: TV scripts from Kaggle, parsed into dialogue turns per episode
2. **Training**: QLoRA fine-tuning via TRL SFTTrainer on Modal (T4 GPU). One model serves all characters — system prompts differentiate character voices.
3. **Serving**: vLLM on Modal with LoRA adapter, OpenAI-compatible API
4. **UI**: Gradio chat interface on Modal — pick a character and start talking

## Setup

```bash
uv sync
uv run modal secret create huggingface-secret HF_TOKEN=<your-token>
uv run modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

## Usage

```bash
uv run python scripts/run_etl.py                    # process training data
uv run modal volume put nothing-gpt-vol data/training/ /data/  # upload to Modal
uv run modal run -m src.nothing_gpt.modal.train      # train
uv run modal deploy -m src.nothing_gpt.modal.serve   # deploy API
uv run modal deploy -m src.nothing_gpt.modal.ui      # deploy UI
```


## Evaluation

LLM-as-judge eval suite that scores generated responses on character consistency, humor, and coherence, and tests whether the model has learned distinct character voices.

```bash
# Quick smoke test (generation only, no judging)
uv run python -m nothing_gpt.eval --skip-judge --max-prompts 10 --cross-character-count 5

# Full eval run
OPENAI_API_KEY=... uv run python -m nothing_gpt.eval

# Re-run judging on saved responses
OPENAI_API_KEY=... uv run python -m nothing_gpt.eval --skip-generation
```

Results are saved to `data/eval/`.

## Development

```bash
uv run pytest                  # tests
uv run ruff check src/ tests/  # lint
uv run pyright src/            # type check
```
