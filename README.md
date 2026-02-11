# Nothing-GPT

Generate a theme inspired by Seinfeld TV show based on a theme powered by a fine-tuned Llama 3.2 3B Instruct model.

## How it works

1. **Data**: TV scripts from Kaggle, parsed into dialogue turns per episode
2. **Training**: QLoRA fine-tuning via TRL SFTTrainer on Modal (T4 GPU). Generates scene based on theme. 
3. **Serving**: vLLM on Modal with LoRA adapter, OpenAI-compatible API
4. **UI**: Gradio chat interface on Modal — give a theme to generate a scene

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
uv run modal run -m nothing_gpt.sft.train             # train
uv run modal deploy -m nothing_gpt.serve.server       # deploy API
uv run modal deploy -m nothing_gpt.ui.app             # deploy UI
```


## Development

```bash
uv run pytest                  # tests
uv run ruff check nothing_gpt/ tests/  # lint
uv run pyright nothing_gpt/            # type check
```
