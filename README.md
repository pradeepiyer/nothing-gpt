# Nothing-GPT

Chat with Jerry, George, Elaine, and Kramer powered by a fine-tuned Llama 3.2 3B Instruct model.

## How it works

1. **Data**: TV scripts from Kaggle, parsed into dialogue turns per episode
2. **Training**: QLoRA fine-tuning via TRL SFTTrainer on Modal (T4 GPU). One model serves all characters — system prompts in the training data differentiate character voices.
3. **Serving**: vLLM on Modal with LoRA adapter, OpenAI-compatible API

## Setup

```bash
uv sync
```

### Modal secrets

```bash
uv run modal secret create huggingface-secret HF_TOKEN=<your-token>
uv run modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

## Usage

### 1. Process training data

```bash
uv run python scripts/run_etl.py
```

This downloads the Kaggle dataset, parses dialogues, and produces `data/training/train.jsonl` and `data/training/val.jsonl`.

### 2. Upload data to Modal

```bash
uv run modal volume put nothing-gpt-vol data/training/ /data/
```

### 3. Train

```bash
uv run modal run -m src.nothing_gpt.modal.train
```

Runs QLoRA fine-tuning (~3 epochs on ~40K examples). Adapter saved to the shared Modal volume.

### 4. Deploy

```bash
uv run modal deploy -m src.nothing_gpt.modal.serve
```

### 5. Chat

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-modal-app>.modal.run/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="seinfeld",
    messages=[
        {"role": "system", "content": "You are George Costanza..."},
        {"role": "user", "content": "[JERRY] So what happened at the interview?"},
    ],
)
print(response.choices[0].message.content)
```

Character system prompts are in `configs/characters.json`.

## Development

```bash
uv run pytest                  # tests
uv run ruff check src/ tests/  # lint
uv run pyright src/            # type check
```

## Project structure

```
configs/characters.json          # System prompts per character
src/nothing_gpt/
  characters.py                  # Load character configs
  data/
    download.py                  # Kaggle dataset download
    parse.py                     # CSV → episode dialogue sequences
    format.py                    # Dialogue → prompt-completion JSONL
  modal/
    config.py                    # App, volumes, images, constants
    train.py                     # QLoRA training
    serve.py                     # vLLM serving
scripts/run_etl.py               # Full data pipeline
tests/
  test_parse.py                  # 19 tests
  test_format.py                 # 17 tests
```
