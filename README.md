# Nothing-GPT

Fine-tuned [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on Seinfeld TV scripts. Give it a premise, get a script.

## Architecture

**Pipeline:** Data &rarr; SFT &rarr; DPO &rarr; Serving &rarr; UI

All stages run on both compute platforms:

- **Modal** &mdash; shared Volume, L4 GPUs, secrets for HF/W&B/Gemini
- **GKE Autopilot** &mdash; GCS FUSE volume at `/vol` (bucket: `nothing-gpt-data`), spot L4 GPUs, Workload Identity via `nothing-gpt-sa`

## Project Structure

```
nothing_gpt/
  data/       # ETL: download, parse, format
  sft/        # SFT training + data formatting
  dpo/        # DPO: generate_pairs, judge, train
  serve/      # vLLM serving
  ui/         # Gradio UI
  modal/      # Modal app definitions and shared config
k8s/          # GKE manifests (sft-train, dpo-train, generate-pairs, judge, serve, ui)
docker/       # Dockerfiles + cloudbuild.yaml
scripts/      # run_etl.py
tests/
```

## Setup

```bash
uv sync
```

**Modal secrets:**
```bash
uv run modal secret create huggingface-secret HF_TOKEN=<token>
uv run modal secret create wandb-secret WANDB_API_KEY=<key>
uv run modal secret create gemini-secret GEMINI_API_KEY=<key>
```

**GKE secrets:**
```bash
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<token>
kubectl create secret generic wandb-secret --from-literal=WANDB_API_KEY=<key>
kubectl create secret generic gemini-secret --from-literal=GEMINI_API_KEY=<key>
kubectl apply -f k8s/service-account.yaml
```

## Data Pipeline

Source: [Kaggle seinfeld-chronicles](https://www.kaggle.com/datasets/thec03u5/seinfeld-chronicles)

```bash
uv run python scripts/run_etl.py                                      # download, parse, format
uv run modal volume put nothing-gpt-vol data/sft/ /data/sft/          # upload to Modal
gsutil -m rsync -r data/sft/ gs://nothing-gpt-data/data/sft/          # upload to GCS
```

Output: script-continuation JSONL (sliding window, 9K train / 1K val examples).

## SFT Training

QLoRA config: r=32, alpha=32, lr=3e-5, dropout=0.1, batch=4, grad_accum=4, max_length=2048, completion_only_loss, cosine schedule, warmup=50, 1 epoch, L4 GPU.

```bash
# Modal
uv run modal run --detach -m nothing_gpt.modal.train::train

# GKE
gcloud builds submit --config docker/cloudbuild.yaml --substitutions=_TARGET=train .
kubectl apply -f k8s/sft-train.yaml
```

## DPO Pipeline

Three stages: generate completions &rarr; judge preferences &rarr; train.

### 1. Generate pairs

5 completions per prompt (temperature=0.9) from the vLLM serve endpoint.

```bash
# Local
uv run python -m nothing_gpt.dpo.generate_pairs

# GKE
gcloud builds submit --config docker/cloudbuild.yaml --substitutions=_TARGET=dpo .
kubectl apply -f k8s/generate-pairs.yaml
```

### 2. Judge

Gemini 2.5 Pro (`JUDGE_MODEL` env var) ranks best/worst completion per prompt.

```bash
# Local
uv run python -m nothing_gpt.dpo.judge

# GKE
kubectl apply -f k8s/judge.yaml
```

### 3. Train

DPO config: beta=0.1, sigmoid loss, lr=5e-6, batch=2, eval_batch=1, grad_accum=8, max_length=2048, cosine schedule, warmup=20, 1 epoch, L4 GPU.

```bash
# Modal
uv run modal run --detach -m nothing_gpt.modal.dpo_train::train

# GKE
kubectl apply -f k8s/dpo-train.yaml
```

## Serving

vLLM with LoRA adapter, OpenAI-compatible API. Config: max-model-len=2048, max-lora-rank=32, enforce-eager, gpu-memory-utilization=0.95, L4 GPU.

Serves the DPO adapter as `model="seinfeld"`.

```bash
# Modal
uv run modal deploy -m nothing_gpt.modal.serve

# GKE
gcloud builds submit --config docker/cloudbuild.yaml --substitutions=_TARGET=serve .
kubectl apply -f k8s/serve.yaml
```

## UI

Gradio scene generator: premise &rarr; multi-round API calls &rarr; script output.

```bash
# Modal
uv run modal deploy -m nothing_gpt.modal.app

# GKE
gcloud builds submit --config docker/cloudbuild.yaml --substitutions=_TARGET=ui .
kubectl apply -f k8s/ui.yaml
```

## Development

```bash
uv run pytest
uv run ruff check nothing_gpt/ tests/
uv run pyright nothing_gpt/
```
