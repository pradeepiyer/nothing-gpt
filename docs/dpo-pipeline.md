# DPO Preference Optimization Pipeline

## Context

SFT training plateaued at eval_loss 2.187 (completion-only) with no overfitting — the model has learned the training distribution well. Further SFT epochs would yield diminishing returns (eval loss flatlined for the last 40% of epoch 1 due to cosine LR decay, and the train-eval gap is only 0.09).

To improve output quality beyond what SFT provides, we'll add a DPO (Direct Preference Optimization) stage. This trains the model to prefer higher-quality completions (better character voice, humor, coherence, premise relevance) using preference pairs judged by GPT-5.

## Pipeline Overview

```
Prompts from val.jsonl
    → generate_pairs.py: 3 completions per prompt via vLLM (local or GKE Job)
    → judge.py: GPT-5 ranks completions → chosen/rejected pairs (local or GKE Job)
    → train.py: DPO training with dual LoRA adapters (Modal or GKE Job)
    → Serve DPO adapter alongside SFT adapter
```

## Step 1: Generate Completions

### `nothing_gpt/dpo/generate_pairs.py`

Generate multiple completions per prompt using the vLLM endpoint:

- Uses `SERVE_URL` env var (same pattern as `ui/app.py`)
- Reads prompts from `DATA_PATH/val.jsonl` (1,021 examples)
- For each prompt, generates 3 completions at temperature=0.9 (diverse sampling)
- Writes `completions.jsonl` to `DPO_DATA_PATH` — each row has `{prompt, completions: [str, str, str]}`

**Local:**
```bash
SERVE_URL=https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1 \
DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.generate_pairs
```

**GKE:** `k8s/dpo-generate-pairs.yaml` — Job using `ui:latest` image, CMD override, `SERVE_URL=http://nothing-gpt-serve:8000/v1`, GCS FUSE mount, no GPU.

## Step 2: GPT-5 Judging

### `nothing_gpt/dpo/judge.py`

For each prompt's 3 completions, GPT-5 ranks them on:
- **Faithfulness to Seinfeld scripts and characters** — Does the dialogue sound like it belongs in an actual Seinfeld episode? Does each character speak in their distinctive voice (Jerry's observational style, George's neurotic complaints, Elaine's sarcasm, Kramer's physicality and wild ideas)?

From each set of 3 completions, extracts the best (chosen) and worst (rejected) → 1 preference pair per prompt.

- Reads from `DPO_DATA_PATH/completions.jsonl`
- Writes `train.jsonl` and `val.jsonl` to `DPO_DATA_PATH` (90/10 split, shuffled with seed)
- Uses OpenAI SDK with `gpt-5` model
- ~1,021 API calls to GPT-5

**Local:**
```bash
OPENAI_API_KEY=... DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.judge
```

**GKE:** `k8s/dpo-judge.yaml` — Job using `ui:latest` image, CMD override, `OPENAI_API_KEY` from k8s secret `openai-secret`, GCS FUSE mount, no GPU.

## Step 3: DPO Training

### `nothing_gpt/dpo/train.py`

Follows the `sft/train.py` pattern: standalone `train(callbacks=None)` function, uses constants for paths, has checkpoint resume logic.

```python
# Dual-adapter setup for DPO
model = PeftModel.from_pretrained(base, ADAPTER_PATH, adapter_name="train")
model.load_adapter(ADAPTER_PATH, adapter_name="reference")

DPOConfig(
    model_adapter_name="train",
    ref_adapter_name="reference",
    beta=0.1,                    # KL penalty strength
    loss_type="sigmoid",         # Standard DPO loss
    learning_rate=5e-6,          # 6x lower than SFT's 3e-5
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=2048,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    eval_steps=25,
    save_steps=25,
    report_to="wandb",
    output_dir="/vol/checkpoints/dpo-r32",
)
```

Key differences from SFT:
- **LR 5e-6** (not 3e-5) — DPO diverges at higher LR
- **Dual adapter** — reference adapter stays frozen, train adapter gets updated
- **Preference data format** — `{prompt, chosen, rejected}` not `{prompt, completion}`
- **beta=0.1** — controls how far the model can drift from the SFT reference
- Reads from `DPO_DATA_PATH`, saves to `DPO_ADAPTER_PATH`

Memory: Fits on L4 (24GB). Second adapter is ~150MB, not a full model copy.

### `nothing_gpt/modal/dpo_train.py`

Mirrors `modal/train.py` exactly:
- App name: `"nothing-gpt-dpo-train"`
- Same image (`train_image`), GPU (L4), timeout (86400), secrets, volumes
- `VolumeCommitCallback` + `vol.commit()` after training

## Step 4: Serve DPO Adapter

### Modified: `nothing_gpt/serve/server.py`

Replace the single `LORA_CONFIG` with a function that builds a JSON list:

```python
def _lora_modules() -> str:
    modules = [{"name": "seinfeld", "path": ADAPTER_PATH}]
    if os.path.isdir(DPO_ADAPTER_PATH):
        modules.append({"name": "seinfeld-dpo", "path": DPO_ADAPTER_PATH})
    return json.dumps(modules)
```

`os.path.isdir` works on GCS FUSE with the `implicit-dirs` mount option already set in `serve.yaml`. DPO adapter is optional — serve works with just the SFT adapter.

No changes needed to `k8s/serve.yaml` (init container only checks SFT adapter), `modal/serve.py`, or `docker/serve.Dockerfile`.

## New Constants

### Modified: `nothing_gpt/constants.py`

```python
DPO_ADAPTER_PATH = os.environ.get("DPO_ADAPTER_PATH", "/vol/adapters/nothing-gpt-dpo")
DPO_DATA_PATH = os.environ.get("DPO_DATA_PATH", "/vol/data/dpo")
```

## K8s Manifests

### `k8s/dpo-generate-pairs.yaml` — Job (no GPU)

- Image: `us-central1-docker.pkg.dev/nothing-gpt/nothing-gpt/ui:latest`
- `command: ["python", "-m", "nothing_gpt.dpo.generate_pairs"]`
- Env: `SERVE_URL=http://nothing-gpt-serve:8000/v1`
- GCS FUSE mount at `/vol` (read-write)
- Resources: 500m CPU, 512MB RAM
- No init container (serve readiness handled by the serve deployment's readiness probe)

### `k8s/dpo-judge.yaml` — Job (no GPU)

- Image: `us-central1-docker.pkg.dev/nothing-gpt/nothing-gpt/ui:latest`
- `command: ["python", "-m", "nothing_gpt.dpo.judge"]`
- Env: `OPENAI_API_KEY` from k8s secret `openai-secret`
- GCS FUSE mount at `/vol` (read-write)
- Resources: 500m CPU, 512MB RAM
- Init container waits for `/vol/data/dpo/completions.jsonl`

### `k8s/dpo-train.yaml` — Job (L4 GPU)

Mirrors `k8s/sft-train.yaml` with two differences:
- Init container waits for `/vol/data/dpo/train.jsonl`
- `command: ["python3.13", "-m", "nothing_gpt.dpo.train"]`

Reuses `train:latest` Docker image. `trl>=0.15` already includes `DPOTrainer`. The `nothing_gpt` package (including the `dpo/` module) gets installed via `pip install .` in the existing Dockerfile.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `nothing_gpt/constants.py` | Modify | Add `DPO_ADAPTER_PATH`, `DPO_DATA_PATH` |
| `nothing_gpt/dpo/__init__.py` | Create | Package init |
| `nothing_gpt/dpo/generate_pairs.py` | Create | Completion generation (local + GKE) |
| `nothing_gpt/dpo/judge.py` | Create | GPT-5 preference ranking (local + GKE) |
| `nothing_gpt/dpo/train.py` | Create | Core DPO training logic |
| `nothing_gpt/modal/dpo_train.py` | Create | Modal wrapper for DPO training |
| `k8s/dpo-generate-pairs.yaml` | Create | GKE Job (no GPU, uses `ui:latest`) |
| `k8s/dpo-judge.yaml` | Create | GKE Job (no GPU, uses `ui:latest`) |
| `k8s/dpo-train.yaml` | Create | GKE Job (L4 GPU, uses `train:latest`) |
| `nothing_gpt/serve/server.py` | Modify | Conditional DPO LoRA module |
| `pyproject.toml` | Modify | Add `nothing_gpt/dpo` to pyright exclude |

No new Dockerfiles needed. generate_pairs and judge reuse `ui:latest`, DPO training reuses `train:latest`.

## Dependencies

- `trl>=0.15` — already in train image
- `openai` — already in UI image and pyproject.toml
- `OPENAI_API_KEY` — needed for GPT-5 judging

## Risks

1. **DPOTrainer data format**: TRL expects `prompt`, `chosen`, `rejected` columns. Need to verify whether TRL's conversational DPO expects message lists or raw strings — will check during implementation.
2. **`os.path.isdir` on GCS FUSE**: Should work with `implicit-dirs` mount option already set in `serve.yaml`. Low risk.

## Cost Estimates

- **Completions**: Free (our own vLLM endpoint)
- **GPT-5 judging**: ~1,021 calls, estimate ~$5-15
- **DPO training**: ~1-2 hours on L4 ($0.80/hr) = ~$1-2
- **Total**: ~$7-17

## Verification

### GKE flow (end-to-end)
1. `kubectl apply -f k8s/dpo-generate-pairs.yaml` — generates completions to GCS
2. `kubectl apply -f k8s/dpo-judge.yaml` — ranks with GPT-5, writes train/val to GCS
3. `kubectl apply -f k8s/dpo-train.yaml` — DPO training on L4
4. Monitor on WandB — watch `rewards/margins` trending upward
5. Redeploy serve (rebuild `train:latest` to pick up `dpo/` module, then rollout)
6. Test: call API with `model="seinfeld-dpo"`

### Modal flow
1. Generate completions locally:
   ```bash
   SERVE_URL=https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1 \
   DPO_DATA_PATH=data/dpo \
   uv run python -m nothing_gpt.dpo.generate_pairs
   ```
2. Judge locally:
   ```bash
   OPENAI_API_KEY=... DPO_DATA_PATH=data/dpo \
   uv run python -m nothing_gpt.dpo.judge
   ```
3. Upload to volume:
   ```bash
   modal volume put nothing-gpt-vol data/dpo/train.jsonl /data/dpo/train.jsonl
   modal volume put nothing-gpt-vol data/dpo/val.jsonl /data/dpo/val.jsonl
   ```
4. Train: `uv run modal run --detach nothing_gpt.modal.dpo_train::train`
5. Deploy: `uv run modal deploy -m nothing_gpt.modal.serve`
6. Test: call API with `model="seinfeld-dpo"`

### Local flow
Same as Modal steps 1-2, then upload to GCS with `gsutil cp`.
