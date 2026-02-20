# DPO Preference Optimization — Configuration and Tuning Guide

## Overview

DPO (Direct Preference Optimization) trains the model to prefer higher-quality completions over lower-quality ones, using preference pairs ranked by an LLM judge. Unlike SFT which maximizes likelihood of training examples, DPO shifts the model's probability distribution toward chosen completions and away from rejected ones, relative to a frozen reference model.

The pipeline has three stages: **generate completions** → **judge preferences** → **train DPO**. Generate and judge run locally; training runs on GKE with an L4 GPU.

## Pipeline

```
SFT val.jsonl (1,021 prompts)
    → generate_pairs.py: 3 completions per prompt via vLLM endpoint
    → judge.py: LLM judge ranks completions → chosen/rejected pairs
    → train.py: DPO training on preference pairs
    → Serve DPO adapter (model="seinfeld")
```

## Stage 1: Completion Generation

### `nothing_gpt/dpo/generate_pairs.py`

Generates diverse completions from the SFT model for each validation prompt. These completions are the raw material the judge will rank.

#### Parameters

| Parameter | Value | Impact | Tuning |
|-----------|-------|--------|--------|
| `TEMPERATURE` | `0.9` | Controls diversity between completions. Higher = more varied. | Below 0.7, completions become too similar for the judge to distinguish. Above 1.0, quality degrades (incoherent text). 0.8–1.0 is the useful range for DPO. |
| `NUM_COMPLETIONS` | `3` | Completions per prompt. Judge picks best and worst from this set. | 3 is the minimum for meaningful ranking. 5 gives better coverage but 2.5× more API calls and judging cost. Diminishing returns above 5. |
| `max_tokens` | `256` | Maximum tokens per completion. | Matches the generation budget used in the UI. Increasing produces longer scripts but the judge criteria don't reward length. |
| `frequency_penalty` | `0.5` | Penalizes token repetition within a completion. | Prevents the model from getting stuck in loops (a common failure mode at high temperature). 0.3–0.7 is reasonable. |
| `BATCH_SIZE` | `8` | Concurrent async API requests. | Bounded by vLLM's `max-num-seqs` (32). Higher batches improve throughput but may increase latency per request. 8 is conservative. |
| `model` | `"seinfeld"` | The SFT LoRA adapter served by vLLM. | Must match the adapter name in `server.py`. |

#### Output

`completions.jsonl` — 1,021 rows, each with `{prompt: [...], completions: [str, str, str]}`.

Supports resume: if the output file already exists, appends from where it left off.

#### Run

```bash
SERVE_URL=https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1 \
DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.generate_pairs
```

## Stage 2: Preference Judging

### `nothing_gpt/dpo/judge.py`

An LLM judge ranks each prompt's 3 completions to produce (chosen, rejected) preference pairs.

#### Judging Criteria

The judge evaluates on a single composite criterion: **faithfulness to Seinfeld scripts and characters**:
- Does the dialogue sound like it belongs in an actual Seinfeld episode?
- Does each character speak in their distinctive voice (Jerry's observational style, George's neurotic complaints, Elaine's sarcasm, Kramer's physicality and wild ideas)?
- Is the dialogue funny, coherent, and relevant to the scene context?

The judge returns `{best: <1|2|3>, worst: <1|2|3>}`. The best completion becomes "chosen", the worst becomes "rejected", and the middle one is discarded.

#### Parameters

| Parameter | Value | Impact | Tuning |
|-----------|-------|--------|--------|
| `JUDGE_MODEL` | `gemini-2.5-flash` | Which LLM ranks the completions. | Stronger models produce more consistent rankings. Gemini 2.5 Flash is used via the OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai/`. GPT-4o or Claude would also work but at higher cost. |
| `temperature` | `0` | Deterministic judging. | Should stay at 0. Non-zero temperature introduces ranking noise that directly degrades preference pair quality. |
| `max_tokens` | `1024` | Token budget for judge response. | Gemini 2.5 Flash is a thinking model — it uses tokens for internal reasoning before producing the JSON response. 50 tokens caused empty responses. 1024 gives ample room. Non-thinking models can use 50. |
| `VAL_RATIO` | `0.1` | Fraction of pairs held out for evaluation. | 10% gives 102 eval pairs — enough to track reward accuracy without wasting training data. |
| `SEED` | `42` | Random seed for shuffle and split. | Fixed for reproducibility. Change if you want a different train/val partition. |
| `BATCH_SIZE` | `16` | Concurrent async judge API requests. | Bounded by API rate limits. Gemini's free tier has low rate limits; paid tier handles 16 easily. |

#### Gemini-Specific Handling

1. **Markdown code fences**: Gemini wraps JSON responses in `` ```json ... ``` ``. The code strips these with a regex before parsing.
2. **Thinking tokens**: Gemini 2.5 Flash uses tokens for chain-of-thought reasoning. The `max_tokens` budget must account for this (1024 vs 50 for non-thinking models).

#### Output

- `train.jsonl` — 919 preference pairs
- `val.jsonl` — 102 preference pairs

Each row: `{prompt: [{role, content}, ...], chosen: [{role: "assistant", content: ...}], rejected: [{role: "assistant", content: ...}]}`

This is TRL's conversational DPO format.

#### Data Quality

From the first run (Gemini 2.5 Flash, 1,021 prompts):
- **0 failures** — every prompt successfully judged
- **0 degenerate pairs** — no cases where chosen == rejected or either was empty
- **53.7% longer-is-chosen** — minimal length bias (random would be 50%)
- Balanced character distribution across chosen and rejected

#### Run

```bash
OPENAI_API_KEY=<gemini-key> \
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.judge
```

## Stage 3: DPO Training

### `nothing_gpt/dpo/train.py`

Trains the SFT adapter to prefer chosen completions over rejected ones using the DPO objective.

### How DPO Works

The DPO loss function is:

```
L_DPO = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
```

Where `y_w` = chosen, `y_l` = rejected, `π_θ` = trainable model, `π_ref` = frozen reference model. The loss pushes the model to increase the log-probability gap between chosen and rejected, relative to the reference model.

### Model Setup

The SFT adapter is loaded onto the quantized base model with `is_trainable=True`. DPOTrainer internally creates a frozen copy as the reference model.

```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config)
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
```

**Why not dual adapters?** The TRL docs suggest loading the SFT adapter twice (one trainable, one frozen) with `model_adapter_name` / `ref_adapter_name`. This failed with quantized models:
1. `adapter_name="train"` collides with PyTorch's `nn.Module.train()` method
2. Gradient checkpointing is incompatible with dual adapters (tensor shape mismatch during backward recomputation)
3. `requires_grad` errors with quantized base weights

The reference model copy approach uses more VRAM (~1.7GB for a 3B model) but works reliably.

### Quantization — `BitsAndBytesConfig`

Same as SFT. See `docs/sft-pipeline.md`.

| Parameter | Value |
|-----------|-------|
| `load_in_4bit` | `True` |
| `bnb_4bit_quant_type` | `"nf4"` |
| `bnb_4bit_compute_dtype` | `"bfloat16"` |
| `bnb_4bit_use_double_quant` | `True` |

### Training — `DPOConfig`

| Parameter | Value | Impact | Tuning |
|-----------|-------|--------|--------|
| `beta` | `0.1` | KL penalty strength. Controls how far the model can drift from the reference. | **The most important DPO hyperparameter.** Lower beta (0.01–0.05) allows larger policy shifts — the model more aggressively changes its behavior, which can improve reward accuracy but risks forgetting SFT quality. Higher beta (0.3–0.5) constrains the model close to SFT — safer but weaker preference learning. 0.1 is the standard starting point. If rewards/accuracies is low (< 0.55), try decreasing beta to 0.05. If generations degrade in quality despite good reward accuracy, increase beta. |
| `loss_type` | `"sigmoid"` | The DPO loss variant. | `"sigmoid"` is standard DPO (Rafailov et al. 2023). Alternatives: `"hinge"` (sharper gradient, less stable), `"ipo"` (identity preference optimization, different scaling). Sigmoid is the most studied and robust. |
| `learning_rate` | `5e-6` | Peak learning rate. | 6× lower than SFT's 3e-5. DPO is more sensitive to LR because it's adjusting a fine-tuned model, not training from a pretrained base. Going higher (1e-5) risks divergence. Going lower (1e-6) would need more epochs. With 919 training examples and 58 steps per epoch, 5e-6 gives a conservative update budget. |
| `num_train_epochs` | `1` | Passes through the training data. | With only 919 examples, a single epoch is 58 steps (at effective batch size 16). Multiple epochs risk overfitting on such a small dataset. If increasing the dataset size substantially (5,000+), 2–3 epochs may be worthwhile. |
| `lr_scheduler_type` | `"cosine"` | LR decay schedule. | Cosine decays smoothly to near-zero by the end of training. Matches SFT. Linear works too but cosine gives a slower start and gentler finish. |
| `warmup_steps` | `20` | Steps of linear LR warmup from 0 to peak. | 20 steps = 34% of a 58-step epoch. This is high — typical warmup is 5–10%. But with so few total steps, a long warmup prevents early instability. Could reduce to 10 if training more steps. |
| `per_device_train_batch_size` | `2` | Micro-batch size. | Constrained by VRAM. DPO processes two sequences per example (chosen + rejected) plus the reference model copy, so each example costs ~2× the VRAM of an SFT example. Batch size 4 OOM'd. |
| `per_device_eval_batch_size` | `1` | Eval micro-batch size. | Eval materializes logits for both chosen and rejected sequences. 1 is conservative but ensures no OOM during eval. |
| `gradient_accumulation_steps` | `8` | Accumulation steps before an optimizer update. | Effective batch size = 2 × 8 = **16**. Matches SFT's effective batch size. Smaller effective batch sizes (< 8) produce noisy gradients; larger (> 32) waste steps on a 58-step epoch. |
| `max_length` | `2048` | Maximum sequence length for chosen and rejected. | Matches SFT's training length. Sequences longer than this are truncated. Our data has P99=1062 tokens, so 2048 gives ample headroom. |
| `bf16` | `True` | Use bfloat16 precision. | Required for L4 (native bf16 support). Eliminates need for GradScaler. |
| `gradient_checkpointing` | `False` | Recompute activations during backward pass to save VRAM. | **Must be False for DPO with reference model copy.** Gradient checkpointing is incompatible with the internal reference model copy that DPOTrainer creates. Enabling it causes `CheckpointError: Recomputed values for tensors have different metadata`. |
| `eval_steps` | `25` | Evaluate every N steps. | 25 steps = ~43% of the epoch. Gives 2 eval checkpoints during training. For a longer run, every 50 steps would be sufficient. |
| `save_steps` | `25` | Checkpoint every N steps. | Aligned with eval_steps. Allows resuming from the best checkpoint if spot-preempted. |
| `logging_steps` | `10` | Log metrics to WandB every N steps. | 10 steps = 6 log entries per epoch. Enough to see trends without excessive logging. |

### Training Math

| Metric | Value |
|--------|-------|
| Training examples | 919 |
| Effective batch size | 16 (2 × 8) |
| Steps per epoch | 919 ÷ 16 ≈ **58** |
| Warmup | 20 steps (34% of epoch) |
| Eval/save frequency | Every 25 steps (~2 evals per epoch) |
| Total optimizer updates | ~58 |

### Key Metrics to Monitor

| Metric | Meaning | Healthy Range |
|--------|---------|---------------|
| `rewards/accuracies` | Fraction where model assigns higher reward to chosen vs rejected. | > 0.55 means the model is learning preferences. > 0.70 is strong. |
| `rewards/margins` | Mean difference between chosen and rejected rewards. | Should be positive and increasing. Negative means the model prefers rejected. |
| `rewards/chosen` | Implicit reward for chosen completions. | Should increase or stay stable. |
| `rewards/rejected` | Implicit reward for rejected completions. | Should decrease relative to chosen. |
| `logps/chosen` | Log-probability of chosen under the policy. | Increasing means the model assigns more probability to chosen completions. |
| `logps/rejected` | Log-probability of rejected under the policy. | Should decrease or increase less than chosen. |
| `eval_loss` | DPO loss on validation set. | Decreasing is good. Watch for divergence from train loss (overfitting). |

### First Run Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Train loss | 1.547 | — |
| Eval loss | 1.588 | Close to train loss, no overfitting |
| Eval rewards/accuracies | 0.461 | Below 0.5 — model is not learning to prefer chosen |
| Eval rewards/margins | -0.345 | Negative — model slightly prefers rejected |
| Train runtime | 808s (~13.5 min) | — |

The first run completed without errors but the model did not learn meaningful preferences. This is a known challenge with small datasets and weak preference signal.

### What to Try Next

Listed in order of expected impact:

1. **Lower beta (0.05 or 0.01)**: With weak preference signal, the current beta=0.1 may constrain the model too tightly to the reference. A lower beta allows larger policy shifts per example, which can help when the chosen/rejected gap is small. Risk: too-low beta can cause the model to drift far from SFT quality.

2. **Stronger judge model**: Gemini 2.5 Flash may not produce rankings with enough quality separation. A stronger judge (Gemini 2.5 Pro, GPT-4o, Claude Sonnet) could produce sharper chosen/rejected distinctions, giving the DPO loss a clearer gradient signal. Cost: ~$10–30 for 1,021 judge calls.

3. **More completions per prompt (5 instead of 3)**: More completions increase the quality gap between best and worst. With 3 completions, the best and worst may be close in quality. With 5, the extremes are more pronounced. Cost: ~2× more vLLM generation time.

4. **Higher generation temperature (1.0–1.2)**: Increases diversity of completions, making it easier for the judge to find clear winners and losers. Risk: temperatures above 1.2 degrade coherence.

5. **More training data**: Generate and judge more prompts. Currently using only the 1,021 validation prompts from SFT. Could also sample from the 9,007 training prompts. More data gives the model more examples to learn from, reducing the chance of overfitting on noise.

6. **Multiple epochs**: With only 58 optimizer updates, the model may not have enough steps to learn. Running 2–3 epochs with a lower LR (1e-6) would give 116–174 updates. Only viable if combined with a lower LR to avoid overfitting.

7. **Adjust warmup**: 20 warmup steps out of 58 total means the model only trains at full LR for 38 steps. Reducing warmup to 5–10 gives more steps at peak LR.

## Serving the DPO Adapter

### `nothing_gpt/serve/server.py`

The serve module serves the DPO adapter as `model="seinfeld"`:

```python
lora_module = json.dumps({"name": "seinfeld", "path": ADAPTER_PATH})
```

`ADAPTER_PATH` defaults to `/vol/adapters/nothing-gpt` (the DPO adapter). The SFT intermediate is at `/vol/adapters/nothing-gpt-sft`.

## OOM Issues Encountered

| Situation | Cause | Fix |
|-----------|-------|-----|
| Dual-adapter DPO | Reference adapter + train adapter + base model | Abandoned dual adapters; let DPOTrainer create reference copy |
| `per_device_train_batch_size=4` | DPO processes chosen + rejected per example (2× SFT) plus reference model | Reduced to batch_size=2, grad_accum=8 |
| `gradient_checkpointing=True` | Incompatible with DPOTrainer's internal reference model | Set to `False` |

## Constants

```python
SFT_ADAPTER_PATH = os.environ.get("SFT_ADAPTER_PATH", "/vol/adapters/nothing-gpt-sft")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/vol/adapters/nothing-gpt")
DPO_DATA_PATH = os.environ.get("DPO_DATA_PATH", "/vol/data/dpo")
```

## Running the Pipeline

### Local (generate + judge) → GKE (train)

```bash
# 1. Generate completions (~20 min against Modal endpoint)
SERVE_URL=https://pradeepiyer--nothing-gpt-serve-serve.modal.run/v1 \
DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.generate_pairs

# 2. Judge with Gemini (~5 min)
OPENAI_API_KEY=<gemini-key> \
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
DPO_DATA_PATH=data/dpo \
uv run python -m nothing_gpt.dpo.judge

# 3. Upload data to GCS
gsutil cp data/dpo/train.jsonl data/dpo/val.jsonl gs://nothing-gpt-data/data/dpo/

# 4. Build and submit training job
gcloud builds submit --config /dev/stdin . <<'EOF'
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'us-central1-docker.pkg.dev/nothing-gpt/nothing-gpt/train:latest',
           '-f', 'docker/train.Dockerfile', '.']
images:
  - 'us-central1-docker.pkg.dev/nothing-gpt/nothing-gpt/train:latest'
EOF

kubectl apply -f k8s/dpo-train.yaml

# 5. Monitor on WandB — watch rewards/accuracies and rewards/margins
# 6. Test: call API with model="seinfeld"
```

### Modal

```bash
# Steps 1-2 same as above, then:
modal volume put nothing-gpt-vol data/dpo/train.jsonl /data/dpo/train.jsonl
modal volume put nothing-gpt-vol data/dpo/val.jsonl /data/dpo/val.jsonl
uv run modal run --detach nothing_gpt.modal.dpo_train::train
uv run modal deploy -m nothing_gpt.modal.serve
```

## Cost

| Stage | Cost |
|-------|------|
| Completion generation | Free (own vLLM endpoint) |
| Gemini 2.5 Flash judging (1,021 calls) | ~$0.50 |
| DPO training (~13 min on L4 spot) | ~$0.20 |
| **Total** | **~$0.70** |
