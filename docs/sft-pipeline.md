# SFT Training — Configuration and History

## Overview

Nothing-GPT fine-tunes Llama 3.2 3B Instruct using **QLoRA** (Quantized Low-Rank Adaptation). The base model is loaded in 4-bit precision, and small trainable adapter matrices are injected into every linear layer. Only the adapters are trained — the base weights stay frozen. This makes it possible to fine-tune a 3B model on a single L4 (24 GB VRAM).

Training runs on Modal with an L4 GPU. The adapter is saved to a shared volume and served via vLLM.

## Current Configuration

### Data Format: Script-Continuation

The model is trained on a **script-continuation** format: a single system prompt (`SCRIPT_PROMPT`) followed by a sliding window of Seinfeld dialogue. The model learns to output `[CHARACTER] dialogue` for all characters — it plays the entire cast, not a single character.

- **Sliding window**: WINDOW_SIZE=35 turns, CONTEXT_TURNS=5 (prompt), MIN_COMPLETION_TURNS=10, STRIDE=5
- **Training data**: 9,007 train / 1,021 val examples (after character name filtering)
- **Token stats** (tiktoken cl100k proxy): min=315, max=1506, mean=731, P90=871, P95=928, P99=1062 — 0% truncated at max_length=2048
- **Format**: prompt/completion JSONL (`{prompt, completion}`) split from messages for `completion_only_loss`

This replaced the earlier per-character multi-turn format, which trained the model to speak as one character at a time. The script-continuation format enables the scene generator UI where the model produces dialogue for all characters in a scene.

### Quantization — `BitsAndBytesConfig`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `load_in_4bit` | `True` | ~4× memory reduction for base weights |
| `bnb_4bit_quant_type` | `"nf4"` | Information-theoretically optimal for normally distributed weights |
| `bnb_4bit_compute_dtype` | `"bfloat16"` | Matches training precision |
| `bnb_4bit_use_double_quant` | `True` | Quantizes the quantization constants, saves ~0.4 bits/param |

These are standard QLoRA defaults. No reason to change them.

### LoRA — `LoraConfig`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` | `32` | Rank of low-rank matrices |
| `lora_alpha` | `32` | Scaling factor. Effective scaling = alpha/r = 1.0 |
| `lora_dropout` | `0.1` | Regularization on LoRA layers |
| `target_modules` | `"all-linear"` | Adapters on every linear layer |
| `task_type` | `"CAUSAL_LM"` | Causal language model architecture |

### Training — `SFTConfig`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_length` | `2048` | 0% of examples truncated at this length |
| `completion_only_loss` | `True` | Loss computed only on completion tokens, not prompt |
| `num_train_epochs` | `1` | Single pass through training data |
| `learning_rate` | `3e-5` | With alpha/r=1.0, effective update magnitude = 3e-5 |
| `weight_decay` | `0.005` | Light L2 regularization |
| `lr_scheduler_type` | `"cosine"` | Smooth decay to near-zero by end of epoch |
| `warmup_steps` | `50` | Linear warmup from 0 to peak LR |
| `per_device_train_batch_size` | `4` | Micro-batch size per GPU |
| `per_device_eval_batch_size` | `2` | Reduced to avoid OOM during eval |
| `gradient_accumulation_steps` | `4` | Effective batch size = 4 × 4 = **16** |
| `bf16` | `True` | Native bf16 on L4; no GradScaler needed |
| `eval_steps` | `50` | Evaluate every 50 steps |
| `save_steps` | `50` | Checkpoint every 50 steps |
| `logging_steps` | `10` | Log to W&B every 10 steps |

### Training Math

| Metric | Value |
|--------|-------|
| Training examples | 9,007 |
| Effective batch size | 16 |
| Steps per epoch | 9,007 ÷ 16 ≈ **563** |
| Eval/save frequency | Every 50 steps (~9% of an epoch) |
| Warmup | 50 steps (~9% of epoch) |

## Final Result

**Eval loss 2.187** (completion-only) at epoch 0.98 — no overfitting, monotonically decreasing across the full epoch. Train-eval gap only 0.09.

Note: completion-only eval loss is not comparable to full-loss eval loss (2.187 completion-only vs 1.732 full-loss in earlier runs). Completion-only loss ignores the prompt tokens where the model gets "easy" predictions, so the number is naturally higher.

## How We Got Here

The final configuration was reached through several iterations, each addressing a specific problem.

### Phase 1: Single-Turn Format, T4/fp16

The first runs used a single-turn chat format (one user message → one assistant response) on a T4 GPU with fp16 precision.

| Run | r | Best Eval Loss | Best Epoch | Outcome |
|-----|---|---------------|------------|---------|
| r=16 | 16 | 2.585 | ~0.9 | Plateaued, then overfit by epoch 1.2 |
| r=32 | 32 | 2.542 | 0.94 | Slightly better, still high loss |

**Problems**: T4 had no native bf16, requiring a float32 cast workaround for LoRA params. T4 also OOM'd during eval at max_length=1024. The single-turn format limited context severely.

### Phase 2: Multi-Turn Format, L4/bf16

Switched to multi-turn chat format (5-turn context window) and upgraded to L4 for bf16 support and 24GB VRAM.

| Run | r | Best Eval Loss | Best Epoch | Outcome |
|-----|---|---------------|------------|---------|
| r=32 | 32 | 1.732 | 0.24 | Overfit by epoch 0.47 |
| r=64 | 64 | 1.727 | 0.24 | Marginal 0.005 gain, severe overfitting, 4× more params |

**Key finding**: Multi-turn format dropped eval loss from 2.5 → 1.7, a massive improvement. The format change mattered far more than rank. But both runs overfit early — the model peaked at epoch 0.24 and degraded from there.

**r=64 verdict**: Not worth it. 4× more trainable parameters for 0.005 better eval loss, faster overfitting, and OOM at batch_size=8.

### Phase 3: Script-Continuation Format, Completion-Only Loss

Replaced multi-turn chat with script-continuation format to enable the scene generator UI. Added `completion_only_loss=True` so the model is only trained on generating dialogue, not predicting the prompt.

First run used lr=1e-4 and alpha=64 (alpha/r=2.0):

| Config | Best Eval Loss | Best Epoch | Outcome |
|--------|---------------|------------|---------|
| lr=1e-4, alpha=64 | 2.187 | 0.27 | Overfit to 2.236 by epoch 0.98 |

The model hit 2.187 early but then degraded — same overfitting pattern as Phase 2.

### Phase 4: Final Tuning — Eliminating Overfitting

**Root cause**: lr=1e-4 was too aggressive for fine-tuning a 3B model. With alpha/r=2.0, the effective update magnitude was 1e-4 × 2.0 = 2e-4. The model learned the training distribution quickly but overshot the optimum.

**Fix**: Reduced lr from 1e-4 to 3e-5 and alpha from 64 to 32 (alpha/r from 2.0 to 1.0). Combined effect: ~6× slower weight updates (3e-5 × 1.0 = 3e-5 vs 1e-4 × 2.0 = 2e-4).

| Config | Best Eval Loss | Best Epoch | Outcome |
|--------|---------------|------------|---------|
| lr=3e-5, alpha=32 | **2.187** | 0.98 | No overfitting, monotonically decreasing |

Same best eval loss (2.187), but achieved at epoch 0.98 instead of 0.27 — the model used the full epoch productively instead of overfitting for the last 75% of training. The train-eval gap was only 0.09 at completion, confirming no overfitting.

## OOM Issues Encountered

| Situation | Cause | Fix |
|-----------|-------|-----|
| T4 eval at step 200, max_length=1024 | T4 has only 16GB VRAM | Upgraded to L4 |
| L4 eval at step 100, max_length=2048, eval_batch=4 | Eval materializes full sequence | Reduced eval_batch to 2 |
| completion_only_loss with batch_size=8 on L4 | `shift_logits` materializes full logit tensor | Reduced batch_size to 4 |
| r=64 with batch_size=8 on L4 | 4× more trainable params | Reduced batch_size to 4, grad_accum to 4 |

**SFTTrainer pads to longest-in-batch**, not max_length. Reducing max_length doesn't help if the longest sequence in the batch is still long.

## Why bf16 (not fp16)

BFloat16 has a wider exponent range than float16, which prevents gradient underflow without needing a GradScaler. With fp16, PyTorch's GradScaler requires trainable parameters in float32, but PEFT creates LoRA adapters in the model's native dtype. The fp16 config required an explicit cast of all trainable params to float32 after trainer initialization. With bf16, no cast is needed.

L4 (Ada Lovelace) has native bf16 hardware support. T4 (Turing) does not.

## Resume Logic

If the checkpoint directory exists and contains checkpoints, training resumes from the latest one. This handles Modal preemptions and timeouts — re-run the function and it picks up where it left off. The `VolumeCommitCallback` ensures checkpoints are persisted to the shared volume on every save.

## Hardware

| | T4 | L4 | A10 |
|---|---|---|---|
| VRAM | 16 GB | 24 GB | 24 GB |
| Memory bandwidth | 320 GB/s | 300 GB/s | 600 GB/s |
| fp16 throughput | ~65 TFLOPS | ~121 TFLOPS | ~125 TFLOPS |
| bf16 support | No | Native | Native |
| $/hour (Modal) | $0.59 | $0.80 | $1.10 |

**Training uses L4.** ~2× compute throughput vs T4 at 1.35× the hourly rate, making the total cost per run lower. 24 GB VRAM enables bf16, larger batch sizes, and 2048 context.

**Serving uses L4.** T4 OOM'd with max-model-len=2048 + CUDA graphs + LoRA. If response latency matters, A10's 2× memory bandwidth would improve token generation speed.

## What NOT to Do

- **Don't increase epochs past 1** — the model uses the full epoch productively with current LR; more would overfit
- **Don't raise alpha/r above 1.0** — ratio of 2.0 caused overfitting with any LR ≥ 1e-4
- **Don't use r=64** — 4× more params for 0.005 better loss, not worth it
- **Don't reduce effective batch size below 8** — too noisy for stable convergence
- **Don't chase train loss below eval loss** — that gap is overfitting, not progress
