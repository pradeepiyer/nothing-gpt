# Training Configuration Reference

## Overview

Nothing-GPT fine-tunes Llama 3.2 3B Instruct using **QLoRA** (Quantized Low-Rank Adaptation). The base model is loaded in 4-bit precision, and small trainable adapter matrices are injected into every linear layer. Only the adapters are trained — the base weights stay frozen. This makes it possible to fine-tune a 3B model on a single L4 (24 GB VRAM).

Training runs on Modal with an L4 GPU. The adapter is saved to a shared volume and served via vLLM.

## Quantization — `BitsAndBytesConfig`

These parameters control how the frozen base model is compressed to fit in GPU memory.

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `load_in_4bit` | `True` | Loads base model weights in 4-bit precision (~4× memory reduction) | `False` — full precision, won't fit on L4 |
| `bnb_4bit_quant_type` | `"nf4"` | Quantization data type. NF4 (NormalFloat4) is information-theoretically optimal for normally distributed weights | `"fp4"` — standard 4-bit float, slightly worse quality |
| `bnb_4bit_compute_dtype` | `"bfloat16"` | Precision for matrix multiplications during forward/backward pass | `"float16"` — narrower dynamic range, requires GradScaler |
| `bnb_4bit_use_double_quant` | `True` | Quantizes the quantization constants themselves, saving ~0.4 bits/param | `False` — marginal memory increase, marginal speed improvement |

**Guidance**: These defaults are standard QLoRA settings. The compute dtype should match the training precision (`bf16` or `fp16`).

## LoRA — `LoraConfig`

These parameters control the adapter architecture — the trainable part of the model.

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `r` | `64` | Rank of the low-rank matrices. Higher = more capacity, more VRAM, slower training | `16` — eval_loss 2.585 (single-turn); `32` — eval_loss 2.542 (single-turn), 1.732 (multi-turn) |
| `lora_alpha` | `128` | Scaling factor. Effective scaling is `alpha/r` (currently 2.0) | `64` (ratio 1.0) — more conservative scaling |
| `lora_dropout` | `0.1` | Dropout on LoRA layers for regularization | `0.0` — faster, risk of overfitting; `0.05` — less regularization |
| `target_modules` | `"all-linear"` | Which layers get LoRA adapters | Specific names like `["q_proj", "v_proj"]` — fewer trainable params but less expressive |
| `task_type` | `"CAUSAL_LM"` | Tells PEFT the model architecture type | No alternatives for this use case |

**Guidance**: `r=64` was chosen after `r=16` (eval_loss 2.585), `r=32` single-turn (eval_loss 2.542), and `r=32` multi-turn (eval_loss 1.732) showed that both higher rank and multi-turn data format improve quality. The alpha/rank ratio of 2.0 amplifies adapter contributions; the learning rate is halved to `1e-4` to keep the effective update magnitude constant (`lr × alpha/r = 1e-4 × 2.0 = 2e-4`). Dropout was increased from 0.05 to 0.1 because the r=32 multi-turn run began overfitting by epoch 0.47.

## Training Loop — `SFTConfig`

### Data Handling

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `max_length` | `1024` | Maximum sequence length in tokens. Conversations are truncated beyond this | `512` — wastes less memory on short sequences; `2048` — feasible on L4 |

### Batch Size and Learning Rate

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `per_device_train_batch_size` | `4` | Micro-batch size per GPU | `8` — OOM with r=64 on L4; `16` — OOM |
| `gradient_accumulation_steps` | `4` | Accumulate gradients over N micro-batches before updating | `2` — effective batch 8, noisier; `1` — effective batch 4, very noisy |
| `learning_rate` | `1e-4` | Peak learning rate. Halved from 2e-4 to compensate for alpha/r ratio of 2.0 | `2e-4` — with alpha/r=1.0; `3e-4` — more aggressive, risk of instability |
| `lr_scheduler_type` | `"cosine"` | How the learning rate decays over training | `"cosine_with_restarts"` — can escape local minima; `"linear"` — simpler decay curve |
| `warmup_steps` | `200` | Steps of linear LR warmup from 0 to peak | `100` — faster ramp; `500` — more conservative warmup |
| `num_train_epochs` | `1` | Number of passes through the training data | `2` — risk of overfitting (r=32 multi-turn peaked at epoch 0.24); `3` — severe overfitting |

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` = 8 × 2 = **16**.

### Memory Optimization

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `bf16` | `True` | Mixed-precision training using bfloat16. No GradScaler needed (wider exponent range prevents underflow) | `fp16=True` — narrower dynamic range, requires GradScaler and float32 cast of LoRA params |

### Evaluation and Checkpointing

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `eval_strategy` | `"steps"` | Evaluate at fixed step intervals | `"epoch"` — evaluate once per epoch (too infrequent for monitoring) |
| `eval_steps` | `100` | Run evaluation every N steps | `200` — less disk I/O; `500` — coarse granularity |
| `save_strategy` | `"steps"` | Save checkpoints at fixed step intervals | `"epoch"` — fewer checkpoints |
| `save_steps` | `100` | Save a checkpoint every N steps | Should match `eval_steps` when using `load_best_model_at_end` |
| `load_best_model_at_end` | `True` | After training, load the checkpoint with the best eval metric | `False` — use the final checkpoint regardless |
| `metric_for_best_model` | `"eval_loss"` | Which metric determines "best" | Any logged metric name |

### Reporting

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `logging_steps` | `10` | Log training metrics every N steps | `1` — very verbose; `50` — less noise |
| `report_to` | `"wandb"` | Send metrics to Weights & Biases | `"tensorboard"`, `"none"` |
| `run_name` | `"nothing-gpt-multiturn"` | Name for the W&B run | Any descriptive string |

### Model Initialization

| Parameter | Current Value | What It Does |
|-----------|--------------|--------------|
| `model_init_kwargs.quantization_config` | `bnb_config` | Passes quantization settings to `from_pretrained()` |
| `model_init_kwargs.torch_dtype` | `torch.bfloat16` | Precision for non-quantized model components |

## Runtime

### Why bf16 (not fp16)

BFloat16 has a wider exponent range than float16, which prevents gradient underflow without needing a GradScaler. This avoids a PEFT compatibility issue: with fp16, PyTorch's GradScaler requires trainable parameters in float32, but PEFT creates LoRA adapters in the model's native dtype. The previous fp16 config required an explicit cast of all trainable params to float32 after trainer initialization. With bf16, no cast is needed.

The L4 (Ada Lovelace) has native bf16 hardware support. The previous T4 (Turing) did not.

### Resume Logic

```python
checkpoints = sorted(
    (d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")),
    key=lambda d: int(d.split("-")[1]),
) if os.path.exists(checkpoint_dir) else []
resume_from = os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None
```

If the checkpoint directory exists and contains checkpoints, training resumes from the latest one. This handles Modal preemptions and timeouts gracefully — just re-run the function and it picks up where it left off. The `VolumeCommitCallback` ensures checkpoints are persisted to the shared volume on every save.

### Modal Configuration

| Setting | Value | Why |
|---------|-------|-----|
| GPU | `"L4"` | 24 GB VRAM, native bf16 support, ~2× fp16 throughput vs T4 |
| `timeout` | `120000` (~33 hours) | Conservative upper bound; 1 epoch on L4 should complete well within this |

## Current Training Math

With the current configuration and dataset (13,482 training examples):

| Metric | Value |
|--------|-------|
| Training examples | 13,482 |
| Micro-batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Steps per epoch | 13,482 ÷ 16 ≈ **843** |
| Total steps (1 epoch) | ~843 |
| Eval/save frequency | Every 100 steps (~12% of an epoch) |
| Warmup | 200 steps (~24% of epoch 1) |

**Best result so far**: eval_loss **1.732** at checkpoint-200 (epoch 0.24), using r=32, alpha=32, multi-turn format, bf16, cosine schedule on L4. The multi-turn data format was the primary driver — the same r=32 config with single-turn format achieved only 2.542.

## Tuning Guide — If Loss Plateaus

### When to Worry

Don't optimize prematurely. Signs of a real plateau:
- Eval loss flat across 2+ eval checkpoints (1000+ steps)
- Train loss oscillating in a tight band with no downward drift

With multi-turn format, loss dropped to ~1.7 (vs ~2.5 with single-turn). The floor for this architecture and data may be around 1.5–1.7. If eval loss flattens in that range, that's likely close to the best achievable.

### Levers to Pull (ordered by impact/ease)

**1. Reduce Epochs**
Impact: High | Cost: None

The best multi-turn checkpoint was at epoch 0.24 — overfitting started by epoch 0.47 (eval loss increased while train loss dropped). Even with single-turn format, the best checkpoint was at epoch 0.94. `num_train_epochs=1` is already set; going lower would require `max_steps` instead.

**2. Increase LoRA Rank (r=64 → 128)**
Impact: High | Cost: More VRAM

More rank = more adapter capacity. The progression from r=16 (2.585) to r=32 (2.542) showed gains with higher rank. r=64 is the current setting; r=128 is feasible on L4 but with less VRAM headroom.

**3. Adjust Alpha/Rank Ratio**
Impact: Medium | Cost: None

The effective LoRA scaling is `alpha/r`. Currently 2.0 (128/64). Reducing to 1.0 would make the adapter more conservative; increasing further risks instability.

**4. Cosine with Warm Restarts**
Impact: Medium | Cost: None

Replace `cosine` scheduler with `cosine_with_restarts`. The LR resets can help escape local minima that a monotonically decaying schedule gets stuck in.

**5. Higher Learning Rate**
Impact: Medium | Cost: Risk of instability

If loss plateaus early, try `3e-4`. More aggressive learning in the critical first epoch, combined with cosine decay. Watch for loss spikes.

**6. Weight Decay**
Impact: Low | Cost: None

Currently defaults to 0. Adding `weight_decay=0.01` can improve generalization slightly but unlikely to break a plateau.

### Hardware — GPU Options

Training and serving have different bottlenecks: training is compute-bound (TFLOPS), serving is memory-bandwidth-bound (GB/s).

| | T4 | L4 (training) | A10 |
|---|---|---|---|
| VRAM | 16 GB | 24 GB | 24 GB |
| Memory bandwidth | 320 GB/s | 300 GB/s | 600 GB/s |
| fp16 throughput | ~65 TFLOPS | ~121 TFLOPS | ~125 TFLOPS |
| bf16 support | No | Native | Native |
| $/hour (Modal) | $0.59 | $0.80 (1.35×) | $1.10 (1.87×) |

**Training uses L4.** The L4 has ~2× compute throughput vs T4 at 1.35× the hourly rate, making the total cost per run lower. The 24 GB VRAM enables bf16 training, larger batch sizes, and no gradient checkpointing. The A10 is not worth it for training — similar compute throughput at 1.38× the price.

**Serving uses T4.** For a demo app with light traffic, the T4 is cheapest. If response latency matters, the A10's 2× memory bandwidth would improve token generation speed. The A10 would also allow dropping `--enforce-eager` and bumping `--max-model-len` to 2048.

### What NOT to Do

- **Don't increase epochs past 3** without evidence that eval loss is still declining at epoch boundary — dialogue data overfits fast
- **Don't reduce effective batch size below 8** — too noisy for stable convergence on this data size
- **Don't chase train loss below eval loss** — that gap is overfitting, not progress
