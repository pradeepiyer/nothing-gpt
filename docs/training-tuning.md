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
| `r` | `32` | Rank of the low-rank matrices. Higher = more capacity, more VRAM, slower training | `16` — eval_loss 2.585 (single-turn); `64` — eval_loss 1.727 (multi-turn, marginal 0.005 gain over r=32 with ~4× more params) |
| `lora_alpha` | `64` | Scaling factor. Effective scaling is `alpha/r` (currently 2.0) | `128` with r=64 (ratio 2.0) |
| `lora_dropout` | `0.1` | Dropout on LoRA layers for regularization | `0.0` — faster, risk of overfitting; `0.05` — less regularization |
| `target_modules` | `"all-linear"` | Which layers get LoRA adapters | Specific names like `["q_proj", "v_proj"]` — fewer trainable params but less expressive |
| `task_type` | `"CAUSAL_LM"` | Tells PEFT the model architecture type | No alternatives for this use case |

**Guidance**: r=64 achieved eval_loss 1.727 vs r=32's 1.732 — a marginal 0.005 improvement despite ~4× more trainable parameters. The extra capacity mainly accelerated overfitting (train-eval gap of 0.97 by epoch 0.95 vs 0.31 for r=32 at epoch 0.47). r=32 is the sweet spot for this dataset. The alpha/rank ratio of 2.0 amplifies adapter contributions; the learning rate is set to `5e-5` to keep effective updates stable (`lr × alpha/r = 5e-5 × 2.0 = 1e-4`). Dropout is 0.1 to help with overfitting.

## Training Loop — `SFTConfig`

### Data Handling

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `max_length` | `2048` | Maximum sequence length in tokens. Conversations are truncated beyond this | `1024` — 35% of examples were truncated at this length (P90=1705, P95=2067); `512` — wastes less memory on short sequences |

### Batch Size and Learning Rate

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `per_device_train_batch_size` | `4` | Micro-batch size per GPU | `8` — OOM with r=64 on L4; `16` — OOM |
| `gradient_accumulation_steps` | `4` | Accumulate gradients over N micro-batches before updating | `2` — effective batch 8, noisier; `1` — effective batch 4, very noisy |
| `learning_rate` | `5e-5` | Peak learning rate. With alpha/r=2.0, effective update magnitude is `5e-5 × 2.0 = 1e-4` | `1e-4` — previous value, more aggressive; `3e-4` — risk of instability |
| `weight_decay` | `0.01` | L2 regularization to combat overfitting over full epoch | `0.0` — previous default, no regularization |
| `lr_scheduler_type` | `"cosine"` | How the learning rate decays over training | `"cosine_with_restarts"` — can escape local minima; `"linear"` — simpler decay curve |
| `warmup_steps` | `200` | Steps of linear LR warmup from 0 to peak | `100` — faster ramp; `500` — more conservative warmup |
| `num_train_epochs` | `1` | Number of passes through the training data | `2` — risk of overfitting (r=32 multi-turn peaked at epoch 0.24); `3` — severe overfitting |

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` = 4 × 4 = **16**.

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
| `save_steps` | `100` | Save a checkpoint every N steps | `200` — fewer checkpoints, less disk I/O |

### Reporting

| Parameter | Current Value | What It Does | Alternatives |
|-----------|--------------|--------------|--------------|
| `logging_steps` | `10` | Log training metrics every N steps | `1` — very verbose; `50` — less noise |
| `report_to` | `"wandb"` | Send metrics to Weights & Biases | `"tensorboard"`, `"none"` |
| `run_name` | `"multiturn-r32-2k"` | Name for the W&B run | Any descriptive string |

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
| `timeout` | `86400` (24 hours) | Modal maximum; 1 epoch on L4 completes in ~6 hours |

## Current Training Math

With the current configuration and dataset (13,482 training examples):

| Metric | Value |
|--------|-------|
| Training examples | 13,482 |
| Micro-batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Steps per epoch | 13,482 ÷ 16 ≈ **843** |
| Total steps (1 epoch) | ~843 |
| Eval/save frequency | Every 100 steps (~12% of an epoch) |
| Warmup | 200 steps (~24% of epoch 1) |

**Best result so far**: eval_loss **1.727** at checkpoint-200 (epoch 0.24), using r=64, alpha=128, max_length=1024, multi-turn format, bf16, cosine schedule on L4. Only marginally better than r=32's 1.732 — the multi-turn data format was the primary driver, not rank (single-turn r=32 achieved only 2.542).

### All Runs

| Run | Format | r | GPU/dtype | Best Eval Loss | Best Epoch |
|-----|--------|---|-----------|---------------|------------|
| r=16 | single-turn | 16 | T4/fp16 | 2.585 | ~0.9 |
| r=32 | single-turn | 32 | T4/fp16 | 2.542 | 0.94 |
| r=32 | multi-turn | 32 | L4/bf16 | 1.732 | 0.24 |
| r=64 | multi-turn | 64 | L4/bf16 | **1.727** | 0.24 |

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

**2. Increase LoRA Rank (r=32 → 64)**
Impact: Low | Cost: More VRAM

Rank increases show diminishing returns: r=16→r=32 improved eval_loss by 0.043 (single-turn), and r=32→r=64 improved by only 0.005 (multi-turn). r=64 also required halving batch size to avoid OOM and overfit significantly faster. r=64 is unlikely to help and would further constrain batch size.

**3. Adjust Alpha/Rank Ratio**
Impact: Medium | Cost: None

The effective LoRA scaling is `alpha/r`. Currently 2.0 (64/32). Reducing to 1.0 would make the adapter more conservative; increasing further risks instability.

**4. Cosine with Warm Restarts**
Impact: Medium | Cost: None

Replace `cosine` scheduler with `cosine_with_restarts`. The LR resets can help escape local minima that a monotonically decaying schedule gets stuck in.

**5. Higher Learning Rate**
Impact: Medium | Cost: Risk of instability

If loss plateaus early, try `3e-4`. More aggressive learning in the critical first epoch, combined with cosine decay. Watch for loss spikes.

**6. Weight Decay**
Impact: Low | Cost: None

Set to 0.01 for L2 regularization. Helps with generalization but unlikely to break a plateau on its own.

### Hardware — GPU Options

Training and serving have different bottlenecks: training is compute-bound (TFLOPS), serving is memory-bandwidth-bound (GB/s).

| | T4 | L4 | A10 |
|---|---|---|---|
| VRAM | 16 GB | 24 GB | 24 GB |
| Memory bandwidth | 320 GB/s | 300 GB/s | 600 GB/s |
| fp16 throughput | ~65 TFLOPS | ~121 TFLOPS | ~125 TFLOPS |
| bf16 support | No | Native | Native |
| $/hour (Modal) | $0.59 | $0.80 (1.35×) | $1.10 (1.87×) |

**Training uses L4.** The L4 has ~2× compute throughput vs T4 at 1.35× the hourly rate, making the total cost per run lower. The 24 GB VRAM enables bf16 training, larger batch sizes, and no gradient checkpointing. The A10 is not worth it for training — similar compute throughput at 1.38× the price.

**Serving uses L4.** The T4 OOM'd with max-model-len=2048 + CUDA graphs + LoRA. The L4's 24 GB VRAM handles the longer context. If response latency matters further, the A10's 2× memory bandwidth would improve token generation speed.

### What NOT to Do

- **Don't increase epochs past 1** — multi-turn data overfits by epoch 0.24–0.47 across both r=32 and r=64
- **Don't reduce effective batch size below 8** — too noisy for stable convergence on this data size
- **Don't chase train loss below eval loss** — that gap is overfitting, not progress
