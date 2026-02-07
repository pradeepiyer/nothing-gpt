# Training Tuning Guide — If Loss Plateaus

## Current Config (train.py)

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 (alpha/r = 1.0) |
| LoRA dropout | 0.05 |
| LoRA targets | all-linear |
| Effective batch size | 16 (4 × 4 accum) |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Warmup | 200 steps |
| Epochs | 3 |
| Max length | 512 |
| GPU | T4 (16GB) |

## When to Worry

Don't optimize prematurely. Signs of a real plateau:
- Eval loss flat across 2+ eval checkpoints (1000+ steps)
- Train loss oscillating in a tight band with no downward drift

A loss floor around 2.3–2.5 may be natural for dialogue data — next tokens in conversation are inherently unpredictable. If eval reaches ~2.4 and flattens, that's probably close to the best this config can do.

## Levers to Pull (ordered by impact/ease)

### 1. Increase LoRA Rank (r=16 → 32 or 64)
**Impact: High | Requires: New run**

More rank = more adapter capacity. If the model has learned everything 16 dimensions can represent, this is the most direct fix. Trade-off: ~2× VRAM for adapter params, slightly slower training. Should still fit on T4 at r=32.

### 2. Increase Alpha/Rank Ratio (alpha=32, keep r=16)
**Impact: Medium | Requires: New run**

The effective LoRA scaling is `alpha/r`. Currently 1.0. Bumping alpha to 32 (ratio 2.0) amplifies the adapter's contribution, giving a stronger learning signal without increasing parameter count. Common recommendation in QLoRA literature.

### 3. Reduce Epochs, Increase LR
**Impact: Medium | Requires: New run**

If loss plateaus early in epoch 1, the model might benefit from a higher LR (3e-4) with fewer epochs (1-2). More aggressive learning early, then stop before overfitting. Watch eval loss carefully.

### 4. Cosine with Warm Restarts
**Impact: Medium | Requires: New run**

Replace `cosine` scheduler with `cosine_with_restarts`. The LR resets can help escape local minima that a monotonically decaying schedule gets stuck in.

### 5. Check Max Length vs Actual Data
**Impact: Low-Medium | Requires: Investigation**

If most training examples are much shorter than 512 tokens, compute is wasted on padding. If some are truncated, we're losing information. Run a quick check on the JSONL to see the token length distribution. Adjusting max_length to match the data can help.

### 6. Weight Decay
**Impact: Low | Requires: New run**

Currently defaults to 0. Adding `weight_decay=0.01` can improve generalization slightly but unlikely to break a plateau.

## What NOT to Do

- **Don't increase epochs past 3** without evidence that eval loss is still declining at epoch boundary — dialogue data overfits fast
- **Don't reduce batch size below 8** — too noisy for stable convergence on this data size
- **Don't chase train loss below eval loss** — that gap is overfitting, not progress

## Decision

No action now. Let the current run complete. Revisit if eval loss flattens before step 3000 (~midpoint). The first thing to try would be r=32 with alpha=32.
