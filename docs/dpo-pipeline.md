# DPO Preference Optimization Pipeline

## Context

SFT training plateaued at eval_loss 2.187 (completion-only) with no overfitting — the model has learned the training distribution well. Further SFT epochs would yield diminishing returns (eval loss flatlined for the last 40% of epoch 1 due to cosine LR decay, and the train-eval gap is only 0.09).

To improve output quality beyond what SFT provides, we'll add a DPO (Direct Preference Optimization) stage. This trains the model to prefer higher-quality completions (better character voice, humor, coherence, premise relevance) using preference pairs judged by GPT-5.

## Pipeline Overview

```
Prompts from val.jsonl
    → Batch inference (current SFT adapter, 3 completions per prompt)
    → GPT-5 judges pairs → (chosen, rejected)
    → DPO training (new LoRA adapter starting from SFT adapter)
    → Deploy DPO adapter
```

## Step 1: Preference Data Generation

### New file: `src/nothing_gpt/dpo/generate_pairs.py`

Generate multiple completions per prompt using the deployed vLLM endpoint:

- Pull prompts from `data/training/val.jsonl` (1,021 examples)
- For each prompt, generate 3 completions at temperature=0.9 (diverse sampling)
- Produces 3 completions × 1,021 prompts = 3,063 completions
- Output: `data/dpo/completions.jsonl` — each row has `{prompt, completions: [str, str, str]}`

Uses the existing vLLM serve endpoint (`SERVE_URL` from `config.py`) and the OpenAI client pattern already in `ui.py`.

## Step 2: GPT-5 Judging

### New file: `src/nothing_gpt/dpo/judge.py`

For each prompt's 3 completions, ask GPT-5 to rank them on a single criterion:
- **Faithfulness to Seinfeld scripts and characters** — Does the dialogue sound like it belongs in an actual Seinfeld episode? Does each character speak in their distinctive voice (Jerry's observational style, George's neurotic complaints, Elaine's sarcasm, Kramer's physicality and wild ideas)?

From each set of 3 completions, extract the best (chosen) and worst (rejected) → 1 preference pair per prompt.

- Input: `data/dpo/completions.jsonl`
- Output: `data/dpo/preferences.jsonl` — each row has `{prompt, chosen, rejected}`
- Uses OpenAI SDK with `gpt-5` model
- Requires `OPENAI_API_KEY` environment variable
- ~1,021 API calls to GPT-5

## Step 3: DPO Training

### New file: `src/nothing_gpt/modal/dpo_train.py`

Separate Modal app/function for DPO training using TRL's `DPOTrainer`:

```python
# Load base model with QLoRA (same as SFT)
# Load SFT adapter as both "train" and "reference"
model = PeftModel.from_pretrained(base, sft_path, adapter_name="train")
model.load_adapter(sft_path, adapter_name="reference")

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

Memory: Fits on L4 (24GB). Second adapter is ~150MB, not a full model copy.

## Step 4: Train/Val Split on Preference Data

Split the preference pairs 90/10 for DPO training and evaluation:
- ~919 train pairs, ~102 val pairs (from the 1,021 prompts)
- Split by prompt to avoid data leakage
- Output: `data/dpo/train.jsonl` and `data/dpo/val.jsonl`
- Can be done in `judge.py` or as a separate step

## Step 5: Upload Preferences to Modal Volume

Upload `data/dpo/train.jsonl` and `data/dpo/val.jsonl` to the Modal volume before training. Can use `modal volume put` or add to the existing data upload flow.

## Step 6: Serve DPO Adapter

### Modified file: `src/nothing_gpt/modal/serve.py`

Add the DPO adapter as a separate model name `"seinfeld-dpo"` alongside the existing `"seinfeld"` (SFT) model. This enables A/B comparison at inference time.

```python
--lora-modules '{"name": "seinfeld", "path": "/vol/adapters/nothing-gpt"}'
               '{"name": "seinfeld-dpo", "path": "/vol/adapters/nothing-gpt-dpo"}'
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/nothing_gpt/dpo/__init__.py` | Create | Package init |
| `src/nothing_gpt/dpo/generate_pairs.py` | Create | Batch inference for multiple completions |
| `src/nothing_gpt/dpo/judge.py` | Create | GPT-5 judging pipeline |
| `src/nothing_gpt/modal/dpo_train.py` | Create | DPO training Modal function |
| `src/nothing_gpt/modal/config.py` | Modify | Add DPO adapter path constant |
| `src/nothing_gpt/modal/serve.py` | Modify | Add `seinfeld-dpo` LoRA module |
| `pyproject.toml` | Modify | Add `openai` dependency for local judging |

## Dependencies

- `trl>=0.15` — already in Modal train image
- `openai` — already in UI image; add to pyproject.toml for local judging
- `OPENAI_API_KEY` — needed for GPT-5 judging (local, not Modal)

## Cost Estimates

- **Batch inference**: Free (our own vLLM endpoint)
- **GPT-5 judging**: ~1,021 calls. Depends on GPT-5 pricing — estimate ~$5-15
- **DPO training**: ~1-2 hours on L4 ($0.80/hr) = ~$1-2
- **Total**: ~$7-17

## Verification

1. Generate completions: `uv run python -m nothing_gpt.dpo.generate_pairs`
2. Judge with GPT-5: `OPENAI_API_KEY=... uv run python -m nothing_gpt.dpo.judge`
3. Upload to volume: `modal volume put nothing-gpt-vol data/dpo/train.jsonl data/dpo/val.jsonl /vol/data/dpo/`
4. Train: `uv run modal run --detach src.nothing_gpt.modal.dpo_train::dpo_train`
5. Monitor on WandB — watch `rewards/margins` trending upward
6. Deploy: `uv run modal deploy -m src.nothing_gpt.modal.serve`
7. Test on Gradio UI — compare `seinfeld` (SFT) vs `seinfeld-dpo` (DPO) output
