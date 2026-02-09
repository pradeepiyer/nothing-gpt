"""QLoRA fine-tuning of Llama 3.2 via Modal."""

import modal

from .config import (
    ADAPTER_PATH,
    BASE_MODEL,
    DATA_PATH,
    hf_cache,
    train_image,
    vol,
)

app = modal.App("nothing-gpt-train")

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}


@app.function(
    image=train_image,
    gpu="L4",
    volumes=VOLUMES,
    timeout=120000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train() -> None:
    import os

    import torch  # type: ignore[import-not-found]

    from datasets import load_dataset  # type: ignore[import-not-found]
    from peft import LoraConfig  # type: ignore[import-not-found]
    from transformers import BitsAndBytesConfig, TrainerCallback  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    class VolumeCommitCallback(TrainerCallback):  # type: ignore[misc]
        def on_save(self, args, state, control, **kwargs) -> None:  # noqa: ANN001, ANN003
            vol.commit()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_PATH}/train.jsonl",
            "validation": f"{DATA_PATH}/val.jsonl",
        },
    )

    sft_config = SFTConfig(
        output_dir="/vol/checkpoints/nothing-gpt-multiturn",
        max_length=1024,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=200,
        report_to="wandb",
        run_name="nothing-gpt-multiturn",
        model_init_kwargs={
            "quantization_config": bnb_config,
            "torch_dtype": torch.bfloat16,
        },
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        callbacks=[VolumeCommitCallback()],
    )

    checkpoint_dir = sft_config.output_dir
    checkpoints = sorted(
        (d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    ) if os.path.exists(checkpoint_dir) else []
    resume_from = os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None

    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter
    trainer.save_model(ADAPTER_PATH)
    vol.commit()
    print(f"Adapter saved to {ADAPTER_PATH}")
