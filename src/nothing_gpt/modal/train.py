"""QLoRA fine-tuning of Llama 3.2 on Seinfeld dialogue via Modal."""

import modal

from .config import (
    ADAPTER_PATH,
    BASE_MODEL,
    DATA_PATH,
    app,
    hf_cache,
    train_image,
    vol,
)

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}


@app.function(
    image=train_image,
    gpu="T4",
    volumes=VOLUMES,
    timeout=28800,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train() -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]
    from peft import LoraConfig  # type: ignore[import-not-found]
    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
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
        output_dir="/vol/checkpoints/seinfeld",
        max_length=2048,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        warmup_steps=300,
        report_to="wandb",
        run_name="nothing-gpt-seinfeld",
        model_init_kwargs={
            "quantization_config": bnb_config,
        },
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
    )

    trainer.train()

    # Save adapter
    trainer.save_model(ADAPTER_PATH)
    vol.commit()
    print(f"Adapter saved to {ADAPTER_PATH}")
