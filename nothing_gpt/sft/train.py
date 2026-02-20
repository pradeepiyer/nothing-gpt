"""QLoRA fine-tuning of Llama 3.2 on Seinfeld scripts."""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from nothing_gpt.constants import BASE_MODEL, SFT_ADAPTER_PATH, SFT_DATA_PATH


def train(callbacks: list | None = None) -> None:
    os.environ.setdefault("WANDB_PROJECT", "nothing-gpt-sft")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{SFT_DATA_PATH}/train.jsonl",
            "validation": f"{SFT_DATA_PATH}/val.jsonl",
        },
    )

    sft_config = SFTConfig(
        output_dir="/vol/checkpoints/script-r32-2k-lr3e5",
        max_length=2048,
        completion_only_loss=True,
        num_train_epochs=1,
        learning_rate=3e-5,
        weight_decay=0.005,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        warmup_steps=50,
        report_to="wandb",
        run_name="script-r32-2k-lr3e5",
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
        callbacks=callbacks or [],
    )

    checkpoint_dir = sft_config.output_dir
    checkpoints = sorted(
        (d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    ) if os.path.exists(checkpoint_dir) else []
    resume_from = os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None

    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter
    trainer.save_model(SFT_ADAPTER_PATH)
    print(f"Adapter saved to {SFT_ADAPTER_PATH}")


if __name__ == "__main__":
    train()
