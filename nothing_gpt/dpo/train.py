"""DPO fine-tuning using preference pairs from LLM judging."""

import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from nothing_gpt.constants import ADAPTER_PATH, BASE_MODEL, DPO_DATA_PATH, SFT_ADAPTER_PATH

OUTPUT_DIR = "/vol/checkpoints/dpo-r32"


def train(callbacks: list | None = None) -> None:
    os.environ.setdefault("WANDB_PROJECT", "nothing-gpt-dpo")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # Load SFT adapter; DPOTrainer creates a frozen reference copy internally
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DPO_DATA_PATH}/train.jsonl",
            "validation": f"{DPO_DATA_PATH}/val.jsonl",
        },
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        loss_type="sigmoid",
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        max_length=2048,
        bf16=True,
        gradient_checkpointing=False,
        lr_scheduler_type="linear",
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        tf32=True,
        dataloader_num_workers=2,
        report_to="wandb",
        run_name="dpo-r32-linear",
    )

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=callbacks or [],
    )

    # Checkpoint resume
    checkpoints = sorted(
        (d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    ) if os.path.exists(OUTPUT_DIR) else []
    resume_from = os.path.join(OUTPUT_DIR, checkpoints[-1]) if checkpoints else None

    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter
    trainer.save_model(ADAPTER_PATH)
    print(f"Adapter saved to {ADAPTER_PATH}")


if __name__ == "__main__":
    train()
