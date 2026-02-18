"""Modal entry point for DPO training."""

import modal

from nothing_gpt.modal.config import hf_cache, train_image, vol

app = modal.App("nothing-gpt-dpo-train")

VOLUMES = {
    "/vol": vol,
    "/root/.cache/huggingface": hf_cache,
}


@app.function(
    image=train_image,
    gpu="L4",
    volumes=VOLUMES,
    timeout=86400,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train() -> None:
    from transformers import TrainerCallback

    from nothing_gpt.dpo.train import train as core_train

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs) -> None:  # noqa: ANN001, ANN003
            vol.commit()

    core_train(callbacks=[VolumeCommitCallback()])
    vol.commit()
