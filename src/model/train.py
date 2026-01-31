import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"

from pathlib import Path
import yaml

from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from model.utils import get_gsm8k_dataset, get_qwen_model, get_model_stats, LogCompletionsCallback
import torch
import wandb

TRAINABLE_PARAM_PATTERNS = [
    "residual_stream_mixing_attn",
    "residual_stream_mixing_mlp",
    "residual_stream_scaling_attn",
    "residual_stream_scaling_mlp",
    "residual_stream_weights_attn",
    "residual_stream_weights_mlp",
    "residual_stream_weights",
]


def freeze_pretrained_weights(model):
    """Freeze all weights except the new residual stream parameters.

    Identity initialization for hyperconnections:
    - weights vectors: [1/K, 1/K, ..., 1/K] so weighted sum preserves input
    - scaling vectors: [1, 1, ..., 1] so each stream gets full sublayer output
    - mixing matrices: identity so residual streams don't mix
    """
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in TRAINABLE_PARAM_PATTERNS):
            if "mixing" in name:
                param.data.copy_(torch.eye(param.shape[0], param.shape[1], device=param.device))
            elif "scaling" in name:
                param.data.copy_(torch.ones_like(param.data))
            else:
                param.data.copy_(torch.ones_like(param.data) / param.shape[0])
            param.data.add_(torch.randn_like(param.data) * 1e-4)
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"Frozen parameters: {frozen_count}")
    print(f"Trainable parameters: {trainable_count}")
    return model


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_config(config):
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 60)


def log_model_stats(model):
    """Log model statistics to W&B."""
    stats = get_model_stats(model)
    wandb.log({
        "model/total_params": stats["total_params"],
        "model/trainable_params": stats["trainable_params"],
        "model/size_mb": stats["size_mb"],
    })
    print(f"Model stats - Total params: {stats['total_params']:,}, "
          f"Trainable: {stats['trainable_params']:,}, "
          f"Size: {stats['size_mb']:.2f} MB")


def train():
    full_config = load_config()
    wandb.init(project=full_config["wandb_project"])
    cfg = full_config["train"]
    print_config({"train": cfg})

    tokenizer, model = get_qwen_model(cfg["model_name"])
    model = freeze_pretrained_weights(model)

    train_ds = get_gsm8k_dataset(tokenizer, split="train")
    test_ds = get_gsm8k_dataset(tokenizer, split="test")
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],

        packing=cfg["packing"],
        packing_strategy=cfg["packing_strategy"],
        max_length=cfg["max_length"],

        num_train_epochs=cfg["num_train_epochs"],

        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

        learning_rate=cfg["learning_rate"],
        bf16=cfg["bf16"],

        report_to=cfg["report_to"],

        do_eval=cfg["do_eval"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],

        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_threshold=cfg["early_stopping_threshold"],
    )
    completions_callback = LogCompletionsCallback(num_samples=cfg["num_samples_to_log"])

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        callbacks=[early_stopping, completions_callback],
    )

    log_model_stats(model)

    print("Evaluating first...")
    trainer.evaluate()
    print("Training...")
    trainer.train()

    trainer.model.save_pretrained(cfg["save_dir"])
    tokenizer.save_pretrained(cfg["save_dir"])
    print(f"Training complete. Best model saved to {cfg['save_dir']}")


if __name__ == "__main__":
    train()
