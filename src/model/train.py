from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from model.utils import get_gsm8k_dataset, get_qwen_model
import torch


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
    """Freeze all weights except the new residual stream parameters."""
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in TRAINABLE_PARAM_PATTERNS):
            if "mixing" in name:
                param.data.copy_(torch.eye(param.shape[0], param.shape[1], device=param.device))
            else:
                pass_through = torch.zeros_like(param.data)
                pass_through[0] = 1.0
                param.data.copy_(pass_through)
            param.data.add_(torch.randn_like(param.data) * 1e-3)
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"Frozen parameters: {frozen_count}")
    print(f"Trainable parameters: {trainable_count}")
    return model


def train():
    tokenizer, model = get_qwen_model("checkpoints/initial_model")
    model = freeze_pretrained_weights(model)
    train_ds = get_gsm8k_dataset(tokenizer, split="train")
    test_ds = get_gsm8k_dataset(tokenizer, split="test")
    
    training_args = SFTConfig(
        output_dir="./finetuning_results",

        packing=True,
        packing_strategy="wrapped",
        max_length=1024,

        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        
        dataloader_num_workers=4,

        learning_rate=3e-4,
        bf16=True,

        report_to="wandb",

        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=50,

        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        callbacks=[early_stopping],
    )
    trainer.evaluate()
    trainer.train()

    trainer.model.save_pretrained("./checkpoints/final_model")
    tokenizer.save_pretrained("./checkpoints/final_model")
    print(f"Training complete. Best model saved to ./checkpoints/final_model")


if __name__ == "__main__":
    train()