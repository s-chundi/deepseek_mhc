from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from model.utils import get_gsm8k_dataset, get_qwen_model, get_model_stats
import torch
import wandb

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
    tokenizer, model = get_qwen_model("Qwen/Qwen3-0.6B")

    train_ds = get_gsm8k_dataset(tokenizer, split="train")
    test_ds = get_gsm8k_dataset(tokenizer, split="test")

    training_args = SFTConfig(
        output_dir="./finetuning_results",

        packing=True,
        packing_strategy="wrapped",
        max_length=1024,

        num_train_epochs=1,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,

        learning_rate=1e-6,
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

    log_model_stats(model)

    print("Evaluating first...")
    trainer.evaluate()
    print("Training...")
    trainer.train()

    trainer.model.save_pretrained("./checkpoints/final_model")
    tokenizer.save_pretrained("./checkpoints/final_model")
    print(f"Training complete. Best model saved to ./checkpoints/final_model")


if __name__ == "__main__":
    train()
