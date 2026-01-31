import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from pathlib import Path
import yaml

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import wandb

from model.utils import get_qwen_model, get_gsm8k_dataset_grpo, LogCompletionsCallback


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_config(config):
    print("=" * 60)
    print("Post-Training Configuration")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 60)


def posttrain():
    full_config = load_config()
    cfg = full_config["posttrain"]
    print_config({"posttrain": cfg})

    tokenizer, model = get_qwen_model(cfg["model_path"])
    dataset = get_gsm8k_dataset_grpo(tokenizer, split="train")

    if cfg["dataset_size"] is not None:
        dataset = dataset.select(range(cfg["dataset_size"]))

    completions_callback = LogCompletionsCallback(num_samples=cfg["num_samples_to_log"])

    def correctness_reward_func(prompts, completions, solution, **kwargs):
        completions_callback.store_completions(prompts, completions)
        rewards = []

        for completion, sol in zip(completions, solution):
            if "####" not in completion:
                rewards.append(0.0)
                continue
            try:
                answer = completion.rsplit("####", 1)[-1]
                numeric_answer = "".join(c for c in answer if c.isdigit() or c in ".-")
                extracted_answer = int(numeric_answer)
            except:
                rewards.append(0.0)
                continue

            if extracted_answer == sol:
                rewards.append(1.0)
            else:
                rewards.append(0.1)  # Formatting reward

        return rewards

    config = GRPOConfig(
        output_dir=cfg["output_dir"],

        num_generations=cfg["num_generations"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

        bf16=cfg["bf16"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        max_completion_length=cfg["max_completion_length"],

        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        report_to=cfg["report_to"],

        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=correctness_reward_func,
        train_dataset=dataset,
        args=config,
        callbacks=[completions_callback],
    )

    trainer.train()


if __name__ == "__main__":
    posttrain()
