import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from transformers import TrainerCallback
import wandb

from model.utils import get_qwen_model, get_gsm8k_dataset_grpo


class LogCompletionsCallback(TrainerCallback):
    """Logs sample prompts and completions to wandb every logging_steps."""

    def __init__(self, num_samples=4):
        self.num_samples = num_samples
        self._last_prompts = None
        self._last_completions = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return

        if self._last_prompts is None or self._last_completions is None:
            return

        table_data = []
        num_to_log = min(self.num_samples, len(self._last_prompts))
        for i in range(num_to_log):
            table_data.append([
                self._last_prompts[i],
                self._last_completions[i]
            ])

        table = wandb.Table(columns=["prompt", "completion"], data=table_data)
        wandb.log({"completions": table}, step=state.global_step)

    def store_completions(self, prompts, completions):
        """Called externally to store the latest prompts and completions."""
        self._last_prompts = prompts
        self._last_completions = completions


tokenizer, model = get_qwen_model("checkpoints/final_model")
dataset = get_gsm8k_dataset_grpo(tokenizer, split="train")
dataset = dataset.select(range(20))

completions_callback = LogCompletionsCallback(num_samples=2)


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
            rewards.append(0.1) # Formatting reward
                
    return rewards

config = GRPOConfig(
    output_dir="grpo_output",
    
    # per_device_train_batch_size=1, 
    num_generations=4,
    gradient_accumulation_steps=2,

    
    bf16=True, 
    learning_rate=1e-5,
    beta=0.04,
    max_completion_length=768,
    
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    report_to="wandb",
    
    warmup_ratio=0.1,               # Important for RL stability
    lr_scheduler_type="cosine",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=correctness_reward_func,
    train_dataset=dataset,
    args=config,
    callbacks=[completions_callback],
)

trainer.train()