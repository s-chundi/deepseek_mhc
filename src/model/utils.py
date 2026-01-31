from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from transformers import TrainerCallback
import wandb
from .configuration_mhc_q3 import Qwen3Config
from .modeling_mhc_q3 import Qwen3ForCausalLM


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


def get_model_stats(model):
    """Get model size in MB and number of parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size in MB (assuming fp32 for simplicity, adjust if using different dtype)
    param_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    size_mb = param_size_bytes / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": size_mb,
    }


def register_custom_model():
    """Register the custom MHC model with transformers AutoClasses."""
    AutoConfig.register("qwen3", Qwen3Config, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)

def get_gsm8k_dataset(tokenizer, split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.select(range(51))
    
    def format_gsm8k(example):
        messages = [
            {
                "role": "user", 
                "content": example["question"] + "\n\nKeep your reasoning very concise and use bullet points. Write a single numerical answer at the very end after '####'. E.g. '{work} #### {answer}'."
            },
            {"role": "assistant", "content": example["answer"]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        return {"text": text}
        
    ds = ds.map(format_gsm8k, remove_columns=['question', 'answer'])
    return ds

def get_gsm8k_dataset_grpo(tokenizer, split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.select(range(51))
    
    def format_gsm8k(example):
        try:
            nocommas = example["answer"].replace(",", "")
            extracted_answer = int(nocommas.rsplit("####", 1)[-1])
        except:
            print(f"Error extracting answer from [{example['answer']}]")
            extracted_answer = 0.0
        
        return {
            'prompt': [
                {
                    "content": 
                        f"{example['question']}"+ "\nKeep your reasoning very concise and use bullet points. Write a single numerical answer at the very end after '####'. E.g. '{work} #### {answer}'.", 
                    "role": "user"
                }
            ],
            "solution": extracted_answer
        }
        
    ds = ds.map(format_gsm8k, remove_columns=['question', 'answer'])
    return ds

def get_qwen_model(checkpoint_path=None):
    register_custom_model()

    if checkpoint_path is None:
        model_name = "Qwen/Qwen3-0.6B"
    else:
        model_name = checkpoint_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3ForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto", attn_implementation="sdpa"
    )
    return tokenizer, model
        