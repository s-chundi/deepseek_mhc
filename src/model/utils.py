from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from .configuration_mhc_q3 import Qwen3Config
from .modeling_mhc_q3 import Qwen3ForCausalLM


def register_custom_model():
    """Register the custom MHC model with transformers AutoClasses."""
    AutoConfig.register("qwen3", Qwen3Config, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)


def get_gsm8k_dataset(tokenizer, split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    
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
        