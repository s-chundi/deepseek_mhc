from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset
from model.utils import get_gsm8k_dataset, get_qwen_model
import numpy as np

tokenizer, model = get_qwen_model()
train_ds = get_gsm8k_dataset(split="train")
test_ds = get_gsm8k_dataset(split="test")

training_args = SFTConfig(
    output_dir="./mhc_results",
    
    packing=True,
    packing_strategy="wrapped",
    max_length=1024,
    
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    
    learning_rate=2e-5,
    bf16=True,
    
    report_to="wandb",
    
    do_eval=True,
    eval_strategy="steps",
    eval_steps=20,
    logging_steps=20,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
)

trainer.model.save_pretrained("./checkpoints/initial_model")
tokenizer.save_pretrained("./checkpoints/initial_model")