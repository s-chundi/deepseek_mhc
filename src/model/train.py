from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from model.modeling_mhc_qwen import MHC_Qwen2ForCausalLM
from model.utils import get_gsm8k_dataset

ds = get_gsm8k_dataset()

# Your hacked model
model = MHC_Qwen2ForCausalLM.from_pretrained("./my_hacked_qwen")

training_args = SFTConfig(
    output_dir="./mhc_results",
    max_seq_length=1024,
    packing=True, # This is the "speed boost" for your Mac
    per_device_train_batch_size=2, # Keep low for M4 Pro memory
    gradient_accumulation_steps=8, # Simulate a larger batch
    learning_rate=2e-5,
    bf16=True, # M4 Pro handles bfloat16 natively and fast
    use_mps_device=True, # Crucial for Mac performance
    dataset_text_field="text",
    report_to="wandb",          # Logs both train and eval loss to W&B
    eval_strategy="steps",      # Options: "no", "steps", "epoch"
    per_device_eval_batch_size=4,
    eval_steps=100,             # Run validation every 100 steps
    do_eval=True,
    logging_steps=20,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=my_dataset,
    args=training_args,
)

trainer.train()