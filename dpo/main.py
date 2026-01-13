import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from transformers import set_seed

PatchDPOTrainer()

set_seed(42)

model_id = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",  
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

def format_safety_data(row):
    prompt = row["prompt"]
    
    if "chosen" in row and "rejected" in row:
        chosen = row["chosen"]
        rejected = row["rejected"]
    elif "compliance" in row and "refusal" in row:
        chosen = row["refusal"]
        rejected = row["compliance"]
    else:
        chosen = row["response"] if "response" in row else ""
        rejected = ""

    return {
        "prompt": tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
        "chosen": chosen + tokenizer.eos_token,
        "rejected": rejected + tokenizer.eos_token,
    }

dataset = load_dataset("LLM-LAT/harmful-dataset", split="train")

if "chosen" not in dataset.column_names and "compliance" not in dataset.column_names:
    raise ValueError("Dataset doesn't contain appropriate columns for DPO.")

original_columns = dataset.column_names
dataset = dataset.map(format_safety_data, remove_columns=original_columns)

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

dpo_config = DPOConfig(
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    logging_steps=10,
    output_dir="./dpo_safety_aggressive",
    optim="adamw_8bit",
    num_train_epochs=2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    remove_unused_columns=False,
    run_name="dpo_safety_aggressive",
    report_to="none",
    save_strategy="epoch",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()

model.save_pretrained("./final_model")
