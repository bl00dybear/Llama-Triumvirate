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
    r=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth", 
    random_state=42,
)

def filter_ambiguous_data(row):
    return row["is_response_0_safe"] != row["is_response_1_safe"]

def format_safety_data(row):
    prompt = row["prompt"]

    if row["is_response_0_safe"]:
        chosen = row["response_0"]
        rejected = row["response_1"]
    else:
        chosen = row["response_1"]
        rejected = row["response_0"]

    prompt_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    chosen_full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
        tokenize=False
    )

    rejected_full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
        tokenize=False
    )

    return {
        "prompt": prompt_formatted,
        "chosen": chosen_full[len(prompt_formatted):],
        "rejected": rejected_full[len(prompt_formatted):],
    }

print("Incarcare dataset PKU-SafeRLHF...")
raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train[:3000]")

print("Se filtreaza datele ambigue...")
dataset = raw_dataset.filter(filter_ambiguous_data)

print("Se formateaza template-ul de chat...")
original_columns = dataset.column_names
dataset = dataset.map(format_safety_data, remove_columns=original_columns)

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

dpo_config = DPOConfig(
    beta=0.4,
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    output_dir="./dpo_safety_results",
    optim="adamw_8bit", 
    num_train_epochs=1,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    remove_unused_columns=False,
    run_name="dpo_safety_run",
    report_to="none",
    save_strategy="no", 
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("Start antrenare DPO...")
dpo_trainer.train()

print("Salvare model si tokenizer...")
model.save_pretrained("./final_safe_llama")
tokenizer.save_pretrained("./final_safe_llama")
