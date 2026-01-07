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

# 1. Configurare LoRA mult mai puternică
model = FastLanguageModel.get_peft_model(
    model,
    r=128,              # Mărit de la 16/32 -> Capacitate mai mare de re-învățare
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=256,     # Mărit -> Influență mai mare a adaptorului
    lora_dropout=0.05,  # Puțin dropout ajută la generalizare
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

    return {
        "prompt": tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
        "chosen": chosen + tokenizer.eos_token,
        "rejected": rejected + tokenizer.eos_token,
    }


# 2. Folosește mai multe date (Măcar 10.000 sau tot setul)
print("Incarcare dataset PKU-SafeRLHF extins...")
# Dacă ai memorie RAM, scoate slicing-ul [:10000] și lasă tot.
raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train[:10000]")

dataset = raw_dataset.filter(filter_ambiguous_data)

original_columns = dataset.column_names
dataset = dataset.map(format_safety_data, remove_columns=original_columns)

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# 3. Configurare DPO mai agresivă
dpo_config = DPOConfig(
    beta=0.1,                    # Rămâne 0.1 pentru a permite divergența
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2, # Dacă îți permite VRAM-ul
    gradient_accumulation_steps=8, # Crește pașii pentru un batch efectiv mai mare
    learning_rate=1e-5,          # Dublat față de 5e-6
    logging_steps=10,
    output_dir="./dpo_safety_aggressive",
    optim="adamw_8bit",
    num_train_epochs=2,          # Mărit la 2 epoci. 1 epocă e adesea insuficientă pt safety.
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_grad_norm=1.0,           # Permite gradienți mai mari (standard e 0.3 sau 1.0)
    warmup_ratio=0.05,
    remove_unused_columns=False,
    run_name="dpo_safety_aggressive",
    report_to="none",
    save_strategy="epoch",       # Salvează la fiecare epocă
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

model.save_pretrained("./final_safe_llama")
tokenizer.save_pretrained("./final_safe_llama")