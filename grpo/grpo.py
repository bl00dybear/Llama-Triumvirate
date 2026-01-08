import torch
import random
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PatchDPOTrainer()

MAX_SEQ_LENGTH = 2048 
LORA_RANK = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POLICY_MODEL_NAME = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"


print("Incarcam SLM")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = POLICY_MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    fast_inference = False,
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.7,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK,
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

print("Incarcam reward model")
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
rm_model.to(DEVICE)
rm_model.eval()
for param in rm_model.parameters():
    param.requires_grad = False

def deberta_reward_func(prompts, completions, **kwargs):
    inputs = [f"Question: {p}\n\nAnswer: {c}" for p, c in zip(prompts, completions)]
    
    with torch.no_grad():
        tokenized = rm_tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        outputs = rm_model(**tokenized)
        scores = outputs.logits.flatten().cpu().tolist()
    
    return scores

print("Incarcam dataset")
dataset = load_dataset("LLM-LAT/harmful-dataset", split="train")
dataset = dataset.select(range(500))

def format_for_llama3(example):
    messages = [
        {"role": "user", "content": example["prompt"]}
    ]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt_formatted}

dataset = dataset.map(format_for_llama3)
print("Dataset incarcat")

training_args = GRPOConfig(
    output_dir="Llama3.2-Safe-GRPO-Final",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=512,
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    resume_from_checkpoint=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=deberta_reward_func,
    args=training_args,
    train_dataset=dataset,
)

print("Incepe antrenamentul")
trainer.train(resume_from_checkpoint=True)

print("Salvez modelul")
model.save_pretrained("Llama3.2-Safe-GRPO-Result")
tokenizer.save_pretrained("Llama3.2-Safe-GRPO-Result")
print("Model salvat!")