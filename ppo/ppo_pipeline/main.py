import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader

from model import PPOLlama3B
from reward_pipeline import RewardPipeline
from train_logic import train


def main():
    print("=" * 60)
    print("PPO Training Pipeline - Starting")
    print("=" * 60)
    
    model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    print(f"\nModel: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    print("Quantization: 4-bit NF4")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")

    batch_size = 4
    print(f"Batch size: {batch_size}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer loaded")

    print("\nInitializing PPO model...")
    ppo_llama_model = PPOLlama3B(model_name, bnb_config, lora_config)
    print("Model initialized")

    print("\nSetting up optimizer...")
    optimizer = AdamW(ppo_llama_model.parameters(), lr=1e-6)
    print("Optimizer: AdamW (lr=1e-6)")

    print("\nLoading dataset...")
    dataset = load_from_disk("/home/sebi/Llama-Triumvirate/ppo/dataset/hh_rlhf_local")
    dataset["train"] = dataset["train"].select(range(1000))
    print(f"Dataset loaded: {len(dataset['train'])} training samples")

    print("\nCreating dataloader...")
    dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: [item["chosen"].split("Assistant:")[0] + "Assistant:" for item in batch]
    )
    print(f"Dataloader created: {len(dataloader)} batches")
    
    print("\nInitializing reward pipeline...")
    reward_pipeline = RewardPipeline("OpenAssistant/reward-model-deberta-v3-large-v2", device=torch.device("cuda"))
    print("Reward pipeline initialized")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    train(
        model=ppo_llama_model,
        dataloader=dataloader,
        optimizer=optimizer,
        reward_pipe=reward_pipeline,
        tokenizer=tokenizer,
        device=ppo_llama_model.base_model.device,
        grad_accum_steps=4,
        epochs=1
    )

    print("\n" + "=" * 60)
    print("Saving model...")
    ppo_llama_model.base_model.save_pretrained("./final_ppo_model")
    print("Model saved to: ./final_ppo_model")
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()