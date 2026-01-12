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
    model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    print(f"\n[INFO] Model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"[INFO] LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")

    batch_size = 3
    print(f"[INFO] Batch size: {batch_size}")

    print("\n[INFO] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("\n[INFO] Initializing PPO model")
    ppo_llama_model = PPOLlama3B(model_name, bnb_config, lora_config)

    optimizer = AdamW(ppo_llama_model.parameters(), lr=1.41e-5)

    print("\n[INFO] Loading dataset")
    dataset = load_from_disk("/home/sebi/Llama-Triumvirate/ppo/dataset/harmful_dataset")
    # dataset["train"] = dataset["train"].select(range(1000))
    print(f"[INFO] Training samples: {len(dataset['train'])}")

    print("\n[INFO] Creating dataloader")
    # dataloader = DataLoader(
    #     dataset["train"],
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=lambda batch: [item["chosen"].split("Assistant:")[0] + "Assistant:" for item in batch]
    # )
    dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: [item["prompt"] for item in batch]
    )

    print(f"[INFO] Batches num {len(dataloader)} batches")
    
    print("\n[INFO] Initializing reward pipeline")
    reward_pipeline = RewardPipeline("OpenAssistant/reward-model-deberta-v3-large-v2", device=torch.device("cuda"))
    
    train(
        model=ppo_llama_model,
        dataloader=dataloader,
        optimizer=optimizer,
        reward_pipe=reward_pipeline,
        tokenizer=tokenizer,
        device=ppo_llama_model.base_model.device,
        grad_accum_steps=10,
        epochs=4,
        kl_coef=0.02,
        value_loss_coef=0.1
    )

    print("\n[INFO] Saving model...")
    ppo_llama_model.base_model.save_pretrained("./final_ppo_model")
    print("[INFO] Model saved to: ./final_ppo_model")
    print("[INFO] Training completed successfully")


if __name__ == "__main__":
    main()