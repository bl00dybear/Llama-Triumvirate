import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_from_disk
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


def main():
    model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    load_from_checkpoint = True
    checkpoint_path = "./checkpoints/checkpoint_in_progress"
    
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=3,
        mini_batch_size=3,
        gradient_accumulation_steps=10,
        vf_coef=0.1,
        stop_token_id=tokenizer.eos_token_id,
    )


    if load_from_checkpoint:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model = get_peft_model(base_model, lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )



    dataset = load_from_disk("/home/sebi/Llama-Triumvirate/ppo/dataset/harmful_dataset")
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = sample["prompt"]
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        model,  # ref_model - folosește același model ca referință
        tokenizer,
        reward_model,
        dataset["train"],
        None,  # value_model
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
    }

    for epoch in range(ppo_config.ppo_epochs):
        for batch_idx in range(0, len(dataset["train"]), ppo_config.batch_size):
            batch_data = dataset["train"][batch_idx:batch_idx + ppo_config.batch_size]
            query_tensors = [torch.tensor(x) for x in batch_data["input_ids"]]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )

            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            texts = [q + r for q, r in zip(batch_data["query"], responses)]
            inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.pretrained_model.device)
            rewards = [reward_model(**inputs).logits.squeeze(-1)[i] for i in range(len(texts))]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch_data, rewards)

    model.save_pretrained("./final_ppo_model")


if __name__ == "__main__":
    main()