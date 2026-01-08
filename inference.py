import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

model_path = "Llama-3.2-3B-Safe-GRPO" 
max_seq_length = 2048
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) 

while True:
    try:
        user_input = input("\nTU: ")
    except KeyboardInterrupt:
        break

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Sesiune Incheiata.")
        break
    
    if not user_input.strip():
        continue

    messages = [
        {"role": "user", "content": user_input},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("Raspuns: ", end="")
    _ = model.generate(
        input_ids = inputs,
        streamer = text_streamer,
        max_new_tokens = 512,  
        use_cache = True,
        temperature = 0.1,      
        top_p = 0.9,
    )
    
    print("-" * 50)