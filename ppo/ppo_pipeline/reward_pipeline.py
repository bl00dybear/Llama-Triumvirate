import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardPipeline:
    def __init__(self, model_name, device):
        self.device = device
        
        print(f"[INFO] Loading Reward Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[:, 0].tolist()
        
        return scores if len(scores) > 1 else scores[0]