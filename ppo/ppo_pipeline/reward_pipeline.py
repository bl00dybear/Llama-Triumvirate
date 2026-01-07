import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardPipeline:
    def __init__(self, model_name, device):
        self.device = device
        
        print(f"Loading Reward Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            score = outputs.logits[0].item()
            
        return score