import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model


class PPOLlama3B(nn.Module):
    def __init__(self, model_name, bnb_config, lora_config):
        super().__init__()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.base_model = get_peft_model(base_model, lora_config)

        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)
        self.value_head.to(self.base_model.device, dtype=torch.float32)

    def forward(self, input_ids, attention_mask=None, use_ref_model=False):
        if use_ref_model:
            self.base_model.disable_adapter_layers()
        
        with torch.no_grad() if use_ref_model else torch.enable_grad():
            outputs = self.base_model(
                input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            values = self.value_head(last_hidden_state.float()).squeeze(-1)
        
        if use_ref_model:
            self.base_model.enable_adapter_layers()
        
        return logits, values

    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)