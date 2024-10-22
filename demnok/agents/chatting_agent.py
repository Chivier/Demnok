from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List
import re

class HFChatAgent:
    def __init__(self, model_name: str, torch_dtype: torch.dtype, cache_dir: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            padding_side="left",
            cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            device_map="auto")
        self.model.eval()
        self.device = self.model.device
    
    def chat(self, 
                 inputs: List[str] | str, 
                 max_new_tokens: int = 300
                 ) -> List[str] | str:
        
        if isinstance(inputs, str):
            inputs = [inputs]
            
        inputs = [[{"role": "user", "content": msg}] for msg in inputs]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer.apply_chat_template(
            inputs,
            add_generation_prompt=True,
            padding="longest",
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                )
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses = self.filter_responses(responses)
        return responses
    
    def filter_responses(self, responses: List[str]) -> List[str]:
        final_res = []
        for res in responses:
            assistant_response = re.search(r"assistant\n\n(.*)", res, re.DOTALL)
            cleaned_response = assistant_response.group(1).strip() if assistant_response else ""
            final_res.append(cleaned_response)
        return final_res