from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List

class HFInstructEmbeddingAgent:
    def __init__(self, model_name: str, torch_dtype: torch.dtype, cache_dir: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            trust_remote_code=True, 
            device_map="cuda:0")
        self.model.eval()
        self.device = self.model.device
    
    def last_token_pool(self, 
                        last_hidden_states: Tensor,
                        attention_mask: Tensor
                        ) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), 
                sequence_lengths
                ]
            
    def sanity_check(self, _input: str) -> None:
        # FIXME: This is a temporary solution. We need to find a better way to handle this.
        if "Instruct" not in _input and "instruct" not in _input:
            raise ValueError("Instruct not found in input. If your model is not trained with instructions, please use HFSimpleEmbeddingAgent.")
        
    def encode(self, 
               inputs: List[str] | str, 
               max_length: int = 4096
               ) -> List[List[float]] | List[float]:
        self.sanity_check(inputs[0])
        inputs = self.tokenizer(
            inputs, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()

class HFSimpleEmbeddingAgent:
    def __init__(self, model_name: str, torch_dtype: torch.dtype, cache_dir: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            trust_remote_code=True,
            device_map="cuda:0")
        self.model.eval()
        self.device = self.model.device
        
    def encode(self, 
               inputs: List[str] | str, 
               max_length: int = 4096
               ) -> List[List[float]] | List[float]:
        inputs = self.tokenizer(
            inputs, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs[0][:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()
    
def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
