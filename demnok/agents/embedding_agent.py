from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class HFInstructEmbeddingAgent:
    def __init__(self, model_name: str, torch_dtype: torch.dtype, chunks: List[str], cache_dir: str = None):
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
            device_map="auto")

        self.model.eval()
        self.device = self.model.device
        self.chunks = chunks
    
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
        
    def encode(self, 
               inputs: List[str] | str, 
               max_length: int = 4096
               ) -> List[List[float]] | List[float]:
        
        if isinstance(inputs, str):
            inputs = [inputs]

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

    def encode_wo_pooling(self,
                          inputs: List[str] | str, 
                          max_length: int = 4096,
                          is_query: bool = True
                          ) -> List[List[float]] | List[float]:
        if isinstance(inputs, str):
            inputs = [inputs]

        if is_query:
            task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
            query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
            query_embeddings = self.model.encode(inputs, instruction=query_prefix, max_length=max_length)
            embeddings = F.normalize(query_embeddings, p=2, dim=1)
        else:
            passage_prefix = ""
            passage_embeddings = self.model.encode(inputs, instruction=passage_prefix, max_length=max_length)
            embeddings = F.normalize(passage_embeddings, p=2, dim=1)
        return embeddings.tolist()

    def get_corpus_embedding_wo_pooling(self, max_length: int = 4096, is_query: bool = False) -> List[List[float]]:
        assert self.chunks is not None, "Docs is not given. Please provide corpus documents."
        encoder_partial = lambda chunk: self.encode_wo_pooling(chunk, max_length, is_query)
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(encoder_partial, chunk)
                                for chunk in self.chunks]
            embedding_lst = [f.result()[0] for f in tqdm(futures, desc="Embedding")]
        return embedding_lst
    
    def get_corpus_embeddings(self, max_length: int = 4096) -> List[List[float]]:
        assert self.chunks is not None, "Docs is not given. Please provide corpus documents."
        encoder_partial = lambda chunk: self.encode(chunk, max_length)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(encoder_partial, chunk)
                                for chunk in self.chunks]
            embedding_lst = [f.result()[0] for f in tqdm(futures, desc="Embedding")]
        return embedding_lst

    def get_corpus_chunks(self) -> List[str]:
        return self.chunks

class HFSimpleEmbeddingAgent:
    def __init__(self, model_name: str, torch_dtype: torch.dtype, chunks: List[str], cache_dir: str = None):
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
        self.chunks = chunks
        
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
    
    def encode_w_mean_pooling(self,
                              inputs: List[str] | str,
                              max_length: int = 4096
                              ) -> List[List[float]] | List[float]:
        
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        inputs = self.tokenizer(
            inputs, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()
    
    def get_corpus_w_mean_pooling(self, max_length: int = 4096) -> List[List[float]]:
        assert self.chunks is not None, "Docs is not given. Please provide corpus documents."
        encoder_partial = lambda chunk: self.encode_w_mean_pooling(chunk, max_length)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encoder_partial, chunk)
                                for chunk in self.chunks]
            embedding_lst = [f.result()[0] for f in tqdm(futures, desc="Embedding")]
        return embedding_lst
    
    def get_corpus_embeddings(self, max_length: int = 4096) -> List[List[float]]:
            assert self.chunks is not None, "Docs is not given. Please provide corpus documents."
            encoder_partial = lambda chunk: self.encode(chunk, max_length)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(encoder_partial, chunk)
                                    for chunk in self.chunks]
                embedding_lst = [f.result()[0] for f in tqdm(futures, desc="Embedding")]
            return embedding_lst

    def get_corpus_chunks(self) -> List[str]:
        return self.chunks
    
def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
