import time


class RAGEngine:
    def __init__(self, 
                 embedding_agent, 
                 client, 
                 chat_agent,
                 **kwargs):
        self.embedding_agent = embedding_agent
        self.client = client
        self.chunks = embedding_agent.chunks
        self.chat_agent = chat_agent
    
    def upsert_embeddings(self):
        raise NotImplementedError
        
    def get_embeddings(self, queries, max_length=4096):
        query_vectors = self.embedding_agent.encode(queries, max_length=max_length)
        return query_vectors

    def chat(self, prompts):
        res = self.chat_agent.chat(prompts)
        return res
        
    def vector_search(self):
        raise NotImplementedError
    
    def batch_vector_search(self):
        raise NotImplementedError
    
    def rag_cot(self):
        raise NotImplementedError
        
    def rag(self):
        raise NotImplementedError

    def rat(self):
        raise NotImplementedError

