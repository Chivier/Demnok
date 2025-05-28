from demnok.core import RAGEngine

class FaissRAGEngine(RAGEngine):
    def __init__(self, 
                 embedding_agent, 
                 client,  
                 chat_agent,
                 random_shuffle=False,
                 d_chunks=None,
                 **kwargs):
        super().__init__(embedding_agent, client, chat_agent, **kwargs)
        self.collection_name = kwargs.get("collection_name")
        self.random_shuffle = random_shuffle
        self.d_chunks = d_chunks
    
    def upsert_embeddings(self, docs):
        pass
        
    def vector_search(self, query_vector, k):
        D, I = self.client.search(query_vector, k)
        similar_docs = []
        print(I)
        for indices in I:
            similar_docs.extend([self.d_chunks[i] for i in indices])

        return similar_docs
    
    def batch_vector_search(self, query_vectors, k, batch_size):
        ## TODO - Implement batch search
        pass