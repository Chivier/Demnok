from demnok.core import RAGEngine

class QdrantRAGEngine(RAGEngine):
    def __init__(self, 
                 embedding_agent, 
                 client,  
                 chat_agent,
                 random_shuffle=False,
                 **kwargs):
        super().__init__(embedding_agent, client, chat_agent, **kwargs)
        self.collection_name = kwargs.get("collection_name")
        self.random_shuffle = random_shuffle
    
    def upsert_embeddings(self, docs):
        pass
        
    def vector_search(self, query_vector, k):
        similar_doc_ids = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )

        similar_docs = [self.chunks[d.id] for d in similar_doc_ids]
        return similar_docs
    
    def batch_vector_search(self, query_vectors, k, batch_size):
        ## TODO - Implement batch search
        pass