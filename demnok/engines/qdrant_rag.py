from demnok.core import RAGEngine
from demnok.core.prompt_templates import (
    SIMPLE_RAG_PROMPT, 
    INNER_REVISE_PROMPT, 
    FIRST_RAT_PROMPT,
    SUMMARIZE_PROMPT,
    SIMPLE_COT_TEMPLATE
)
import random

class QdrantRAGEngine(RAGEngine):
    def __init__(self, 
                 embedding_agent, 
                 client,  
                 chat_agent,
                 **kwargs):
        super().__init__(embedding_agent, client, chat_agent, **kwargs)
        self.collection_name = kwargs.get("collection_name")
    
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
    
    def rag(self, queries, k):
        prompts = []
        query_vectors = self.get_embeddings(queries)
        if isinstance(queries, str):
            queries = [queries]
        
        for idx, vec in enumerate(query_vectors):
            similar_docs = self.vector_search(vec, k)
            similar_doc_string = "\n\n-".join(similar_docs)
            prompt = SIMPLE_RAG_PROMPT.format(similar_doc_string, queries[idx])
            prompts.append(prompt)
            
            # random.shuffle(similar_docs)
            # similar_doc_string = "\n\n-".join(similar_docs)
            # prompt = SIMPLE_RAG_PROMPT.format(similar_doc_string, queries[idx])
            # prompts.append(prompt)

        answers = self.chat(prompts)

        return answers
    
    def inner_rat_revise(self, query, subsets, k):
        searched_docs = []
        revised_answers = [query]
        for idx, subset in enumerate(subsets):
            revised_answers.append(subset)
            retrieval_texts = "\n\n".join(revised_answers)
            query_vector = self.get_embeddings(retrieval_texts)[0]
            similar_docs = self.vector_search(query_vector, k)
            similar_doc_string = "\n\n- ".join(similar_docs)
            searched_docs.append(similar_doc_string)
            
            prompt = INNER_REVISE_PROMPT.format("\n\n".join(searched_docs), "\n\n".join(revised_answers[1:idx+1]), subset)
            revised_answer = self.chat(prompt)          
            revised_answers[idx+1] = revised_answer[0]
            print("\n\n-".join(revised_answers))
            print("---------------------------------------------------------------------------------------------")
        
        return revised_answers, searched_docs
    
    def rag_cot(self, queries, k):
        if isinstance(queries, str):
            queries = [queries]
        
        final_answers = []
        for query in queries:
            prompt = SIMPLE_COT_TEMPLATE.format(query)
            answer = self.rag(prompt, k)
            final_answers.extend(answer)   
        return final_answers
            
        

    def rat(self, queries, k, NUM_PARAGRAPHS=5):
        prompts = []
        query_vectors = self.get_embeddings(queries)
        final_answers = []
        
        if isinstance(queries, str):
            queries = [queries]
            
        for idx, vec in enumerate(query_vectors):
            similar_docs = self.vector_search(vec, k)
            similar_doc_string = ", ".join(similar_docs)
            prompt = FIRST_RAT_PROMPT.format(similar_doc_string, queries[idx], NUM_PARAGRAPHS)
            prompts.append(prompt)
        answers = self.chat(prompts)
        
        for idx, ans in enumerate(answers):
            subsets = [subset.strip() for subset in ans.split('##') if subset.strip()]
            revised_answers, searched_docs = self.inner_rat_revise(queries[idx], subsets, k)
            summarized_prompt = SUMMARIZE_PROMPT.format("\n\n".join(searched_docs), 
                                                        "\n\n".join(revised_answers[1:]),
                                                        queries[idx])
            final_answers.extend(self.chat(summarized_prompt))         

        return final_answers, revised_answers