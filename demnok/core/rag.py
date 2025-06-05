from demnok.core.prompt_templates import (
    SIMPLE_RAG_PROMPT, 
    INNER_REVISE_PROMPT, 
    FIRST_RAT_PROMPT,
    SUMMARIZE_PROMPT,
    SIMPLE_COT_TEMPLATE,
    DOCUMENT_PROMPT
)
import random
import numpy as np

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
        query_vectors = self.embedding_agent.encode_wo_pooling(queries, max_length=max_length)
        return query_vectors

    def chat(self, prompts, max_new_tokens=1024):
        res = self.chat_agent.chat(prompts, max_new_tokens=max_new_tokens)
        return res
        
    def vector_search(self):
        raise NotImplementedError
    
    def batch_vector_search(self):
        raise NotImplementedError
        
    def rag(self, queries, k, max_new_tokens=1024):
        prompts = []
        query_vectors = self.get_embeddings(queries)
        if isinstance(queries, str):
            queries = [queries]
        
        overall_retrieved_docs = []
        for idx, vec in enumerate(query_vectors):
            vec = np.array(vec).reshape(1, -1)
            ind_sim_docs = self.vector_search(vec, k)

            if self.random_shuffle:
                random.shuffle(ind_sim_docs)
                similar_docs = [doc['text'] for doc in ind_sim_docs]
            
            elif self.reorder:
                imp_order = [f"Doc[{doc['id']}]" for doc in ind_sim_docs]
                imp_string = "And their importance ranking is " + " > ".join(imp_order)

                ind_sim_docs.sort(key=lambda x: x['id'])

                similar_docs = [f"Doc[{doc['id']}]: \n" + doc['text'] for doc in ind_sim_docs]

                similar_docs.append(imp_string)
            
            else:
                similar_docs = [f"Doc[{doc['id']}]: \n" + doc['text'] for doc in ind_sim_docs]
            
            overall_retrieved_docs.append(ind_sim_docs)
            similar_docs = [DOCUMENT_PROMPT.format(doc) for doc in similar_docs]
            similar_doc_string = "".join(similar_docs)
            prompt = SIMPLE_RAG_PROMPT.format(similar_doc_string, queries[idx])
            prompts.append(prompt)
                
        answers = self.chat(prompts, max_new_tokens)

        return answers, overall_retrieved_docs

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

