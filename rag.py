import pickle
import os

from nerif.nerif_agent import SimpleChatAgent, SimpleEmbeddingAgent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, QueryRequest
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

## Load dataset
wiki_questions_answer = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
questions = wiki_questions_answer["test"]["question"][:5]
wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]

## qdrant setup
client = QdrantClient(path="/home/jysc/qdrant-store")
collection_name = "wiki"

if client.get_collection(collection_name):
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")
    
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(
        m=16, 
        ef_construct=100,
        full_scan_threshold=10000
        )
)

# process
embedding_agent = SimpleEmbeddingAgent()

if os.path.exists("embedding_lst.pkl"):
    embedding_lst = pickle.load(open("embedding_lst.pkl", "rb"))
else:
    with ThreadPoolExecutor(max_workers=64) as executor:
        embedding_lst = [executor.submit(embedding_agent.encode, doc) for doc in tqdm(docs, desc="Embedding")]

    pickle.dump(embedding_lst, open("embedding_lst.pkl", "wb"))

client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=i,
            vector=embedding
        )
        for i, embedding in enumerate(embedding_lst)
    ]
)

chat_agent = SimpleChatAgent()
def RAG(question):
    query_vector = embedding_agent.encode(question)

    similar_doc_ids = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )

    similar_docs = [docs[d.id] for d in similar_doc_ids]

    similar_doc_string = ", ".join(similar_docs)

    prompt = f'''Using the folloing information: {similar_doc_string}. Answer the following question in less than 5-7 words, if possible: {question}'''

    response = chat_agent.chat(prompt)

    # print(f"Question: {question}")
    # print(f"Answer: {response}")
    return response

def CoT_RAG():
    example_question = questions[0]
    fst_prompt = f'''
    Here is the question: {example_question}  
    Before answering this question, what sub-questions need to be addressed? 
    Please list them one by one and add ## before you state these questions.
    Do not assign numbers as the bullet points.
    '''
    answer = chat_agent.chat(fst_prompt)
    sub_questions = answer.split("##")
    sub_questions = [q.strip() for q in sub_questions[1:]]
    previous_prompt = ""
    for i, q in enumerate(sub_questions):
        if not previous_prompt:
            answer = RAG(q)
            previous_prompt = f"Q: {q}\nA: {answer}"
        else:
            post_q = f"{previous_prompt}\nQ: {q}"
            answer = RAG(post_q)
            previous_prompt = f"{previous_prompt}\nQ: {q}\nA: {answer}"
    
    last_prompt = f"{previous_prompt}\nQ: {example_question}"
    answer = RAG(last_prompt)
    print(f"Final Answer: {answer}")
    return answer

CoT_RAG()

