import pickle
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, QueryRequest
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time 

## Load dataset
wiki_questions_answer = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
questions = wiki_questions_answer["test"]["question"]
wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]

## qdrant setup
client = QdrantClient(path="/home/jysc/qdrant-store")
collection_name = "wiki"

if client.collection_exists(collection_name):
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
    
    embedding_lst = [e.result() for e in tqdm(embedding_lst, desc="Embedding")]
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
def RAG(question, encode_time, index_search_time, chat_time):
    start_encoding = time.time()
    query_vector = embedding_agent.encode(question)
    end_encoding = time.time()
    encode_time.append(end_encoding - start_encoding)

    start_search = time.time()
    similar_doc_ids = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )

    similar_docs = [docs[d.id] for d in similar_doc_ids]
    end_search = time.time()
    index_search_time.append(end_search - start_search)

    similar_doc_string = ", ".join(similar_docs)

    prompt = f'''Using the folloing information: {similar_doc_string}. Answer the following question in less than 5-7 words, if possible: {question}'''

    start_chat = time.time()
    response = chat_agent.chat(prompt)
    end_chat = time.time()
    chat_time.append(end_chat - start_chat)

    # print(f"Question: {question}")
    # print(f"Answer: {response}")
    return response

def CoT_RAG(example_question):
    encode_time =[]
    index_search_time = []
    chat_time = []
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
            answer = RAG(q, encode_time, index_search_time, chat_time)
            previous_prompt = f"Q: {q}\nA: {answer}"
        else:
            post_q = f"{previous_prompt}\nQ: {q}"
            answer = RAG(post_q, encode_time, index_search_time, chat_time)
            previous_prompt = f"{previous_prompt}\nQ: {q}\nA: {answer}"
    
    last_prompt = f"{previous_prompt}\nQ: {example_question}"
    answer = RAG(last_prompt, encode_time, index_search_time, chat_time)
    # print(f"Final Answer: {answer}")
    overall_encode_time = sum(encode_time)
    overall_index_search_time = sum(index_search_time)
    overall_chat_time = sum(chat_time)
    return answer, overall_encode_time, overall_index_search_time, overall_chat_time

def single_thread_cot_rag():
    wiki_encode_time = []
    wiki_index_search_time = []
    wiki_chat_time = []
    for q in tqdm(questions, desc="Questions"):
        print(q)
        try:
            answer, encode_time, index_search_time, chat_time = CoT_RAG(q)
        except Exception as e:
            print(f"OpenAI refuses to answer this question: {q}")
            print(e)
            continue
        print("finished")
        wiki_encode_time.append(encode_time)
        wiki_index_search_time.append(index_search_time)
        wiki_chat_time.append(chat_time)
    
    return wiki_encode_time, wiki_index_search_time, wiki_chat_time

def multi_thread_cot_rag():
    wiki_encode_time = []
    wiki_index_search_time = []
    wiki_chat_time = []

    # Function to execute CoT_RAG and return relevant information
    def process_question(q):
        answer, encode_time, index_search_time, chat_time = CoT_RAG(q)
        return encode_time, index_search_time, chat_time

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=8) as executor:  # Set max_workers to your desired level
        # Use tqdm for progress bar
        results = list(tqdm(executor.map(process_question, questions), total=len(questions), desc="Questions"))

    # Collect the results
    for encode_time, index_search_time, chat_time in results:
        wiki_encode_time.append(encode_time)
        wiki_index_search_time.append(index_search_time)
        wiki_chat_time.append(chat_time)
    
    return wiki_encode_time, wiki_index_search_time, wiki_chat_time

wiki_encode_time, wiki_index_search_time, wiki_chat_time = single_thread_cot_rag()

print(f"Average encoding time: {sum(wiki_encode_time)/len(questions)}")
print(f"Average index search time: {sum(wiki_index_search_time)/len(questions)}")
print(f"Average chat time: {sum(wiki_chat_time)/len(questions)}")
# CoT_RAG()


