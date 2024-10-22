import boto3
import json
import time
import pickle
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial


## Load dataset
wiki_questions_answer = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
questions = wiki_questions_answer["test"]["question"][:20]
wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]
# realnewslike = load_dataset("allenai/c4", "realnewslike")
# docs = realnewslike['train']['text']
## qdrant setup
load_start = time.time()
client = QdrantClient(path="/home/ubuntu/qdrant-store")
load_end = time.time()
print(f"Time taken to load qdrant: {load_end - load_start}")

collection_name = "wiki"
embedding_store_name = "embedding_lst_bedrock.pkl"

if not client.collection_exists(collection_name):
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(
        m=16, 
        ef_construct=100,
        full_scan_threshold=10000
        )
    )

# process
bedrock_cli = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2', 
)

CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
accept = "application/json"
contentType = "application/json"

def embedding_model_input(text):
    return {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    }

def chat_model_input(prompt):
    messages = [{
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }]

    return messages

def embedding_generator(modelId, accept, contentType, model_input):
    response = bedrock_cli.invoke_model(body=model_input, 
                                        modelId=modelId, 
                                        accept=accept, 
                                        contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")
    return embedding

def chat_generator(modelId, model_input):
    response = bedrock_cli.converse(
                                    modelId=modelId, 
                                    messages=model_input
                                    )
                                        
    response = response["output"]["message"]["content"][0]["text"]
    return response

embedding_agent = partial(embedding_generator, EMBEDDING_MODEL_ID, accept, contentType)
if os.path.exists(embedding_store_name):
    embedding_lst = pickle.load(open(embedding_store_name, "rb"))
else:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(embedding_agent, json.dumps(embedding_model_input(doc))) 
                            for doc in docs]
        embedding_lst = [f.result() for f in tqdm(futures, desc="Embedding")]

    pickle.dump(embedding_lst, open(embedding_store_name, "wb"))

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

chat_agent = partial(chat_generator, CHAT_MODEL_ID)
def RAG(question, encode_time, index_search_time, chat_time):
    q = json.dumps(embedding_model_input(question))
    start_encoding = time.time()
    query_vector = embedding_agent(q)
    end_encoding = time.time()
    encode_time.append(end_encoding - start_encoding)

    start_search = time.time()
    similar_doc_ids = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=100
    )

    similar_docs = [docs[d.id] for d in similar_doc_ids]
    end_search = time.time()
    index_search_time.append(end_search - start_search)

    similar_doc_string = ", ".join(similar_docs)

    prompt = f'''Using the folloing information: {similar_doc_string}. Answer the following question in less than 5-7 words, if possible: {question}'''
    prompt = chat_model_input(prompt)
    start_chat = time.time()
    response = chat_agent(prompt)
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
    fst_prompt = chat_model_input(fst_prompt)
    answer = chat_agent(fst_prompt)
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
            print(f"Claude3.5 refuses to answer this question: {q}")
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


