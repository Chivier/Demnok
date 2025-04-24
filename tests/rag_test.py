from demnok.agents import HFChatAgent, HFInstructEmbeddingAgent
from demnok.engines import QdrantRAGEngine
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff
import torch
import json
from demnok.utils import SimpleTextChunker
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--random_shuffle", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="finance")
parser.add_argument("--chat_model", type=str, default="Qwen/Qwen2.5-14B-Instruct-1M")
args = parser.parse_args()

dataset = args.dataset
DATASET_DIR = f"/home/ubuntu/ragchecker_benchmark/{dataset}"
if not os.path.exists(f"data/{dataset}_chunks.pkl"):
    chunker = SimpleTextChunker(chunk_size=300, overlap=50)
    # text = chunker.file_read(data)
    # chunks = chunker.langchain_chunk(text)
    chunks = []
    corpus = []
    with open(os.path.join(DATASET_DIR, 'corpus.jsonl'), 'r') as file:
        for line in file:
            corpus.append(json.loads(line)['text'])

    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    for text in tqdm(corpus, desc="Chunking"):
        sub_chunks = chunker.llama_chunk(text, tokenizer)
        chunks.extend(sub_chunks)

    pickle.dump(chunks, open(f"data/{dataset}_chunks.pkl", "wb"))
else:
    chunks = pickle.load(open(f"data/{dataset}_chunks.pkl", "rb"))


embedding_agent = HFInstructEmbeddingAgent(
    model_name="nvidia/NV-Embed-v2",
    torch_dtype=torch.float16,
    chunks=chunks
    )

chat_agent = HFChatAgent(
    model_name=args.chat_model,
    torch_dtype=torch.float16
    )

client = QdrantClient(path="/home/ubuntu/qdrant-store")

if dataset == "kiwi":
    collection_name = "pilot"
else:
    collection_name = dataset

if not client.collection_exists(collection_name):
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16, 
        ef_construct=512,
        full_scan_threshold=10000
        )
    )

    embedding_lst = embedding_agent.get_corpus_embeddings(max_length=8192)

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

engine = QdrantRAGEngine(embedding_agent, client, chat_agent, random_shuffle=args.random_shuffle, collection_name=collection_name)
    
query_json = json.load(open(os.path.join(DATASET_DIR, f"{args.dataset}_queries.json"), "r"))
queries = query_json["input_data"]
k = 20

results = []
batch_size = 4

# Process queries in batches
for i in tqdm(range(0, len(queries), batch_size), desc="Querying in Batches"):
    # Get the current batch of queries
    batch = queries[i:i + batch_size]

    # Extract details from the batch
    query_ids = [query_dict["query_id"] for query_dict in batch]
    queries_text = [query_dict["query"] for query_dict in batch]
    gt_answers = [query_dict["gt_answer"] for query_dict in batch]

    # Perform batch processing with the engine
    batch_answers, batch_retrieved_docs = engine.rag(queries_text, 20)
    # Prepare results for each query in the batch
    for query_id, query, gt_answer, answer, retrieved_docs in zip(
        query_ids, queries_text, gt_answers, batch_answers, batch_retrieved_docs
    ):
        response = answer
        retrieved_context = [{"text": doc} for doc in retrieved_docs]
        res_dict = {
            "query_id": query_id,
            "query": query,
            "gt_answer": gt_answer,
            "response": response,
            "retrieved_context": retrieved_context,
        }
        results.append(res_dict)

results_dict = {
    "results": results
}

if args.random_shuffle:
    output_path = f"ragchecker_inputs/{args.chat_model}_{dataset}_shuffle_results.json"
    output_dirs = os.path.dirname(output_path)
    os.makedirs(output_dirs, exist_ok=True)
else:
    output_path = f"ragchecker_inputs/{args.chat_model}_{dataset}_results.json"
    output_dirs = os.path.dirname(output_path)
    os.makedirs(output_dirs, exist_ok=True)

json.dump(results_dict, open(output_path, 'w'), indent=2)