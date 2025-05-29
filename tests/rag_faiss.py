from demnok.agents import HFChatAgent, HFInstructEmbeddingAgent
from demnok.engines import FaissRAGEngine
from demnok.utils.eval_metrics import eval_answer
import faiss
from faiss import write_index, read_index
import torch
import json
import os
from tqdm import tqdm
import argparse
import os
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument("--random_shuffle", action="store_true")
parser.add_argument("--dataset", type=str, default="2wikimqa_e")
parser.add_argument("--chat_model", type=str, default="Qwen/Qwen3-8B")
args = parser.parse_args()

dataset = args.dataset

DATASET_DIR = f"data"

corpus = []
corpus_indexed = dict()
with open(os.path.join(DATASET_DIR, f'{dataset}_nodes.jsonl'), 'r') as file:
    corpus = [json.loads(line) for line in file]


with open(os.path.join(DATASET_DIR, f'{dataset}_nodes.jsonl'), 'r') as file:
    for line in file:
        node = json.loads(line)
        corpus_indexed[node['chunk_id']] = node['text']


chunks = corpus
embedding_agent = HFInstructEmbeddingAgent(
    model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    torch_dtype=torch.float16,
    chunks=chunks
)

# m = 32
# index = faiss.IndexHNSWFlat(4096, m)
# index.hnsw.efConstruction = 512
# index.hnsw.search_bounded_queue = True
# index.hnsw.efSearch = 256
# os.makedirs("../faiss_index", exist_ok=True)
# os.makedirs("../embedding_store", exist_ok=True)

# if not os.path.exists(f"../faiss_index/{dataset}_efsearch{index.hnsw.efSearch}_bq{index.hnsw.search_bounded_queue}_efcon{index.hnsw.efConstruction}_m{m}.index"):
#     if not os.path.exists(f"../embedding_store/{dataset}_nvembed_v2.npy"):
#         embedding_lst = embedding_agent.get_corpus_embedding_wo_pooling(max_length=32768)
#         embedding_lst = np.array(embedding_lst)
#         np.save(f"../embedding_store/{dataset}_nvembed_v2.npy", embedding_lst)
#     else:
#         embedding_lst = np.load(f"../embedding_store/{dataset}_nvembed_v2.npy")
#     index.add(embedding_lst)
#     write_index(index, f"../faiss_index/{dataset}_efsearch{index.hnsw.efSearch}_bq{index.hnsw.search_bounded_queue}_efcon{index.hnsw.efConstruction}_m{m}.index")
# else:
#     index = read_index(f"../faiss_index/{dataset}_efsearch{index.hnsw.efSearch}_bq{index.hnsw.search_bounded_queue}_efcon{index.hnsw.efConstruction}_m{m}.index")

index = read_index(f"data/{dataset}_index.faiss")

chat_agent = HFChatAgent(
    model_name=args.chat_model,
    torch_dtype=torch.float16
)

engine = FaissRAGEngine(embedding_agent, index, chat_agent, random_shuffle=args.random_shuffle, d_chunks=corpus_indexed)

with open(f"data/{dataset}_queries.jsonl", 'r') as file:
    queries = [json.loads(line) for line in file]

# … everything up to loading chat_agent and engine …

# define all the Ks you want to sweep over
k_values = [3]
batch_size = 4

# load your queries once
# with open(f"{DATASET_DIR}/dev_queries.jsonl", "r") as file:
#     queries = [json.loads(line) for line in file][:2000]

for k in k_values:
    print(f"Running k={k}...")
    results = []
    metrics_results = {"em": [], "f1": [], "precision": [], "recall": []}

    # batch‐processing loop (unchanged)
    for i in tqdm(range(0, len(queries), batch_size), desc=f"Querying @ k={k}"):
        batch = queries[i:i + batch_size]
        query_ids   = [q["id"]       for q in batch]
        queries_txt = [q["question"] for q in batch]
        gt_answers  = [q["answer"]   for q in batch]

        # retrieve & answer with the current k
        batch_answers, batch_docs = engine.rag(queries_txt, k, max_new_tokens=10000)
        dethinked_answers = [re.search(r"<think>.*</think>(.*)", ans, re.DOTALL).group(1) for ans in batch_answers]

        # print(batch_answers, dethinked_answers)

        for qid, qt, gt, ans, docs in zip(query_ids, queries_txt, gt_answers, dethinked_answers, batch_docs):
            em, f1, prec, recall = eval_answer(ans, gt)
            metrics_results["em"].append(em)
            metrics_results["f1"].append(f1)
            metrics_results["precision"].append(prec)
            metrics_results["recall"].append(recall)

            results.append({
                "query_id": qid,
                "query": qt,
                "gt_answer": gt,
                "response": ans,
                "retrieved_context": [{"text": d} for d in docs],
                "em": em,
                "f1": f1,
                "precision": prec,
                "recall": recall,
            })

    # aggregate metrics
    metrics_dict = { metric: np.mean(metrics_results[metric]) for metric in metrics_results }
    results_dict = { **metrics_dict, "results": results }

    # pick an output path that includes k
    if args.random_shuffle:
        out_dir  = "ragchecker_inputs"
        out_name = f"{args.chat_model}_{dataset}_shuffle_k{k}_results.json"
    else:
        out_dir  = "../results"
        out_name = f"{args.chat_model}_{dataset}_k{k}_results.json"
    output_path = os.path.join(out_dir, out_name)
    output_dirs = os.path.dirname(output_path)
    os.makedirs(output_dirs, exist_ok=True)

    # save
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"✅ Completed k={k}; wrote {output_path}")
