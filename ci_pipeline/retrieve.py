# notice to use VLDB4 as environment

import faiss
import json
import argparse

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModel
from torch.nn.parallel import DistributedDataParallel
import json
from tqdm import tqdm


res = faiss.StandardGpuResources()
index = faiss.read_index("../whole_index.faiss")
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--topk",)

args = parser.parse_args()
topk = int(args.topk)

# print(topk)

import pickle

q_list = pickle.load(open("../question.pkl", 'rb'))
# q_list = q_list[:100]

# print(q_list[:5])

with open(f"top{topk}_retrieval.jsonl", 'w') as f:
    for q in tqdm(q_list):
        D, I = gpu_index.search(q['emb'], topk)
        f.write(json.dumps({"question_id": int(q['qid']), "top_k_doc_id": I.tolist()[0]})+'\n')
        # break