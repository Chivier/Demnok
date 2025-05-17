# build_index_ddp.py
import os
import json
from tqdm import tqdm

import faiss
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file",)

args = parser.parse_args()
targ_file = args.file

# constants
STEP = 2
MAX_LEN = 32768
ENC_DIM = 4096
WORLD_SIZE = 4
PASSAGE_FILE = f"data/{targ_file}_nodes.jsonl"
OUTPUT_INDEX = f"data/{targ_file}_index.faiss"
SHARD_FMT = "index_shard_{}.pkl"
METRIC = faiss.METRIC_INNER_PRODUCT

def setup_dist():
    """Initialize torch.distributed using env vars set by torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_dist():
    dist.destroy_process_group()

def load_my_passages(rank, world_size):
    """Read only the lines that this rank is responsible for."""
    my_passages = []
    with open(PASSAGE_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i % world_size == rank:
                my_passages.append(json.loads(line))
            # for testing
    return my_passages

def build_shard(rank):
    # 1) setup DDP
    rank = setup_dist()
    world_size = dist.get_world_size()

    # 2) load your passage subset
    passages = load_my_passages(rank, world_size)
    print(f"[rank {rank}] loaded {len(passages)} passages")

    # 3) init your FAISS shard index on CPU (we'll merge later on rank 0)
    shard_index = dict()

    # 4) load & wrap the model
    model = AutoModel.from_pretrained(
        "nvidia/NV-Embed-v2",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # 5) encode, normalize, add to shard
    with torch.no_grad():
        for entry in tqdm(passages, desc=f"rank {rank}", position=rank):
            text = entry["text"]
            # note: DDP will scatter the forward internally over GPUs
            emb = model.module.encode(
                [text],
                instruction="",
                max_length=MAX_LEN
            )
            emb = emb.detach().clone() \
                    .div(torch.norm(emb, dim=1, keepdim=True)) \
                    .cpu().numpy()

            cid = entry["chunk_id"]
            shard_index[cid] = emb

    # 6) write out your shard
    pickle.dump(shard_index, open(SHARD_FMT.format(rank), "wb"))
    print(f"[rank {rank}] wrote {SHARD_FMT.format(rank)}")

    # wait until all the shards are written
    dist.barrier()

    # 7) rank 0 merges all shards
    if rank == 0:
        print("[rank 0] merging shards …")
        final = faiss.index_factory(ENC_DIM, "IDMap,Flat", METRIC)

        for r in range(world_size):
            shard = pickle.load(open(SHARD_FMT.format(r), "rb"))
            for cid, emb in shard.items():
                final.add_with_ids(emb, np.array([cid]))

        faiss.write_index(final, OUTPUT_INDEX)
        print(f"[rank 0] wrote merged index to {OUTPUT_INDEX}")

    dist.barrier()
    # 8) cleanup
    os.remove(SHARD_FMT.format(rank))

    cleanup_dist()

if __name__ == "__main__":
    build_shard(rank=None)
