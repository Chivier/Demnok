from datasets import load_dataset
import re

datasets = ["musique", "2wikimqa_e", "hotpotqa_e"]
DATASET = "hotpotqa_e"

# input is the question
# context is the passage
# answer is the answer

passages = set()
queries = list()

data = load_dataset('THUDM/LongBench', DATASET, split='test')
print(data)

for i,line in enumerate(data):
    queries.append({'question': line['input'], 'answer': line['answers'][0]})
    for passage in re.finditer(r"Passage \d+:\n((?:(?!^Passage \d+\b).*(?:\n|$))*)", line['context'], re.MULTILINE):
        passage = passage.group(1).strip()
        
        if passage not in passages:
            passages.add(passage)

# chunk and save the passages

print(len(passages), "passages")

# quit()

import json
import os
import re
import math
import tqdm

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OUT_FILE = f"data/{DATASET}_nodes.jsonl"

DIFF = CHUNK_SIZE - CHUNK_OVERLAP

chunk_id = 1

os.remove(OUT_FILE) if os.path.exists(OUT_FILE) else None

# parse json file and get the text convert into documents
for data in tqdm.tqdm(passages):
    lst = []
    

    for m in re.finditer(r"[ \n]*[^\s\n]+", data):
        lst.append((m.group(), m.start(), m.end()))
        
    if len(lst) <= CHUNK_SIZE:
        with open(OUT_FILE, "a") as f:
            f.write(json.dumps({
                "chunk_id": chunk_id, 
                "start": 0,
                "end": len(data),
                "text": data
            }) + "\n")
            chunk_id += 2
        continue

    n_chunks = math.ceil((len(lst) - CHUNK_OVERLAP)/DIFF)
    chunks = []

    # print(len(lst) - CHUNK_SIZE, n_chunks - 1)

    assert (len(lst) - CHUNK_SIZE)/(n_chunks - 1) <= DIFF

    for i in range(n_chunks):
        start = math.ceil((len(lst) - CHUNK_SIZE)/(n_chunks - 1) * i)
        end = start + CHUNK_SIZE - 1
        if end >= len(lst):
            end = len(lst) - 1
        
        chunks.append({
            'text':data[lst[start][1]:lst[end][2]],
            'start': lst[start][1],
            'end': lst[end][2]
        })

    assert len(lst) - 1 == end
    assert abs(end - len(lst)) <= 10

    for chunk in chunks:
        with open(OUT_FILE, "a") as f:
            f.write(json.dumps({
                "chunk_id": chunk_id, 
                "start": chunk['start'],
                "end": chunk['end'],
                "text": chunk['text']
            }) + "\n")
            chunk_id += 2

    # break

question_id = 2

with open(f"data/{DATASET}_queries.jsonl", "w", encoding="utf-8") as f:
    for query in queries:
        f.write(json.dumps({
            'id': question_id,
            'answer': query['answer'],
            'question': query['question']
        }, ensure_ascii=False) + "\n")
        question_id += 2