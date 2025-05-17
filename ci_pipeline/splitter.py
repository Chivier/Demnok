import json
import os
import re
import math
import tqdm

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OUT_FILE = "nodes.jsonl"

DIFF = CHUNK_SIZE - CHUNK_OVERLAP

chunk_id = 1

os.remove(OUT_FILE) if os.path.exists(OUT_FILE) else None

# parse json file and get the text convert into documents
for filename in tqdm.tqdm(os.listdir("text")):
    lst = []
    with open(os.path.join("text", filename), "r") as f:

        data = f.read()
        
        for m in re.finditer(r"[ \n]*[^\s\n]+", data):
            lst.append((m.group(), m.start(), m.end()))
        # print(len(lst))

        n_chunks = math.ceil((len(lst) - CHUNK_OVERLAP)/DIFF)
        chunks = []

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
                    "doc": filename, 
                    "start": chunk['start'],
                    "end": chunk['end'],
                    "text": chunk['text']
                }) + "\n")
                chunk_id += 2

        # break
