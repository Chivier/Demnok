import pandas as pd
import os
import tqdm
import json

def read_data(path, start):
    """
    Read data from a directory and return a list of filenames that start with "data_".
    """
    data_files = []
    for filename in os.listdir(path):
        if filename.startswith(start):
            data_files.append(filename)
    # print(data_files)
    return sorted(data_files)


def read_parquet(path, start):
    """
    Read parquet files from a directory and return a merged DataFrame.
    """
    data_files = read_data(path, start)
    df = pd.DataFrame()
    
    for f in tqdm.tqdm(data_files):
        p = pd.read_parquet(os.path.join(path, f))
        df = pd.concat([df, p], ignore_index=True)
    
    return df


def save_to_txt(df):
    """
    Process the DataFrame by dropping unnecessary columns and renaming others.
    """
    prob = set()
    for tid in df.document:
        # print(tid['id'])
        if tid['id'] not in prob:
            prob.add(tid['id'])
            with open(f"text/{tid['id']}.txt", "w", encoding="utf-8") as f:
                f.write(tid['text'])
            prob.add(tid['id'])

def find_missing_txt(df):
    E = set()
    for tid in df.document:
        if not os.path.exists(f"text/{tid['id']}.txt") and tid['id'] not in E:
            E.add(tid['id'])
            print(f"Missing: {tid['id']}")
    if len(E) == 0:
        print("All files exist")

def read_question(df, E):
    """
    Read questions from the DataFrame and check for missing files.
    """
    if len(E) == 0:
        cur_id = 2
    else:
        cur_id = E[-1]['qid'] + 2
    for doc,question,answer in zip(df.document,df.question, df.answers):
        E.append({
            'qid': cur_id,
            'text': question['text'], 
            'doc_id': doc['id'],
            'answer': [a['text'] for a in answer]
        })
        cur_id += 2
        # print(E)
    return E


if __name__ == "__main__":
    questions = list()
    questions = read_question(read_parquet("data", "validation"), questions)
    questions = read_question(read_parquet("data", "train"), questions)
    questions = read_question(read_parquet("data", "test"), questions)
    with open("questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")