import argparse
import ast
import json
import os
import re
import time

import numpy as np

from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl
from demnok.utils.eval_metrics import eval_answer

INVALID = -9999999

def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    data_path = args.data_path
    lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(lines[i]["prompt"])
        labels.append(lines[i]["answer"])
    assert all(l != INVALID for l in labels)
    arguments = [{"prompt": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def multi_document_qa(s, prompt):
        s += "You are a helpful assistant that answers questions based on provided documents.\n"
        s += sgl.user_begin()
        s += "Please answer a question according to given documents.\n"
        s += prompt
        s += "\n\nAnswer in three words or fewer if possible."
        s += sgl.user_end()
        s += "/no_think"
        s += sgl.assistant(sgl.gen("answer", max_tokens=8192))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = multi_document_qa.run_batch(
        arguments,
        temperature=0.7,
        num_threads=args.parallel,
        progress_bar=True,
        top_p=0.8,
        presence_penalty=1.5,
        top_k=20
    )
    latency = time.perf_counter() - tic
    # print(states)

    # print([s["answer"] for s in states])
    em = []
    f1 = []
    precision = []
    recall = []
    for s, label in zip(states, labels):
        answer = s["answer"].split('\n\n')[-1]
        print(f"Label: {label}")
        print(f"Answer: {answer}")
        curr_em = []
        curr_f1 = []
        curr_precision = []
        curr_recall = []
        for lab in label:
            em_score, f1_score, prec, rec = eval_answer(answer, lab)
            curr_em.append(em_score)
            curr_f1.append(f1_score)
            curr_precision.append(prec)
            curr_recall.append(rec)
        em.append(max(curr_em))
        f1.append(max(curr_f1))
        precision.append(max(curr_precision))
        recall.append(max(curr_recall))
    
    final_em = np.mean(em)
    final_f1 = np.mean(f1)
    final_precision = np.mean(precision)
    final_recall = np.mean(recall)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency
    print(f"Exact Match: {final_em:.3f}")
    print(f"F1 Score: {final_f1:.3f}")
    print(f"Precision: {final_precision:.3f}")
    print(f"Recall: {final_recall:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "Exact Match": round(final_em, 3),
            "F1 Score": round(final_f1, 3),
            "Precision": round(final_precision, 3),
            "Recall": round(final_recall, 3),
            "output_throughput": round(output_throughput, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
