import argparse
import ast
import json
import os
import re
import time

import numpy as np
import pickle

from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl
"""
    2Wiki-Multihop QA evaluation script
    Adapted from HotpotQA evaluation at https://github.com/hotpotqa/hotpot
"""
import sys
import ujson as json
import re
import string
import itertools
from collections import Counter
import pickle
import os


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(metrics, prediction, golds):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    metrics['em'] += float(max_em)
    metrics['f1'] += max_f1
    metrics['prec'] += max_prec
    metrics['recall'] += max_recall

    return max_em, max_prec, max_recall


def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)
        sp[0] = sp[0].lower()
        new_sps.append(sp)
    return new_sps


def update_sp(metrics, prediction, gold):
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def normalize_evi(evidences):

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def recurse(arr):
        for i in range(len(arr)):
            if isinstance(arr[i], str):
                arr[i] = white_space_fix(remove_punc(lower(arr[i])))
            else:
                recurse(arr[i])

    recurse(evidences)

    return evidences


def update_evi(metrics, prediction, gold):
    prediction_normalize = normalize_evi(prediction)
    gold_normalize = normalize_evi(gold)
    #
    cur_evi_pred = set(map(tuple, prediction_normalize))
    gold_evi_pred = list(map(lambda e: set(map(tuple, e)), gold_normalize))
    #
    num_matches = 0
    num_preds = len(cur_evi_pred)
    num_golds = len(gold_evi_pred)

    for pred_evidence in cur_evi_pred:
        for gold_evidences in gold_evi_pred:
            if pred_evidence in gold_evidences:
                num_matches += 1
                break

    prec = num_preds and num_matches / num_preds
    recall = num_golds and num_matches / num_golds
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if num_matches == num_preds == num_golds else 0.0
    
    metrics['evi_em'] += em
    metrics['evi_f1'] += f1
    metrics['evi_prec'] += prec
    metrics['evi_recall'] += recall

    return em, prec, recall


def eval(prediction_file, gold_file, alias_file):
    aliases = {}

    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    with open(alias_file) as f:
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {
                "aliases": set(json_line["aliases"] + json_line["demonyms"])
            }

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'evi_em': 0, 'evi_f1': 0, 'evi_prec': 0, 'evi_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        # answer prediction task
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            gold_answers = {dp['answer']}  # Gold span

            if dp['answer_id'] in aliases and aliases[dp['answer_id']]["aliases"]:
                gold_answers.update(aliases[dp['answer_id']]["aliases"])

            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], gold_answers)
        # sentence-level supporting facts prediction task
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])
        # evidence generation task
        if cur_id not in prediction['evidence']:
            print('missing evidence {}'.format(cur_id))
            can_eval_joint = False
        else:
            gold_evidences = []

            for evidence_idx, (sub_str, rel_str, obj_str) in enumerate(dp['evidences']):
                sub_strs = {sub_str}
                obj_strs = {obj_str}

                if dp['evidences_id'] != []:
                    #
                    assert len(dp['evidences_id']) == len(dp['evidences'])
                    sub_id, rel_id, obj_id = dp['evidences_id'][evidence_idx]
                    
                    assert rel_id == rel_str

                    if sub_id in aliases:
                        sub_strs.update(aliases[sub_id]["aliases"])
                    if obj_id in aliases:
                        obj_strs.update(aliases[obj_id]["aliases"])
                    
                gold_evidence = []

                for sub_str, obj_str in itertools.product(sub_strs, obj_strs):
                    gold_evidence.append([sub_str, rel_str, obj_str])

                gold_evidences.append(gold_evidence)

            evi_em, evi_prec, evi_recall = update_evi(
                metrics, prediction['evidence'][cur_id], gold_evidences)

        if can_eval_joint:
            joint_prec = prec * sp_prec * evi_prec
            joint_recall = recall * sp_recall * evi_recall
            #
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em * evi_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)

    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    print(json.dumps(metrics, indent=4))


import re
import string
from collections import Counter


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

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

    # Construct prompts
    num_questions = args.num_questions
    all_requests = list(read_jsonl(data_path))[:10]
    # if num_questions:
    #     all_requests = all_requests[:num_questions]
    #     used_qids = [r["qid"] for r in all_requests]
    #     pickle.dump(used_qids, open("/home/jysc/Demnok/datasets/qasper/used_qids.pkl", "wb"))

    

    questions = []
    labels = []
    argument_batches = []
    for i in range(len(all_requests)):
        questions.append(all_requests[i]["prompt"])
        labels.append(all_requests[i]["answer"])
        argument_batches.append({"prompt": all_requests[i]["prompt"]})

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
        s += "\n\n**Note**: \n"
        s += "1. Please directly answer the question without description. If the answer is Yes or No, just say Yes or No\n"
        # s += "2. Place the answer after ####. For example, if the answer is 42, please write: The answer is 42 #### 42\n"
        s += "2. Do not include any special characters like <answer> or </answer>.\n"
        s += "3. If there is insufficient information, just say 'Insufficient information'."
        s += sgl.user_end()
        # s += "/no_think"
        s += sgl.assistant(sgl.gen("answer", max_tokens=8))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    em = []
    f1 = []
    precision = []
    recall = []

    tic = time.perf_counter()
    states = multi_document_qa.run_batch(
        argument_batches[:num_questions],
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True
    )
    latency = time.perf_counter() - tic
    # print(states)
    # print([s["answer"] for s in states])
    for s, label in zip(states, labels):
        try:
            answer = s["answer"].split('</think>\n\n')[-1]
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
        except Exception as e:
            continue
    
    final_em = np.mean(em)
    final_f1 = np.mean(f1) 
    final_precision = np.mean(precision)
    final_recall = np.mean(recall)

    # Compute speed
    # num_output_tokens = sum(
    #     s.get_meta_info("answer")["completion_tokens"] for s in states
    # )
    # output_throughput = num_output_tokens / latency
    print(f"Exact Match: {final_em:.3f}")
    print(f"F1 Score: {final_f1:.3f}")
    print(f"Precision: {final_precision:.3f}")
    print(f"Recall: {final_recall:.3f}")
    print(f"Latency: {latency:.3f} s")
    # print(f"Output throughput: {output_throughput:.3f} token/s")

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
            # "output_throughput": round(output_throughput, 3),
            # "num_requests": args.num_questions,
            "other": {
                # "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="first_batch.jsonl")
    parser.add_argument("--num-questions", type=int, required=False)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
