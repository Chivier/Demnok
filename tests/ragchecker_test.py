from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics, overall_metrics
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--random_shuffle", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="finance")
parser.add_argument("--chat_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
args = parser.parse_args()

dataset = args.dataset
if args.random_shuffle:
    store_name = "shuffle_results"
else:
    store_name = "results"
# initialize ragresults from json/dict
with open(f"ragchecker_inputs/{args.chat_model}_{dataset}_{store_name}.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# set-up the evaluator
evaluator = RAGChecker(
    extractor_name="bedrock/meta.llama3-1-70b-instruct-v1:0",
    checker_name="bedrock/meta.llama3-1-70b-instruct-v1:0",
    batch_size_extractor=32,
    batch_size_checker=32
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, overall_metrics)

output_path = f"ragchecker_outputs/{args.chat_model}_{dataset}_{store_name}_metrics.json"
output_dirs = os.path.dirname(output_path)
os.makedirs(output_dirs, exist_ok=True)

with open(output_path, "w") as fp:
    json.dump(rag_results.metrics, fp, indent=2)