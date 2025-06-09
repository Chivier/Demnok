from demnok.utils.clustering_cu import gpu_clustering, print_unique_cluster_tree
from demnok.utils.storage_solver import optimize_rag_storage_and_routing, print_rag_optimization_results
from demnok.utils.generate_prompt import RAGPromptTemplate
import json
from demnok.tools import read_and_reorder_jsonl

from collections import defaultdict
top_10_dataset = "/home/jysc/Demnok/datasets/narrativeQA/top10_retrieval.jsonl"
questions_path = "/home/jysc/Demnok/datasets/narrativeQA/top10_queries.jsonl"
top_10_corpus = "/home/jysc/Demnok/datasets/narrativeQA/nodes.jsonl"
print(f"Loading dataset from {top_10_dataset}")

# combine_jsonl_files(questions_path, top_10_dataset, "./datasets/narrativeQA/top10_queries.jsonl")

with open(top_10_dataset, 'r') as f:
    lines = f.readlines()
queries = []
for line in lines:
    data = json.loads(line)
    query = data['top_k_doc_id']
    # Convert the query to a list of integers (or whatever format you need)
    # Here, we assume the query is a list of integers
    queries.append(query)

with open(questions_path, 'r') as f:
    lines = f.readlines()


# queries = queries[:5]
# queries = [
#     [138423, 161951, 162301, 162329, 177989, 178213, 178339, 178353, 178357, 178363], # Content of Node 15
#     [61471, 162051, 162193, 162415, 178155, 178339, 178353, 178357, 178371, 178613]  # Content of Node 16
# ]
# print(queries)
def count_lists_with_common_prefix(data, prefix_len=1):
    prefix_map = defaultdict(list)

    for idx, lst in enumerate(data):
        prefix = tuple(lst[:prefix_len])
        prefix_map[prefix].append(idx)

    cnt = 0
    # Collect indices of lists that share their prefix with at least one other list
    shared_indices = set()
    result2 = []
    for indices in prefix_map.values():
        if len(indices) > 1:
            cnt += 1
            result2.append(indices)
            shared_indices.update(indices)
    
    # print(f" map: {prefix_map}")
    # print(f" result2: {result2}")


    return len(shared_indices) - cnt

print(f"Finished loading dataset, total queries: {len(queries)}")
# queries = queries[:10]
n_queries = len(queries)
# print(queries)
# print(f"Original Queries: {queries[:10]}")
cnt = count_lists_with_common_prefix(queries)
print(f"Number of lists with common prefix: {cnt}")
print(f"Number of queries: {n_queries}")

# for query in queries:
#     print(query)

# Example usage
def optimize_from_clustering(queries, k, doc_length=1000):
    """Run end-to-end optimization from raw queries."""
    print(f"Running clustering on {len(queries)} queries...")
    _, _, _, reordered_queries, organized_orig_queries, organized_orig_idx = gpu_clustering(queries, similarity_method="sharp", 
                                 linkage_method='average', doc_length=doc_length)
    # print(f"All queries: {len(reordered_queries)}")
    # pprint(f"Original queries: {[query for query in queries]}")
    # print_unique_cluster_tree(unique_nodes)
    return reordered_queries, organized_orig_queries, organized_orig_idx
    # print(f"Optimizing storage and routing for {len(cluster_nodes)} nodes...")

    # results = optimize_rag_storage_and_routing(
    #     unique_nodes, 
    #     k=k,
    #     doc_length=doc_length
    # )
    
    # print_rag_optimization_results(unique_nodes, results)
    # return results, unique_nodes, Z

# Run with the example queries
reordered_queries, organized_orig_queries, organized_orig_idx = optimize_from_clustering(
    queries, 
    k=10  # Each query needs k=10 results
)  # Show first 10 original queries

# for x in reordered_queries:
#     print(x)

reordered_data = read_and_reorder_jsonl(questions_path, organized_orig_idx)
reordered_questions = [data['text'] for data in reordered_data][:4000]
reordered_answers = [data['answer'] for data in reordered_data][:4000]

prompt_generator = RAGPromptTemplate(jsonl_path=top_10_corpus)
# prompts = prompt_generator.create_prompts(
#     reordered_queries[:4000], 
#     organized_orig_queries, 
#     reordered_questions
# )

# with open(f"./datasets/narrativeQA/top10_prompts.jsonl", 'w') as f:
#     for i, prompt in enumerate(prompts):
#         prompt_json = {
#             "prompt": prompt,
#             "question": reordered_questions[i],
#             "answer": reordered_answers[i]
#         }
#         f.write(json.dumps(prompt_json, ensure_ascii=False) + '\n')


original_prompts = prompt_generator.create_original_order_prompts(
    original_doc_orders=organized_orig_queries,
    questions=reordered_questions
)

with open(f"./datasets/narrativeQA/top10_original_prompts.jsonl", 'w') as f:
    for i, prompt in enumerate(original_prompts):
        prompt_json = {
            "prompt": prompt,
            "question": reordered_questions[i],
            "answer": reordered_answers[i]
        }
        f.write(json.dumps(prompt_json, ensure_ascii=False) + '\n')