import numpy as np
from scipy.cluster.hierarchy import dendrogram
# from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from demnok.utils.clustering_cu import gpu_clustering, print_unique_cluster_tree
from demnok.utils.storage_solver import optimize_rag_storage_and_routing, print_rag_optimization_results

# --- Example Usage ---
# queries = [
#     [ 97, 103, 101, 100,  99,  98, 102,  96, 104,  94], # 0
#     [ 97, 101, 100, 105, 102,  98,  99, 103,  95, 104], # 1
#     [100,  97,  98, 103, 101, 102,  99,  96, 104,  95], # 2
#     [ 97, 100, 101, 105,  99,  98, 102,  96, 104, 103], # 3
#     [101, 100,  97,  96, 103, 102,  99, 104, 105,  98], # 4
#     [ 99, 103, 101, 102, 100,  98, 104,  96, 105,  97], # 5
#     [101,  99,  98,  97, 102, 100, 104,  96, 103,  94], # 6 Same unique items as 0
#     [ 98, 102, 100, 101,  99, 104,  94, 103,  97,  95], # 7
#     [101, 103,  99, 100,  98, 104, 102,  96,  94,  97], # 8 Same unique items as 0, 6
#     [ 99, 101, 103,  98, 100, 105,  96, 104,  97, 102]  # 9
# ]
# # Add duplicates to test frequency accumulation based on identical sets
# queries.append(list(queries[0])) # 10 is identical to 0
# queries.append([ 97, 101, 100, 105, 102,  98,  99, 103,  95, 104]) # 11 is identical to 1
# queries.append(list(queries[8])) # 12 is identical to 8 (and thus 0, 6)
import json
top_5_dataset = "/home/jysc/Demnok/datasets/narrativeQA/top5_retrieval.jsonl"
print(f"Loading dataset from {top_5_dataset}")
with open(top_5_dataset, 'r') as f:
    lines = f.readlines()
queries = []
for line in lines:
    data = json.loads(line)
    query = data['top_k_doc_id']
    # Convert the query to a list of integers (or whatever format you need)
    # Here, we assume the query is a list of integers
    queries.append(query)
queries = queries[:100]
print(queries)
print(f"Finished loading dataset, total queries: {len(queries)}")
n_queries = len(queries) # Now n=13
# Perform clustering
# print("Performing clustering...")
# Z, cluster_nodes, unique_nodes = gpu_clustering(queries, similarity_method="sharp", linkage_method='average')
# print("Clustering completed.")
# # --- Use the Debug Module ---
# print("Debugging cluster nodes...")
# print_unique_cluster_tree(unique_nodes)
# # --- Optional: Visualize the dendrogram ---
# if n_queries >= 2:
#     plt.figure(figsize=(14, 8)) # Wider figure for more leaves
#     plt.title('Hierarchical Clustering Dendrogram (Sharp Similarity, Average Linkage)')
#     plt.xlabel('Query Index (Leaves) / Cluster ID (Internal Nodes)')
#     plt.ylabel('Distance (1 - Sharp Similarity)')
#     leaf_labels = [str(i) for i in range(n_queries)]
#     dendrogram(Z,
#                leaf_rotation=90.,
#                leaf_font_size=8.,
#                labels=leaf_labels,
#                )
#     plt.tight_layout()
#     plt.show()
# # --- Example: Check a node known to have high frequency ---
# # Find the content set for query 0
# content_of_0 = set(queries[0])
# print(f"\nChecking frequency for content: {sorted(list(content_of_0))}")
# for node_id, info in cluster_nodes.items():
#     if info['content'] == content_of_0:
#         print(f"  Node {node_id}: Frequency={info['frequency']}, Orig_Indices={sorted(list(info['original_indices']))}")
#         # Expect freq=4 for nodes 0, 6, 8, 10, 12 if they merge correctly
#         # Note: Intermediate merge nodes might also have this content set

# Example usage
def optimize_from_clustering(queries, k=5, doc_length=1200):
    """Run end-to-end optimization from raw queries."""
    print(f"Running clustering on {len(queries)} queries...")
    Z, cluster_nodes, unique_nodes = gpu_clustering(queries, similarity_method="sharp", 
                                 linkage_method='average', doc_length=doc_length)
    print_unique_cluster_tree(unique_nodes)
    print(f"Optimizing storage and routing for {len(cluster_nodes)} nodes...")
    results = optimize_rag_storage_and_routing(
        unique_nodes, 
        k=k,
        doc_length=doc_length
    )
    
    print_rag_optimization_results(unique_nodes, results)
    return results, unique_nodes, Z

# Run with the example queries
results, unique_nodes, Z = optimize_from_clustering(
    queries, 
    k=5  # Each query needs k=10 results
)