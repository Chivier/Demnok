import numpy as np
from scipy.cluster.hierarchy import linkage
import cupy as cp

def batch_intersect_matrix(queries_cp_list): # queries_cp_list has unique, sorted CuPy arrays
    """
    Vectorized computation of intersection counts between all query pairs using GPU acceleration.
    queries_cp_list is a list of 1D CuPy arrays, each containing unique, sorted elements.
    Returns a CuPy matrix of same_items counts.
    """
    n = len(queries_cp_list)
    if n == 0:
        # Handle empty list of queries gracefully
        return cp.array([], dtype=cp.int32).reshape(0, 0)

    # Array of lengths of each query
    L = cp.array([len(q) for q in queries_cp_list], dtype=cp.int32)
    
    # If all queries are empty, the intersection matrix is all zeros.
    if L.sum() == 0:
        return cp.zeros((n, n), dtype=cp.int32)

    flat_queries = cp.concatenate(queries_cp_list)

    offsets_reduceat = cp.zeros(n, dtype=cp.int32) 
    if n > 1: 
        offsets_reduceat[1:] = cp.cumsum(L[:-1])
    
    # Initialize the result matrix
    intersection_counts = cp.empty((n, n), dtype=cp.int32)

    for j in range(n):
        Bj_elements = queries_cp_list[j] # The j-th query
        len_Bj = L[j]                   # Length of the j-th query

        if len_Bj == 0:
            # If the current query Bj is empty, its intersection with any other query is 0.
            intersection_counts[:, j] = 0
            continue
        
        indices_in_Bj = cp.searchsorted(Bj_elements, flat_queries)
        
        clipped_indices = cp.clip(indices_in_Bj, 0, len_Bj - 1)
        
        values_from_Bj = Bj_elements[clipped_indices]
        matches_flat = (values_from_Bj == flat_queries)

        current_col_sums = cp.add.reduceat(matches_flat.astype(cp.int32), offsets_reduceat)
        intersection_counts[:, j] = current_col_sums
        
    return intersection_counts

def batch_get_distance(queries, method="sharp"):
    """Batched similarity computation for all query pairs on GPU"""
    # Convert to CuPY arrays and ensure uniqueness
    queries_cp = [cp.unique(cp.asarray(q)) for q in queries]
    lens = cp.array([len(q) for q in queries_cp])
    
    # Get batched intersection counts
    same_items = batch_intersect_matrix(queries_cp)
    
    # Compute max lengths matrix
    k1 = lens[:, None]
    k2 = lens[None, :]
    max_k = cp.maximum(k1, k2)
    
    # Handle zero-length queries
    valid_pairs = (k1 > 0) & (k2 > 0)
    x = cp.zeros_like(same_items, dtype=cp.float32)
    x[valid_pairs] = same_items[valid_pairs] / max_k[valid_pairs]
    
    # Apply similarity method
    if method == "direct":
        similarity = x
    elif method == "square":
        similarity = x ** 2
    elif method == "sharp":
        a = 2
        exp_a = cp.exp(a)
        similarity = (cp.exp(a * x**2) - 1) / (exp_a - 1)
        similarity = cp.clip(similarity, 0.0, 1.0)
    else:
        similarity = x
        
    return 1.0 - similarity  # Return distance matrix
    
def gpu_clustering(queries, similarity_method="sharp", linkage_method='average', doc_length=1000):
    """
    Performs hierarchical clustering of queries with proper handling of duplicates.
    The frequency of a parent node is the sum of the frequencies of its children.
    
    Args:
        queries (list of lists/np.arrays): Input queries.
        similarity_method (str): Methods for get_distance function.
        linkage_method (str): Linkage method for scipy's linkage function.
        doc_length (int): Length of each document in the content.
    
    Returns:
        tuple: (Z, cluster_nodes, unique_nodes, reordered_queries) - linkage matrix, all node references, unique nodes, and reordered queries.
    """
    n = len(queries)
    # Convert queries to sets for efficient intersection operations
    original_query_sets = [set(q) for q in queries]
    
    # We'll maintain two structures:
    # 1. cluster_nodes - maps node IDs to their data or references to other nodes
    # 2. unique_nodes - only contains unique nodes with no duplicates
    cluster_nodes = {}
    unique_nodes = {}
    redirects = {}  # Maps redundant node IDs to their canonical node IDs
    
    if n < 2:
        print("Need at least two queries.")
        for i in range(n):
            query_set = original_query_sets[i]
            content_size = len(query_set)
            kv_seq_length = content_size * doc_length
            
            node_data = {
                'node_id': i,
                'content': query_set,
                'doc_ids': sorted(list(query_set)),
                'kv_memory_size': content_size,
                'kv_seq_length': kv_seq_length,
                'original_indices': {i},
                'distance': 0.0,
                'children': [],
                'parent': None,
                'frequency': 1
            }
            cluster_nodes[i] = node_data
            unique_nodes[i] = node_data
            
        # Return original queries as reordered queries when no clustering is possible
        return np.empty((0, 4)), cluster_nodes, unique_nodes, queries.copy()
    
    # Calculate distance matrix
    full_dist_matrix = batch_get_distance(queries, similarity_method)
    # print(full_dist_matrix)
    condensed_dist_matrix = full_dist_matrix[cp.triu_indices(n, k=1)].get()
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    Z = linkage(condensed_dist_matrix, method=linkage_method)
    print("Clustering completed.")
    
    # Track content to node ID mapping for deduplication
    content_to_node_id = {}
    
    # First create leaf nodes and handle duplicates
    for i in range(n):
        query_set = original_query_sets[i]
        content_key = frozenset(query_set)
        
        if content_key in content_to_node_id:
            # This content already exists
            canonical_id = content_to_node_id[content_key]
            node = unique_nodes[canonical_id]
            
            # Update frequency and original indices
            node['frequency'] += 1
            node['original_indices'].add(i)
            
            # Record this redirect
            redirects[i] = canonical_id
            
            # Point to the canonical node
            cluster_nodes[i] = node
        else:
            # New content
            content_size = len(query_set)
            kv_seq_length = content_size * doc_length
            
            node_data = {
                'node_id': i,
                'content': query_set,
                'doc_ids': sorted(list(query_set)),
                'kv_memory_size': content_size,
                'kv_seq_length': kv_seq_length,
                'original_indices': {i},
                'distance': 0.0,
                'children': [],
                'parent': None,
                'frequency': 1,
                'merge_distance': 0.0
            }
            
            # Register in both dictionaries
            cluster_nodes[i] = node_data
            unique_nodes[i] = node_data
            
            # Register this content
            content_to_node_id[content_key] = i
    
    # Process internal nodes
    for i in range(Z.shape[0]):
        new_id = n + i
        idx1, idx2, dist, _ = Z[i]
        idx1, idx2 = int(idx1), int(idx2)
        
        # Resolve to canonical nodes for both children
        canonical_idx1 = redirects.get(idx1, idx1)
        canonical_idx2 = redirects.get(idx2, idx2)
        
        # Skip self-references
        if canonical_idx1 == canonical_idx2:
            redirects[new_id] = canonical_idx1
            cluster_nodes[new_id] = unique_nodes[canonical_idx1]
            continue
            
        node1 = unique_nodes[canonical_idx1]
        node2 = unique_nodes[canonical_idx2]
        
        # Calculate intersection content
        intersection_content = node1['content'].intersection(node2['content'])
        content_key = frozenset(intersection_content)
        
        # Calculate combined frequency (sum of children frequencies)
        combined_frequency = node1['frequency'] + node2['frequency']
        
        # Combine original indices
        combined_indices = set()
        combined_indices.update(node1['original_indices'])
        combined_indices.update(node2['original_indices'])
        
        # Check if this content already exists
        if content_key in content_to_node_id and len(intersection_content) > 0:
            # This content already exists
            canonical_id = content_to_node_id[content_key]
            
            # Avoid self-reference
            if canonical_id == canonical_idx1 or canonical_id == canonical_idx2:
                # Create a new node instead to avoid self-reference
                content_size = len(intersection_content)
                kv_seq_length = content_size * doc_length
                
                node_data = {
                    'node_id': new_id,
                    'content': intersection_content,
                    'doc_ids': sorted(list(intersection_content)),
                    'kv_memory_size': content_size,
                    'kv_seq_length': kv_seq_length,
                    'original_indices': combined_indices,
                    'distance': dist,
                    'children': [canonical_idx1, canonical_idx2],
                    'parent': None,
                    'frequency': combined_frequency,
                    'merge_distance': dist
                }
                
                # Register in both dictionaries
                cluster_nodes[new_id] = node_data
                unique_nodes[new_id] = node_data
                
                # Update parent references
                node1['parent'] = new_id
                node2['parent'] = new_id
                
                # Don't register this content since it would conflict
            else:
                # Use the existing node
                node = unique_nodes[canonical_id]
                
                # Ensure we don't add self-references
                if canonical_idx1 != canonical_id and canonical_idx1 not in node['children']:
                    node['children'].append(canonical_idx1)
                    node1['parent'] = canonical_id
                
                if canonical_idx2 != canonical_id and canonical_idx2 not in node['children']:
                    node['children'].append(canonical_idx2)
                    node2['parent'] = canonical_id
                
                # Update frequency to ensure it's at least the sum of these children
                node['frequency'] = max(node['frequency'], combined_frequency)
                
                # Update original indices
                node['original_indices'].update(combined_indices)
                
                # Record this redirect
                redirects[new_id] = canonical_id
                
                # Point to the canonical node
                cluster_nodes[new_id] = node
        else:
            # New content
            content_size = len(intersection_content)
            kv_seq_length = content_size * doc_length
            
            node_data = {
                'node_id': new_id,
                'content': intersection_content,
                'doc_ids': sorted(list(intersection_content)),
                'kv_memory_size': content_size,
                'kv_seq_length': kv_seq_length,
                'original_indices': combined_indices,
                'distance': dist,
                'children': [canonical_idx1, canonical_idx2],
                'parent': None,
                'frequency': combined_frequency,  # Sum of child frequencies
                'merge_distance': dist
            }
            
            # Register in both dictionaries
            cluster_nodes[new_id] = node_data
            unique_nodes[new_id] = node_data
            
            # Register this content if not empty
            if len(intersection_content) > 0:
                content_to_node_id[content_key] = new_id
            
            # Set parent references
            node1['parent'] = new_id
            node2['parent'] = new_id
    
    # Clean up any self-references in children lists
    for node_id, node in unique_nodes.items():
        if 'children' in node:
            node['children'] = [child for child in node['children'] if child != node_id]
    
    # Identify empty nodes
    empty_nodes = {node_id for node_id, node in unique_nodes.items() if len(node['doc_ids']) == 0}
    
    if empty_nodes:
        print(f"Removing {len(empty_nodes)} empty nodes and fixing hierarchical relationships...")
        
        # First, update parent-child relationships to bypass empty nodes
        for empty_node_id in empty_nodes:
            if empty_node_id in unique_nodes:
                empty_node = unique_nodes[empty_node_id]
                children = empty_node.get('children', [])
                parent = empty_node.get('parent')
                
                # Connect children to parent, bypassing the empty node
                if parent is not None and parent not in empty_nodes:
                    parent_node = unique_nodes.get(parent)
                    if parent_node:
                        for child_id in children:
                            if child_id not in empty_nodes:
                                # Add child to parent's children if not already there
                                if child_id not in parent_node['children']:
                                    parent_node['children'].append(child_id)
                                
                                # Update child's parent reference
                                if child_id in unique_nodes:
                                    unique_nodes[child_id]['parent'] = parent
        
        # Second, update redirects to skip empty nodes
        for node_id, target_id in list(redirects.items()):
            if target_id in empty_nodes:
                # Follow redirect chain until we find a non-empty node
                while target_id in empty_nodes:
                    next_target = unique_nodes.get(target_id, {}).get('parent')
                    if next_target is None or next_target == target_id:
                        # If no parent or self-reference, find any valid child
                        children = unique_nodes.get(target_id, {}).get('children', [])
                        valid_children = [c for c in children if c not in empty_nodes]
                        if valid_children:
                            next_target = valid_children[0]  # Pick first valid child
                        else:
                            # If no valid child, find any valid leaf node
                            valid_leaves = [id for id in unique_nodes if id not in empty_nodes]
                            next_target = valid_leaves[0] if valid_leaves else None
                    
                    if next_target is None or next_target not in unique_nodes:
                        break  # Can't find a valid target
                    
                    target_id = next_target
                    
                    # If found a non-empty node, update redirect
                    if target_id not in empty_nodes:
                        redirects[node_id] = target_id
                        break
        
        # Finally, remove empty nodes from unique_nodes
        for node_id in empty_nodes:
            if node_id in unique_nodes:
                del unique_nodes[node_id]
    
    # Make a copy of all IDs to avoid modifying during iteration
    all_node_ids = list(cluster_nodes.keys())
    
    # Ensure cluster_nodes only contains or points to valid nodes
    for node_id in all_node_ids:
        # If node_id was an empty node or points to one through redirects
        if node_id not in unique_nodes:
            target_id = redirects.get(node_id)
            
            if target_id is not None and target_id in unique_nodes:
                # Valid redirect exists, update cluster_nodes
                cluster_nodes[node_id] = unique_nodes[target_id]
            else:
                # No valid redirect, need to find a valid node
                # Try to find any valid node - preferably a root node
                valid_nodes = list(unique_nodes.keys())
                if valid_nodes:
                    # Look for root nodes (nodes without parents)
                    root_nodes = [id for id in valid_nodes if unique_nodes[id].get('parent') is None]
                    # If no root nodes, use any valid node
                    valid_target = root_nodes[0] if root_nodes else valid_nodes[0]
                    
                    cluster_nodes[node_id] = unique_nodes[valid_target]
                    redirects[node_id] = valid_target
                else:
                    # No valid nodes at all (should never happen if we had valid input)
                    del cluster_nodes[node_id]
    
    # Validate that all nodes in unique_nodes have valid children and parent references
    for node_id, node in unique_nodes.items():
        # Fix children list to only include valid nodes
        if 'children' in node:
            node['children'] = [child_id for child_id in node['children'] if child_id in unique_nodes]
        
        # Fix parent reference if it points to an invalid node
        if node.get('parent') is not None and node['parent'] not in unique_nodes:
            node['parent'] = None

    for node_id in list(cluster_nodes.keys()):
        node = cluster_nodes[node_id]
        target_id = node.get('node_id')
        
        if target_id not in unique_nodes:
            print(f"Warning: node {node_id} points to missing node {target_id}, fixing...")
            
            valid_ids = list(unique_nodes.keys())
            if valid_ids:
                new_target = valid_ids[0]
                cluster_nodes[node_id] = unique_nodes[new_target]
                redirects[node_id] = new_target
            else:
                del cluster_nodes[node_id]
    
    print(f"Final tree has {len(unique_nodes)} unique nodes")
    leaf_nodes = [node_id for node_id, node in unique_nodes.items() 
                  if not node.get('children') or len(node['children']) == 0]
    print(f"Tree has {len(leaf_nodes)} leaf nodes")
    
    reordered_queries = get_reordered_queries(queries, unique_nodes)

    # for x in queries:
    #     print(f"Original query: {x}")
    
    # for x in reordered_queries:
    #     print(f"Reordered query early: {x}")
    # organized_queries = post_process_and_reorder_for_max_lcp(reordered_queries)
    organized_queries, organized_orig_queries, organized_orig_idx = post_process_and_reorder_for_max_lcp_optimized_multi_core_with_indices(reordered_queries, queries)
    
    return Z, cluster_nodes, unique_nodes, organized_queries, organized_orig_queries, organized_orig_idx


def print_unique_cluster_tree(unique_nodes):
    """
    Prints only unique nodes in the cluster tree, avoiding duplicates.
    
    Args:
        unique_nodes (dict): Dictionary of unique nodes from clustering function.
    """
    print("\n--- Unique Cluster Tree Nodes ---")
    
    sorted_nodes = sorted(unique_nodes.items())
    freq_sum = 0
    node_cnt = 0
    for node_id, info in sorted_nodes:
        content_list = sorted(list(info['content']))
        original_indices_list = sorted(list(info['original_indices']))
        leaf_status = "Leaf" if len(info.get('children', [])) == 0 else "Internal"
        
        print(f"Node {node_id} ({leaf_status}):")
        print(f"  node_id: {info['node_id']}")
        print(f"  doc_ids (content): {info['doc_ids']}")
        print(f"  kv_memory_size: {info['kv_memory_size']}")
        print(f"  kv_seq_length: {info['kv_seq_length']}")
        print(f"  parent: {info['parent'] if info['parent'] is not None else 'None (Root)'}")
        print(f"  frequency: {info['frequency']}")
        print(f"  original_indices: {original_indices_list}")
        
        if leaf_status == "Internal":
            print(f"  merge_distance: {info.get('merge_distance', 0.0):.4f}")
            print(f"  children_nodes: {info['children']}")
   
        print("-" * 40)
        if len(info['doc_ids']) >= 1 and info['parent'] is None:
            freq_sum += len(info['doc_ids'])
            node_cnt += 1
    
    print("-" * 40)
    print(f"Max frequency in unique nodes: {freq_sum - node_cnt}")

def get_reordered_queries(queries, unique_nodes):
    """
    Generate reordered queries based on clustering results.
    
    For each original query, find its corresponding leaf node and traverse up the tree
    to find the largest node (by kv_memory_size) with frequency > 1. Use that node's
    content as prefix, followed by remaining elements from original query.
    
    Special case: If the leaf node's parent is a root node, directly return the leaf node.
    
    Args:
        queries (list): Original input queries
        unique_nodes (dict): Dictionary of unique nodes from clustering
        
    Returns:
        list: Reordered queries following the clustering-based optimization
    """
    reordered_queries = []
    
    for i, original_query in enumerate(queries):
        original_set = set(original_query)
        
        leaf_node = None
        for node_id, node in unique_nodes.items():
            if i in node['original_indices'] and len(node.get('children', [])) == 0:
                leaf_node = node
                break
        
        if leaf_node is None:
            reordered_queries.append(list(original_query))
            continue
        
        if leaf_node.get('parent') is None:
            reordered_query = list(leaf_node['content'])
            reordered_queries.append(reordered_query)
            continue
        
        best_node = None
        current_node = leaf_node
        
        if current_node['frequency'] > 1:
            best_node = current_node
            prefix_content = set(best_node['content'])
            prefix_list = sorted(list(prefix_content))
            
            remaining_elements = original_set - prefix_content
            remaining_list = sorted(list(remaining_elements))
            
            reordered_query = prefix_list + remaining_list
            reordered_queries.append(reordered_query)
            continue
        
        while current_node.get('parent') is not None:
            parent_id = current_node['parent']
            parent_node = unique_nodes[parent_id]
                
            if parent_node['frequency'] > 1:
                best_node = parent_node
                break
            
            current_node = parent_node
        
        if best_node is not None:
            prefix_content = set(best_node['content'])
            prefix_list = sorted(list(prefix_content))
            
            remaining_elements = original_set - prefix_content
            remaining_list = sorted(list(remaining_elements))
            
            reordered_query = prefix_list + remaining_list
        else:
            reordered_query = list(original_query)
        
        reordered_queries.append(reordered_query)
    
    return reordered_queries


def _longest_common_prefix_for_grouping(q1, q2):
    """
    Helper to find the length of the longest common prefix between two lists.
    Used for the initial grouping mechanism.
    """
    lcp = 0

    for item1, item2 in zip(q1, q2): 
        if item1 == item2:
            lcp += 1
        else:
            break
    return lcp

def _get_prefix_organized_groups(reordered_queries):
    """
    Core logic from organize_queries_by_prefix_advanced.
    Forms clusters of queries based on LCP, sorts clusters by their formation
    strength, and sorts queries within clusters.
    
    Returns:
        list of lists: A list where each inner list is a group (cluster) of queries.
                       Groups are ordered, and queries within groups are ordered.
    """
    if not reordered_queries:
        return []
    
    n = len(reordered_queries)
    if n == 0:
        return []

    similarity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n): 
            lcp = _longest_common_prefix_for_grouping(reordered_queries[i], reordered_queries[j])
            similarity_matrix[i][j] = lcp
            similarity_matrix[j][i] = lcp
            
    used = [False] * n
    clusters_with_strength = [] 
    
    for i in range(n): 
        if used[i]:
            continue
            
        current_cluster_indices = [i] 
        used[i] = True
        initial_merge_strength = 0 
        
        while True:
            best_candidate_to_add_idx = -1
            max_lcp_with_current_cluster = 0 
            
            for j in range(n): 
                if used[j]:
                    continue
                
                current_candidate_max_lcp = 0
                for member_idx_in_cluster in current_cluster_indices:
                    current_candidate_max_lcp = max(current_candidate_max_lcp, 
                                                    similarity_matrix[member_idx_in_cluster][j])
                
                if current_candidate_max_lcp > max_lcp_with_current_cluster:
                    max_lcp_with_current_cluster = current_candidate_max_lcp
                    best_candidate_to_add_idx = j
            
            if max_lcp_with_current_cluster > 0: 
                if len(current_cluster_indices) == 1: 
                    initial_merge_strength = max_lcp_with_current_cluster
                
                current_cluster_indices.append(best_candidate_to_add_idx)
                used[best_candidate_to_add_idx] = True
            else:
                break
        
        actual_queries_in_cluster = [reordered_queries[idx] for idx in current_cluster_indices]
        
        actual_queries_in_cluster.sort(key=lambda q_list: (-len(q_list), q_list))
        
        clusters_with_strength.append((initial_merge_strength, actual_queries_in_cluster))

    clusters_with_strength.sort(key=lambda entry_tuple: (-entry_tuple[0], entry_tuple[1][0] if entry_tuple[1] else []))
    
    grouped_queries_ordered = []
    for _, query_list_in_cluster in clusters_with_strength:
        if query_list_in_cluster: 
            grouped_queries_ordered.append(query_list_in_cluster)
            
    return grouped_queries_ordered

def post_process_and_reorder_for_max_lcp(reordered_queries_from_clustering):
    """
    Takes queries (already reordered by the main clustering logic),
    groups them by LCP, and then post-processes each group to maximize
    the shared prefix based on the group's full set intersection.
    Maintains inter-group and intra-group ordering from the initial grouping.

    Args:
        reordered_queries_from_clustering (list of lists): Queries as output by
            the get_reordered_queries function in your main clustering pipeline.

    Returns:
        list of lists: The final flat list of queries, reordered and rewritten.
    """
    if not reordered_queries_from_clustering:
        return []

    list_of_query_groups = _get_prefix_organized_groups(reordered_queries_from_clustering)

    final_reordered_queries_flat = []
    
    for group in list_of_query_groups:
        if not group:
            continue

        if len(group) == 1: 
            final_reordered_queries_flat.append(group[0])
            continue

        common_elements_set = set(group[0])
        for i in range(1, len(group)):
            common_elements_set.intersection_update(set(group[i]))
        
        group_maximized_shared_prefix_list = sorted(list(common_elements_set))

        for original_query_in_group in group:
            original_query_as_set = set(original_query_in_group)
            
            remainder_elements_set = original_query_as_set - common_elements_set
            remainder_elements_list = sorted(list(remainder_elements_set))

            new_query = group_maximized_shared_prefix_list + remainder_elements_list
            final_reordered_queries_flat.append(new_query)
            
    return final_reordered_queries_flat


import multiprocessing
from functools import partial
import time

def _longest_common_prefix_for_grouping_v3(q1, q2):
    """
    Helper to find the length of the longest common prefix between two lists.
    (Identical to v1, repeated for self-containment)
    """
    lcp = 0
    for item1, item2 in zip(q1, q2):
        if item1 == item2:
            lcp += 1
        else:
            break
    return lcp

# Worker function for multiprocessing the LCP calculation
def _calculate_lcp_for_pair_v3(args, queries_ref):
    i, j = args
    return i, j, _longest_common_prefix_for_grouping_v3(queries_ref[i][0], queries_ref[j][0])  # Access the query part

def _get_prefix_organized_groups_optimized_mc(reordered_queries_with_indices):
    """
    Optimized core logic for forming clusters, with multiprocessing for similarity matrix.
    Now works with (query, original_index) tuples.
    """
    if not reordered_queries_with_indices:
        return []
    
    n = len(reordered_queries_with_indices)
    if n == 0:
        return []
    similarity_matrix = [[0] * n for _ in range(n)]
    
    # --- Parallelize Similarity Matrix Calculation ---
    indices_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            indices_pairs.append((i, j))
    
    if indices_pairs:
        num_processes = min(multiprocessing.cpu_count(), len(indices_pairs))
        if num_processes > 0:
            worker_func = partial(_calculate_lcp_for_pair_v3, queries_ref=reordered_queries_with_indices)
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(worker_func, indices_pairs)
                
                for r_i, r_j, lcp in results:
                    similarity_matrix[r_i][r_j] = lcp
                    similarity_matrix[r_j][r_i] = lcp
            except Exception as e:
                print(f"Multiprocessing pool error during LCP calculation: {e}")
                for i_s, j_s in indices_pairs:
                    lcp_s = _longest_common_prefix_for_grouping_v3(
                        reordered_queries_with_indices[i_s][0], 
                        reordered_queries_with_indices[j_s][0]
                    )
                    similarity_matrix[i_s][j_s] = lcp_s
                    similarity_matrix[j_s][i_s] = lcp_s
        else:
             for i_s, j_s in indices_pairs:
                lcp_s = _longest_common_prefix_for_grouping_v3(
                    reordered_queries_with_indices[i_s][0], 
                    reordered_queries_with_indices[j_s][0]
                )
                similarity_matrix[i_s][j_s] = lcp_s
                similarity_matrix[j_s][i_s] = lcp_s
    # --- End Parallel Similarity Matrix ---
            
    used = [False] * n
    clusters_with_strength = [] 
    
    # Clustering logic remains largely sequential after similarity matrix is built
    for i in range(n): 
        if used[i]:
            continue
            
        current_cluster_indices = [i] 
        used[i] = True
        initial_merge_strength = 0
        
        best_partner_for_seed_idx = -1
        max_lcp_for_seed = 0
        for j_partner in range(n):
            if i == j_partner or used[j_partner]:
                continue
            if similarity_matrix[i][j_partner] > max_lcp_for_seed:
                max_lcp_for_seed = similarity_matrix[i][j_partner]
                best_partner_for_seed_idx = j_partner
        
        if max_lcp_for_seed > 0 and best_partner_for_seed_idx != -1:
            if not used[best_partner_for_seed_idx]:
                initial_merge_strength = max_lcp_for_seed
                current_cluster_indices.append(best_partner_for_seed_idx)
                used[best_partner_for_seed_idx] = True
        
        max_lcp_to_cluster_candidate = [-1] * n
        for member_idx_in_cluster in current_cluster_indices:
            for j_cand in range(n):
                if used[j_cand]: 
                    continue
                max_lcp_to_cluster_candidate[j_cand] = max(max_lcp_to_cluster_candidate[j_cand], 
                                                           similarity_matrix[member_idx_in_cluster][j_cand])
        
        while True:
            best_candidate_to_add_idx = -1
            current_max_lcp_to_cluster = 0 
            for j_cand in range(n):
                if used[j_cand]:
                    continue
                if max_lcp_to_cluster_candidate[j_cand] > current_max_lcp_to_cluster:
                    current_max_lcp_to_cluster = max_lcp_to_cluster_candidate[j_cand]
                    best_candidate_to_add_idx = j_cand
            
            if current_max_lcp_to_cluster > 0 and best_candidate_to_add_idx != -1:
                if not used[best_candidate_to_add_idx]:
                    current_cluster_indices.append(best_candidate_to_add_idx)
                    used[best_candidate_to_add_idx] = True
                    newly_added_member_idx = best_candidate_to_add_idx
                    for j_update in range(n):
                        if used[j_update]:
                            continue
                        max_lcp_to_cluster_candidate[j_update] = max(max_lcp_to_cluster_candidate[j_update],
                                                                    similarity_matrix[newly_added_member_idx][j_update])
                else:
                    break
            else:
                break
        
        # Keep both query and original index
        actual_queries_in_cluster = [reordered_queries_with_indices[idx] for idx in current_cluster_indices]
        actual_queries_in_cluster.sort(key=lambda q_tuple: (-len(q_tuple[0]), q_tuple[0]))
        clusters_with_strength.append((initial_merge_strength, actual_queries_in_cluster))
    
    clusters_with_strength.sort(key=lambda entry_tuple: (-entry_tuple[0], entry_tuple[1][0][0] if entry_tuple[1] else []))
    grouped_queries_ordered = [query_list for _, query_list in clusters_with_strength if query_list]
    return grouped_queries_ordered

def _process_group_worker_v3(group):
    """
    Worker function for post-processing a single group in parallel.
    Now works with (query, original_index) tuples.
    """
    if not group:
        return []
    if len(group) == 1:
        return [group[0]]
    
    processed_queries_for_group = []
    try:
        # Extract queries for processing
        queries_only = [item[0] for item in group]
        
        common_elements_set = set(queries_only[0])
        for i in range(1, len(queries_only)):
            common_elements_set.intersection_update(set(queries_only[i]))
        
        group_maximized_shared_prefix_list = sorted(list(common_elements_set))
        
        for original_query_tuple in group:
            original_query, original_idx = original_query_tuple
            original_query_as_set = set(original_query)
            remainder_elements_set = original_query_as_set - common_elements_set
            remainder_elements_list = sorted(list(remainder_elements_set))
            new_query = group_maximized_shared_prefix_list + remainder_elements_list
            processed_queries_for_group.append((new_query, original_idx))
    except TypeError:
        processed_queries_for_group.extend(group)  # Fallback
            
    return processed_queries_for_group

def post_process_and_reorder_for_max_lcp_optimized_multi_core_with_indices(reordered_queries, original_queries):
    """
    Post-processes queries using multi-core optimized grouping and parallel group processing.
    Returns organized queries, corresponding original queries, and index mapping.
    
    Args:
        reordered_queries: List of reordered queries
        original_queries: List of original queries (same length as reordered_queries)
    
    Returns:
        tuple: (organized_queries, organized_original_queries, index_mapping)
        where index_mapping shows the new position order of original indices
        
    Example:
        If original order was [0,1,2] and after organizing it becomes [2,1,0],
        then index_mapping = [2,1,0]
    """
    if not reordered_queries:
        return [], [], []
    
    # Create tuples of (reordered_query, original_index) to track indices
    reordered_queries_with_indices = [(reordered_queries[i], i) for i in range(len(reordered_queries))]
    
    # Step 1: Get groups (this uses multiprocessing for similarity matrix)
    list_of_query_groups = _get_prefix_organized_groups_optimized_mc(reordered_queries_with_indices)
    
    if not list_of_query_groups:
        return [], [], []
    
    final_reordered_queries_flat = []
    final_original_indices = []
    
    # --- Parallelize Group Processing ---
    if list_of_query_groups:
        num_processes_groups = min(multiprocessing.cpu_count(), len(list_of_query_groups))
        if num_processes_groups > 0:
            try:
                with multiprocessing.Pool(processes=num_processes_groups) as pool:
                    list_of_processed_group_queries = pool.map(_process_group_worker_v3, list_of_query_groups)
                
                for processed_group in list_of_processed_group_queries:
                    for processed_query, original_idx in processed_group:
                        final_reordered_queries_flat.append(processed_query)
                        final_original_indices.append(original_idx)
            except Exception as e:
                print(f"Multiprocessing pool error during group processing: {e}")
                # Fallback to serial processing for groups
                for group in list_of_query_groups:
                    processed_group = _process_group_worker_v3(group)
                    for processed_query, original_idx in processed_group:
                        final_reordered_queries_flat.append(processed_query)
                        final_original_indices.append(original_idx)
        else:
            for group in list_of_query_groups:
                processed_group = _process_group_worker_v3(group)
                for processed_query, original_idx in processed_group:
                    final_reordered_queries_flat.append(processed_query)
                    final_original_indices.append(original_idx)
    
    # Extract the original queries in the new order
    organized_original_queries = [original_queries[idx] for idx in final_original_indices]
    
    # Return the organized queries, original queries, and the index mapping
    return final_reordered_queries_flat, organized_original_queries, final_original_indices