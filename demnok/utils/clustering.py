import numpy as np
from scipy.cluster.hierarchy import linkage

def get_distance(query_1, query_2, method="sharp"):
    """
    Calculates the similarity between two queries (lists or arrays of items).
    [Function content identical to previous answer - omitted for brevity]
    """
    # Ensure inputs are numpy arrays for intersect1d
    q1 = np.asarray(query_1)
    q2 = np.asarray(query_2)
    k1 = len(q1)
    k2 = len(q2)
    if k1 == 0 or k2 == 0:
      return 0.0
    max_k = max(k1, k2)
    # Use direct intersection if presence matters more than count/order
    same_items = len(np.intersect1d(q1, q2))
    # Avoid division by zero if max_k is 0 (both queries empty)
    if max_k == 0:
        return 1.0 # Both empty, max similarity
    x = same_items / max_k
    if method == "direct":
        return x
    elif method == "square":
        return x ** 2
    elif method == "sharp":
        # f(x) = (e^(ax^2) - 1)/(e^a - 1), a = 2
        a = 2
        exp_a = np.exp(a)
        if exp_a == 1: # Avoid division by zero if a=0
             return x**2 # Limit case as a -> 0
        # Use np.clip to avoid potential precision issues near x=0 or x=1
        val = (np.exp(a * x**2) - 1) / (exp_a - 1)
        return np.clip(val, 0.0, 1.0)
    else:
        # default to sharp
        a = 2
        exp_a = np.exp(a)
        if exp_a == 1:
             return x**2
        val = (np.exp(a * x**2) - 1) / (exp_a - 1)
        return np.clip(val, 0.0, 1.0)
    
def clustering(queries, similarity_method="sharp", linkage_method='average', doc_length=1200):
    """
    Performs hierarchical clustering of queries with proper handling of duplicates.
    The frequency of a parent node is the sum of the frequencies of its children.
    
    Args:
        queries (list of lists/np.arrays): Input queries.
        similarity_method (str): Methods for get_distance function.
        linkage_method (str): Linkage method for scipy's linkage function.
        doc_length (int): Length of each document in the content.
    
    Returns:
        tuple: (Z, cluster_nodes, unique_nodes) - linkage matrix, all node references, and unique nodes.
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
            
        return np.empty((0, 4)), cluster_nodes, unique_nodes
    
    # Calculate distance matrix
    print("Calculating distance matrix...")
    condensed_dist_matrix = np.zeros(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            similarity = get_distance(queries[i], queries[j], method=similarity_method)
            distance = 1.0 - similarity
            condensed_dist_matrix[k] = max(0.0, distance)
            k += 1
    print("Distance matrix calculation completed.")
    
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
    
    return Z, cluster_nodes, unique_nodes


def print_unique_cluster_tree(unique_nodes):
    """
    Prints only unique nodes in the cluster tree, avoiding duplicates.
    
    Args:
        unique_nodes (dict): Dictionary of unique nodes from clustering function.
    """
    print("\n--- Unique Cluster Tree Nodes ---")
    
    # Sort nodes by ID for cleaner output
    sorted_nodes = sorted(unique_nodes.items())
    
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
        
        if leaf_status == "Internal": # Internal node specific info
            print(f"  merge_distance: {info.get('merge_distance', 0.0):.4f}")
            print(f"  children_nodes: {info['children']}")
        
        print("-" * 40) # Separator