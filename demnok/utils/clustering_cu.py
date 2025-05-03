import numpy as np
from scipy.cluster.hierarchy import linkage
import cupy as cp

import cupy as cp
import numpy as np
from scipy.cluster.hierarchy import linkage

def batch_intersect_matrix(queries):
    """
    Vectorized computation of intersection counts between all query pairs using GPU acceleration.
    Returns matrix of same_items counts.
    """
    # Pad queries to uniform length and sort for efficient intersection
    max_len = max(len(q) for q in queries)
    padded = cp.array([cp.hstack([q, cp.full(max_len - len(q), -1)]) for q in queries])
    padded.sort(axis=1)
    
    # Create expanded views for broadcasting
    q1 = padded[:, None, :]  # Shape (n, 1, max_len)
    q2 = padded[None, :, :]  # Shape (1, n, max_len)
    
    # Vectorized intersection using sorted properties
    matches = q1 == q2
    valid = (q1 != -1) & (q2 != -1)
    intersection_counts = (matches & valid).sum(axis=2)
    
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
    
def gpu_clustering(queries, similarity_method="sharp", linkage_method='average', doc_length=1200):
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
    full_dist_matrix = batch_get_distance(queries, similarity_method)
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
    
    # Final validation to ensure all cluster_nodes point to valid unique_nodes
    for node_id in list(cluster_nodes.keys()):
        node = cluster_nodes[node_id]
        target_id = node.get('node_id')
        
        if target_id not in unique_nodes:
            # This should not happen after our fixes, but just in case:
            print(f"Warning: node {node_id} points to missing node {target_id}, fixing...")
            
            # Find a valid target
            valid_ids = list(unique_nodes.keys())
            if valid_ids:
                new_target = valid_ids[0]
                cluster_nodes[node_id] = unique_nodes[new_target]
                redirects[node_id] = new_target
            else:
                del cluster_nodes[node_id]
    
    # Print final stats for debugging
    print(f"Final tree has {len(unique_nodes)} unique nodes")
    leaf_nodes = [node_id for node_id, node in unique_nodes.items() 
                  if not node.get('children') or len(node['children']) == 0]
    print(f"Tree has {len(leaf_nodes)} leaf nodes")
    
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