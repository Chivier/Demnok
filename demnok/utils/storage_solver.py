import pulp
import numpy as np
import collections
import time

def prepare_optimization_data(unique_nodes, k, doc_length, block_size=None):
    """
    Prepare all necessary data structures for optimization from unique nodes.
    
    Args:
        unique_nodes: Dictionary of unique cluster nodes
        k: Number of documents each query needs
        doc_length: Average sequence length per document
        block_size: Size of KV memory blocks in bytes (optional)
        
    Returns:
        Dictionary with prepared data structures
    """
    # Set default block size if not provided
    if block_size is None:
        # ~33.5MB per content element
        block_size = 1200 * 576 * 61 * 2 * 5
    
    # Find all leaf nodes (nodes without children)
    leaf_nodes = []
    for node_id, info in unique_nodes.items():
        if info.get('children', []) == []:  # Leaf node has empty children list
            leaf_nodes.append(node_id)
    
    print(f"Tree has {len(unique_nodes)} unique nodes with {len(leaf_nodes)} leaf nodes")
    
    # Build parent-child relationship maps
    parent_map = {}  # child_id -> parent_id
    children_map = {}  # parent_id -> [child_ids]
    
    for node_id, info in unique_nodes.items():
        parent_id = info.get('parent')
        if parent_id is not None:
            parent_map[node_id] = parent_id
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node_id)
    
    # Get ancestor map (all ancestors for each node)
    ancestor_map = {}
    
    for node_id in unique_nodes:
        ancestor_map[node_id] = set()
        curr = node_id
        while curr in parent_map:
            parent_id = parent_map[curr]
            ancestor_map[node_id].add(parent_id)
            curr = parent_id
    
    # Calculate memory requirements for each node
    memory_requirements = {}
    for node_id, info in unique_nodes.items():
        content_size = len(info['content'])
        memory_requirements[node_id] = content_size * block_size
    
    return {
        'leaf_nodes': leaf_nodes,
        'parent_map': parent_map,
        'children_map': children_map,
        'ancestor_map': ancestor_map,
        'memory_requirements': memory_requirements,
        'block_size': block_size
    }


def calculate_storage_cost(query_id, server_id, storage, unique_nodes, memory_requirements, k, doc_length, performance_bench):
    """
    Calculate cost for serving a query from a specific server node and storage location.
    
    Args:
        query_id: ID of the query being served
        server_id: ID of the server node providing the content
        storage: Storage location ('gpu', 'dram', 'disk', or 'not_stored')
        unique_nodes: Dictionary of unique cluster nodes
        memory_requirements: Dictionary with memory requirements for each node
        k: Number of documents each query needs
        doc_length: Average sequence length per document
        performance_bench: Performance benchmarks dictionary
        
    Returns:
        Total cost value
    """
    # Get content sizes
    server_content_size = len(unique_nodes[server_id]['content'])
    
    # How many more documents do we need to compute beyond what's in the server?
    # If server content is >= k, we don't need to compute more
    # If server content < k, we need to compute (k - server_content_size)
    remaining_k = max(0, k - server_content_size)
    
    # Calculate memory size in GB
    server_memory_bytes = memory_requirements[server_id]
    server_memory_gb = server_memory_bytes / (1024**3)
    
    # Calculate cost based on storage tier
    if storage == 'gpu':
        # Just compute time for remaining items
        compute_seq_length = remaining_k * doc_length
        compute_time = compute_seq_length / performance_bench["prefill_throughput"]
        access_cost = 0.00001  # Negligible cost for GPU access
        total_cost = access_cost + compute_time
    elif storage == 'dram':
        # DRAM to GPU transfer + compute time
        h2d_time = server_memory_gb / performance_bench["PCIe"]
        compute_seq_length = remaining_k * doc_length
        compute_time = compute_seq_length / performance_bench["prefill_throughput"]
        total_cost = h2d_time + compute_time
    elif storage == 'disk':
        # Disk to DRAM to GPU + compute time
        disk2dram_time = server_memory_gb / performance_bench["NVMe"]
        dram2gpu_time = server_memory_gb / performance_bench["PCIe"]
        compute_seq_length = remaining_k * doc_length
        compute_time = compute_seq_length / performance_bench["prefill_throughput"]
        total_cost = disk2dram_time + dram2gpu_time + compute_time
    else:  # not_stored
        # Full computation from scratch
        compute_seq_length = k * doc_length
        compute_time = compute_seq_length / performance_bench["prefill_throughput"]
        # Make not_stored very expensive
        total_cost = compute_time
    
    return total_cost


def build_optimization_model(unique_nodes, data, capacities, k, doc_length, performance_bench):
    """
    Build the ILP optimization model.
    
    Args:
        unique_nodes: Dictionary of unique cluster nodes
        data: Data structures from prepare_optimization_data
        capacities: Dictionary with storage capacities in bytes
        k: Number of documents each query needs
        doc_length: Average length of documents
        performance_bench: Performance benchmarks
        
    Returns:
        pulp.LpProblem object with the optimization model
    """
    leaf_nodes = data['leaf_nodes']
    ancestor_map = data['ancestor_map']
    memory_requirements = data['memory_requirements']
    
    # Create the ILP problem
    prob = pulp.LpProblem("RAG_Storage_Optimization", pulp.LpMinimize)
    
    # DECISION VARIABLES
    
    # 1. Storage placement variables: where each content node is stored
    storage_vars = {}
    for node_id in unique_nodes:
        for storage in ['gpu', 'dram', 'disk', 'not_stored']:
            storage_vars[(node_id, storage)] = pulp.LpVariable(
                f"store_{node_id}_on_{storage}", cat='Binary')
    
    # 2. Query routing variables: which node serves each query
    routing_vars = {}
    for leaf_id in leaf_nodes:
        # A query can be served by itself or any ancestor
        possible_servers = [leaf_id] + list(ancestor_map[leaf_id])
        for server_id in possible_servers:
            routing_vars[(leaf_id, server_id)] = pulp.LpVariable(
                f"route_{leaf_id}_via_{server_id}", cat='Binary')
    
    # CONSTRAINTS
    
    # 1. Each content node must be in exactly one storage location (or not stored)
    for node_id in unique_nodes:
        prob += pulp.lpSum(storage_vars[(node_id, storage)] 
                          for storage in ['gpu', 'dram', 'disk', 'not_stored']) == 1
    
    # 2. Each query (leaf node) must be routed through exactly one server node
    for leaf_id in leaf_nodes:
        possible_servers = [leaf_id] + list(ancestor_map[leaf_id])
        prob += pulp.lpSum(routing_vars[(leaf_id, server_id)] 
                          for server_id in possible_servers) == 1
    
    # 3. Storage capacity constraints
    prob += pulp.lpSum(storage_vars[(node_id, 'gpu')] * memory_requirements[node_id]
                      for node_id in unique_nodes) <= capacities['gpu']
    
    prob += pulp.lpSum(storage_vars[(node_id, 'dram')] * memory_requirements[node_id]
                      for node_id in unique_nodes) <= capacities['dram']
    
    prob += pulp.lpSum(storage_vars[(node_id, 'disk')] * memory_requirements[node_id]
                      for node_id in unique_nodes) <= capacities['disk']
    
    # OBJECTIVE FUNCTION: Minimize total query processing cost
    
    # Cache cost calculations
    cost_cache = {}
    
    # Build the objective function
    obj_expr = 0
    
    for leaf_id in leaf_nodes:
        # Use frequency to weight the cost of serving this query
        frequency = unique_nodes[leaf_id]['frequency']
        possible_servers = [leaf_id] + list(ancestor_map[leaf_id])
        
        for server_id in possible_servers:
            for storage in ['gpu', 'dram', 'disk', 'not_stored']:
                # Calculate or retrieve cost
                cost_key = (leaf_id, server_id, storage)
                if cost_key not in cost_cache:
                    cost_cache[cost_key] = calculate_storage_cost(
                        leaf_id, server_id, storage, 
                        unique_nodes, memory_requirements,
                        k, doc_length, performance_bench
                    )
                cost = cost_cache[cost_key]
                
                # Linearize the product: routing_vars * storage_vars
                aux_var = pulp.LpVariable(
                    f"aux_{leaf_id}_via_{server_id}_on_{storage}", lowBound=0, upBound=1)
                
                # Enforce that aux_var = routing_vars[(leaf_id, server_id)] * storage_vars[(server_id, storage)]
                prob += aux_var <= routing_vars[(leaf_id, server_id)]
                prob += aux_var <= storage_vars[(server_id, storage)]
                prob += aux_var >= routing_vars[(leaf_id, server_id)] + storage_vars[(server_id, storage)] - 1
                
                # Add this component to the objective function
                obj_expr += frequency * aux_var * cost
    
    prob += obj_expr
    
    return prob, storage_vars, routing_vars


def process_optimization_results(unique_nodes, data, storage_vars, routing_vars, prob, k, capacities):
    """
    Process the results from the optimization problem.
    
    Args:
        unique_nodes: Dictionary of unique cluster nodes
        data: Data structures from prepare_optimization_data
        storage_vars: Storage placement decision variables
        routing_vars: Query routing decision variables
        prob: The solved pulp problem
        k: Number of documents each query needs
        capacities: Dictionary with storage capacities in bytes
        
    Returns:
        Dictionary with optimization results
    """
    leaf_nodes = data['leaf_nodes']
    parent_map = data['parent_map']
    memory_requirements = data['memory_requirements']
    doc_length = data.get('doc_length', 1200)
    performance_bench = data.get('performance_bench', {
        "NVMe": 6,
        "PCIe": 128,
        "prefill_throughput": 5000
    })
    
    # 1. Storage assignments
    storage_assignment = {}
    for node_id in unique_nodes:
        for storage in ['gpu', 'dram', 'disk', 'not_stored']:
            if pulp.value(storage_vars[(node_id, storage)]) == 1:
                storage_assignment[node_id] = storage
                break
    
    # 2. Query routing decisions
    query_routing = {}
    for leaf_id in leaf_nodes:
        possible_servers = [leaf_id] + list(data['ancestor_map'][leaf_id])
        for server_id in possible_servers:
            if pulp.value(routing_vars[(leaf_id, server_id)]) == 1:
                query_routing[leaf_id] = server_id
                break
    
    # Calculate final costs and statistics
    query_costs = {}
    total_computed_items = 0
    
    for leaf_id in leaf_nodes:
        server_id = query_routing[leaf_id]
        storage = storage_assignment[server_id]
        frequency = unique_nodes[leaf_id]['frequency']
        
        # Calculate cost
        cost = calculate_storage_cost(
            leaf_id, server_id, storage, 
            unique_nodes, memory_requirements,
            k, doc_length, performance_bench
        )
        
        # Calculate how many items needed computation
        server_content_size = len(unique_nodes[server_id]['content'])
        remaining_k = max(0, k - server_content_size)
        total_computed_items += remaining_k * frequency
        
        query_costs[leaf_id] = {
            'server_node': server_id,
            'storage': storage,
            'server_content_size': server_content_size,
            'additional_compute': remaining_k,
            'per_query_cost': cost,
            'frequency': frequency,
            'total_cost': cost * frequency
        }
    
    # Calculate storage utilization
    gpu_used = sum(memory_requirements[node_id] for node_id, storage in storage_assignment.items()
                  if storage == 'gpu')
    dram_used = sum(memory_requirements[node_id] for node_id, storage in storage_assignment.items()
                   if storage == 'dram')
    disk_used = sum(memory_requirements[node_id] for node_id, storage in storage_assignment.items()
                   if storage == 'disk')
    
    storage_util = {
        'gpu_used_bytes': gpu_used,
        'gpu_used_gb': gpu_used / (1024**3),
        'gpu_capacity_gb': capacities['gpu'] / (1024**3),
        'gpu_utilization': gpu_used / capacities['gpu'] * 100,
        
        'dram_used_bytes': dram_used,
        'dram_used_gb': dram_used / (1024**3),
        'dram_capacity_gb': capacities['dram'] / (1024**3),
        'dram_utilization': dram_used / capacities['dram'] * 100,
        
        'disk_used_bytes': disk_used,
        'disk_used_gb': disk_used / (1024**3),
        'disk_capacity_gb': capacities['disk'] / (1024**3),
        'disk_utilization': disk_used / capacities['disk'] * 100,
    }
    
    # Calculate query routing statistics
    routing_stats = {
        'direct': 0,  # Queries served by their own leaf node
        'via_ancestor': 0,  # Queries served by an ancestor
        'by_level': collections.defaultdict(int),  # By ancestor level
        'total_computed_items': total_computed_items
    }
    
    # Calculate total query frequency across all leaf nodes (accounting for frequency)
    total_queries = sum(unique_nodes[leaf_id]['frequency'] for leaf_id in leaf_nodes)
    
    for leaf_id in leaf_nodes:
        server_id = query_routing[leaf_id]
        frequency = unique_nodes[leaf_id]['frequency']
        
        if leaf_id == server_id:
            routing_stats['direct'] += frequency
        else:
            routing_stats['via_ancestor'] += frequency
            
            # Calculate level (distance from leaf to ancestor)
            level = 1
            curr = leaf_id
            while parent_map.get(curr) != server_id and curr in parent_map:
                curr = parent_map[curr]
                level += 1
                
            routing_stats['by_level'][level] += frequency
    
    # Calculate the total solution cost
    total_cost = pulp.value(prob.objective)
    
    # Return results
    return {
        'storage_assignment': storage_assignment,
        'query_routing': query_routing,
        'query_costs': query_costs,
        'storage_util': storage_util,
        'routing_stats': routing_stats,
        'total_cost': total_cost,
        'solver_status': pulp.LpStatus[prob.status],
        'k': k
    }


def optimize_rag_storage_and_routing(
    unique_nodes, 
    k=10, 
    gpu_capacity=48, 
    dram_capacity=512, 
    disk_capacity=1024, 
    doc_length=1200,
    performance_bench=None,
    time_limit=3600  # 5 minutes solver time limit
):
    """
    Optimize RAG storage placement and query routing simultaneously.
    This version works with unique_nodes from the improved clustering method.
    
    Args:
        unique_nodes: Dictionary of unique cluster nodes from the clustering function
        k: Number of documents each query needs
        gpu_capacity: GPU capacity in GB
        dram_capacity: DRAM capacity in GB
        disk_capacity: Disk capacity in GB
        doc_length: Average sequence length per document
        performance_bench: Performance benchmarks, defaults if None
        time_limit: Time limit for solver in seconds
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    
    # Set default performance benchmarks if not provided
    if performance_bench is None:
        performance_bench = {
            "NVMe": 6,       # Disk transfer speed in GB/s
            "PCIe": 128,     # DRAM transfer speed in GB/s
            "prefill_throughput": 5000,  # Compute throughput (tokens/sec)
        }
    
    # 1. Prepare data structures
    data = prepare_optimization_data(unique_nodes, k, doc_length)
    data['doc_length'] = doc_length
    data['performance_bench'] = performance_bench
    
    # Convert capacities to bytes
    capacities = {
        'gpu': gpu_capacity * (1024**3),
        'dram': dram_capacity * (1024**3),
        'disk': disk_capacity * (1024**3)
    }
    
    # 2. Build the optimization model
    prob, storage_vars, routing_vars = build_optimization_model(
        unique_nodes,
        data,
        capacities,
        k,
        doc_length,
        performance_bench
    )
    
    # 3. Solve the model
    solver = pulp.GUROBI(timeLimit=time_limit, msg=True, threads=16)
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    print(f"Solver finished in {solve_time:.2f} seconds with status: {pulp.LpStatus[prob.status]}")
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Warning: Problem could not be solved to optimality within time limit.")
    
    # 4. Process results
    results = process_optimization_results(
        unique_nodes, 
        data, 
        storage_vars, 
        routing_vars, 
        prob, 
        k, 
        capacities
    )
    
    # Add solver time to results
    results['solver_time'] = solve_time
    results['performance_bench'] = performance_bench
    
    return results


def print_rag_optimization_results(unique_nodes, results):
    """
    Print comprehensive results of the RAG storage and routing optimization.
    
    Args:
        unique_nodes: Dictionary of unique cluster nodes
        results: Results from optimize_rag_storage_and_routing
    """
    print("\n=== RAG STORAGE AND ROUTING OPTIMIZATION RESULTS ===")
    print(f"Solver Status: {results['solver_status']} in {results['solver_time']:.2f} seconds")
    print(f"Total Cost: {results['total_cost']:.4f}")
    print(f"K value: {results['k']}")
    
    # Print storage utilization
    print("\n--- STORAGE UTILIZATION ---")
    print(f"GPU: {results['storage_util']['gpu_used_gb']:.2f}GB / {results['storage_util']['gpu_capacity_gb']}GB "
          f"({results['storage_util']['gpu_utilization']:.2f}%)")
    print(f"DRAM: {results['storage_util']['dram_used_gb']:.2f}GB / {results['storage_util']['dram_capacity_gb']}GB "
          f"({results['storage_util']['dram_utilization']:.2f}%)")
    print(f"DISK: {results['storage_util']['disk_used_gb']:.2f}GB / {results['storage_util']['disk_capacity_gb']}GB "
          f"({results['storage_util']['disk_utilization']:.2f}%)")
    
    # Print routing statistics
    print("\n--- QUERY ROUTING STATISTICS ---")
    total_freq = sum(info['frequency'] for info in results['query_costs'].values())
    
    direct = results['routing_stats']['direct']
    via_ancestor = results['routing_stats']['via_ancestor']
    total_computed = results['routing_stats']['total_computed_items']
    
    print(f"Direct routing: {direct} queries ({direct/total_freq*100:.1f}%)")
    print(f"Via ancestor: {via_ancestor} queries ({via_ancestor/total_freq*100:.1f}%)")
    print(f"Total additional items computed: {total_computed}")
    
    if via_ancestor > 0:
        print("\nRouting by ancestor level:")
        for level, count in sorted(results['routing_stats']['by_level'].items()):
            print(f"  Level {level}: {count} queries ({count/total_freq*100:.1f}%)")
    
    # Group nodes by storage location
    by_storage = {'gpu': [], 'dram': [], 'disk': [], 'not_stored': []}
    for node_id, storage in results['storage_assignment'].items():
        by_storage[storage].append(node_id)
    
    print("\n--- STORAGE PLACEMENT SUMMARY ---")
    for storage, node_ids in by_storage.items():
        if not node_ids:
            continue
            
        print(f"\n{storage.upper()} NODES ({len(node_ids)}):")
        
        # Sort by frequency and node type
        nodes_info = []
        for node_id in node_ids:
            is_leaf = len(unique_nodes[node_id].get('children', [])) == 0
            nodes_info.append((
                node_id,
                unique_nodes[node_id]['frequency'],
                len(unique_nodes[node_id]['content']),
                "Leaf" if is_leaf else "Internal"
            ))
        
        # Sort by frequency (descending)
        nodes_info.sort(key=lambda x: x[1], reverse=True)
        
        # Display top 10 nodes by frequency
        for i, (node_id, freq, content_size, node_type) in enumerate(nodes_info[:10]):
            print(f"  Node {node_id} ({node_type}): Frequency={freq}, Content Size={content_size}")
        
        if len(nodes_info) > 10:
            print(f"  ... and {len(nodes_info) - 10} more nodes")
        
        # Statistics for this storage tier
        leaf_count = sum(1 for _, _, _, ntype in nodes_info if ntype == "Leaf")
        internal_count = sum(1 for _, _, _, ntype in nodes_info if ntype == "Internal")
        avg_freq = np.mean([freq for _, freq, _, _ in nodes_info])
        avg_size = np.mean([size for _, _, size, _ in nodes_info])
        
        print(f"  Summary: {leaf_count} leaf nodes, {internal_count} internal nodes")
        print(f"  Average frequency: {avg_freq:.1f}, Average content size: {avg_size:.1f}")
    
    # Print query routing details
    print("\n--- DETAILED QUERY ROUTING ---")
    print("Top 10 queries by cost:")
    
    # Get top queries by total cost
    top_queries = sorted(results['query_costs'].items(), key=lambda x: x[1]['total_cost'], reverse=True)[:10]
    
    for leaf_id, cost_info in top_queries:
        server_id = cost_info['server_node']
        storage = cost_info['storage']
        server_size = cost_info['server_content_size']
        add_compute = cost_info['additional_compute']
        freq = cost_info['frequency']
        per_query = cost_info['per_query_cost']
        total = cost_info['total_cost']
        
        if leaf_id == server_id:
            route_str = "Direct"
        else:
            route_str = f"Via ancestor (node {server_id})"
            
        print(f"Query {leaf_id}: Frequency={freq}, Per-query cost={per_query:.4f}, Total cost={total:.4f}")
        print(f"  Routing: {route_str}")
        print(f"  Server: Node {server_id} on {storage.upper()} with {server_size} items")
        print(f"  Additional compute: {add_compute} items")