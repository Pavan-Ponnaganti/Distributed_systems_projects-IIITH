import networkx as nx
import random
import math
import time
from mpi4py import MPI
import argparse
from collections import defaultdict
# --- Helper Functions (no changes) ---
def generate_graph_data(n, m_edges):
    """ (This runs only on the master process) """
    print(f"[Master] Generating graph with {n} nodes...")
    G = nx.barabasi_albert_graph(n, m_edges, seed=42)
    all_nodes = list(G.nodes())
    all_edges = list(G.edges())
    degrees = [d for n, d in G.degree()]
    delta = max(degrees) if degrees else 1
    print(f"[Master] Graph generated. Max Degree (Delta) = {delta}")
    return G, all_nodes, all_edges, delta

def create_arbitrary_chunks(all_edges, num_chunks):
    """ (This runs only on the master process) """
    random.shuffle(all_edges)
    chunk_size = len(all_edges) // num_chunks
    chunks = [all_edges[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    remainder = len(all_edges) % num_chunks
    for i in range(remainder):
        chunks[i].append(all_edges[-(i + 1)])
    return chunks

def get_local_residual_edges(local_edges, matching):
    nodes_to_remove = set()
    for u, v in matching:
        nodes_to_remove.add(u)
        nodes_to_remove.add(v)
    
    return [
        (u, v) for u, v in local_edges 
        if u not in nodes_to_remove and v not in nodes_to_remove
    ]

def get_local_max_degree(local_edges, local_nodes_map):
    # We need to compute degrees for *all* nodes, not just local ones,
    # as an edge (u, v) can have 'v' be non-local.
    # A dictionary is the most space-efficient way.
    degrees = defaultdict(int)
    for u, v in local_edges:
        degrees[u] += 1
        degrees[v] += 1
    
    # Find the max degree *only* among nodes this process owns
    local_max = 0
    for node in local_nodes_map:
        local_max = max(local_max, degrees[node])
    return local_max


# --- Main "True Linear" Algorithm ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500, help='Number of nodes (n)')
    parser.add_argument('--m', type=int, default=4, help='Edges per node (for generator)')
    args = parser.parse_args()
    N_NODES = args.n
    M_EDGES_PER_NODE = args.m

    # --- NEW: All processes track their own data ---
    my_workloads_per_round = [] # Tracks parallel workload
    my_edges = []              # Our *local* partition of the *full* residual graph
    Total_Matching = set()     # Only Rank 0 will store the final matching
    
    # --- 1. Master Process (Rank 0) Setup ---
    if rank == 0:
        G, all_nodes, all_edges, _ = generate_graph_data(N_NODES, M_EDGES_PER_NODE)
        
        # --- Create arbitrary chunks for initial scatter ---
        # This simulates the "arbitrarily distributed" starting state
        arbitrary_edge_chunks = create_arbitrary_chunks(all_edges, size)
    else:
        arbitrary_edge_chunks = None
        all_nodes = None # Will receive this
    
    # --- 2. Initial State: Scatter & Broadcast ---
    # All processes receive their initial, arbitrary chunk of edges
    my_edges = comm.scatter(arbitrary_edge_chunks, root=0)
    
    # Broadcast the full list of nodes (needed for vertex_map generation)
    all_nodes = comm.bcast(all_nodes, root=0)

    
    iteration = 1
    while True:
        # --- 0. Calculate Current Delta (Distributed) ---
        # This is a distributed operation.
        
        # A. Create a map of *all* nodes to their *local* degrees
        local_degrees = defaultdict(int)
        for u, v in my_edges:
            local_degrees[u] += 1
            local_degrees[v] += 1

        # B. Gather all local degree maps to root
        all_local_degrees = comm.gather(local_degrees, root=0)
        
        # C. Root computes the *true* global delta and broadcasts it
        if rank == 0:
            global_degrees = defaultdict(int)
            for local_map in all_local_degrees:
                for node, deg in local_map.items():
                    global_degrees[node] += deg
            
            delta_current = max(global_degrees.values()) if global_degrees else 0
            # Also send the full list of currently alive nodes
            alive_nodes = list(global_degrees.keys())
        else:
            delta_current = 0
            alive_nodes = None

        delta_current = comm.bcast(delta_current, root=0)
        alive_nodes = comm.bcast(alive_nodes, root=0) # Broadcast current alive nodes

        if rank == 0:
            print(f"\n[Master] --- Starting Iteration {iteration} (Global Delta = {delta_current}) ---")
            
        if delta_current <= 2:
            break
            
        # --- 1. Run Parallel Algorithm 1 (Phase A) ---
        
        # A. Rank 0 creates NEW random parameters for this round
        if rank == 0:
            k = size
            p = math.pow(delta_current, -0.77)
            q = math.pow(delta_current, -0.91)
            threshold = math.pow(delta_current, 0.92)
            
            # --- THIS IS THE FIX ---
            # A NEW vertex_map and hash_seed are created in *every* round
            vertex_map = {node: random.randint(0, k - 1) for node in alive_nodes}
            hash_seed = random.randint(0, 2**32 - 1)
            # --- END FIX ---
            
            round_params = {'p': p, 'q': q, 'threshold': threshold, 
                            'vertex_map': vertex_map, 'hash_seed': hash_seed}
        else:
            round_params = None
            
        # B. Broadcast the NEW parameters for this round
        round_params = comm.bcast(round_params, root=0)
        p = round_params['p']
        q = round_params['q']
        threshold = round_params['threshold']
        vertex_map = round_params['vertex_map'] # Use the new map
        hash_seed = round_params['hash_seed']   # Use the new seed
        
        # C. Local Sampling (for Algorithm 1)
        # We sample our *current* residual edges
        sampled_local_edges = [edge for edge in my_edges if random.random() <= p]
        
        # D. The "Shuffle" for G^L[Vi] (using the NEW vertex_map)
        send_buckets = [[] for _ in range(size)]
        for u, v in sampled_local_edges:
            # Nodes might not be in the new map if they were 
            # isolated and removed in a previous round's degree check
            if u not in vertex_map or v not in vertex_map:
                continue
                
            owner_u = vertex_map[u]
            owner_v = vertex_map[v]
            if owner_u == owner_v:
                send_buckets[owner_u].append((u, v))
                
        received_buckets = comm.alltoall(send_buckets)
        my_subgraph_edges = [edge for bucket in received_buckets for edge in bucket]
        
        # --- Record Workload ---
        my_round_workload = len(my_subgraph_edges)
        my_workloads_per_round.append(my_round_workload)
        
        # E. Parallel Greedy Matching (Hash-based, using NEW hash_seed)
        local_matching = set()
        matched_nodes = set()
        prioritized_edges = []
        for u, v in my_subgraph_edges:
            edge_tuple = tuple(sorted((u, v)))
            priority = hash((hash_seed, edge_tuple)) # Use the new seed
            prioritized_edges.append((priority, (u, v)))
        prioritized_edges.sort()
        for priority, (u, v) in prioritized_edges:
            if u not in matched_nodes and v not in matched_nodes:
                local_matching.add((u, v))
                matched_nodes.add(u)
                matched_nodes.add(v)
        
        # F. Gather M_parallel to Rank 0
        all_local_matchings = comm.gather(local_matching, root=0)
        
        if rank == 0:
            M_parallel = set()
            for local_set in all_local_matchings:
                M_parallel.update(local_set)
            print(f"[Master] Iteration {iteration}: M_parallel found {len(M_parallel)} edges.")
        else:
            M_parallel = None
            
        # Bcast M_parallel so all processes can update their residual graph
        M_parallel = comm.bcast(M_parallel, root=0)
        
        # ALL processes update their local residual graph
        my_edges = get_local_residual_edges(my_edges, M_parallel)
        
        # --- 2. Centralized Clean-up M' (Phase B) ---
        local_g_prime_edges = [edge for edge in my_edges if random.random() <= q]
        all_g_prime_chunks = comm.gather(local_g_prime_edges, root=0)
        
        if rank == 0:
            g_prime_edges = [edge for chunk in all_g_prime_chunks for edge in chunk]
            G_prime = nx.Graph(g_prime_edges)
            M_prime = nx.maximal_matching(G_prime)
            print(f"[Master] Iteration {iteration}: M_prime found {len(M_prime)} edges.")
        else:
            M_prime = None
            
        M_prime = comm.bcast(M_prime, root=0)
        my_edges = get_local_residual_edges(my_edges, M_prime)
        
        # --- 3. Centralized Clean-up M'' (Phase C) ---
        
        # A. All processes find their *local* straggler edges
        my_degrees = defaultdict(int)
        for u, v in my_edges:
            my_degrees[u] += 1
            my_degrees[v] += 1
            
        my_straggler_nodes = {n for n, d in my_degrees.items() if d >= threshold}
        my_straggler_edges = [
            (u, v) for u, v in my_edges 
            if u in my_straggler_nodes or v in my_straggler_nodes
        ]
            
        # B. Gather all straggler edges to Rank 0
        all_straggler_chunks = comm.gather(my_straggler_edges, root=0)
            
        if rank == 0:
            g_double_prime_edges = [edge for chunk in all_straggler_chunks for edge in chunk]
            G_double_prime = nx.Graph(g_double_prime_edges)
            M_double_prime = nx.maximal_matching(G_double_prime)
            print(f"[Master] Iteration {iteration}: M'' found {len(M_double_prime)} edges.")
        else:
            M_double_prime = None
                
        # C. Bcast M''
        M_double_prime = comm.bcast(M_double_prime, root=0)
        my_edges = get_local_residual_edges(my_edges, M_double_prime)

        # --- 4. Combine and prepare for next iteration ---
        if rank == 0:
            M_round = M_parallel.union(M_prime).union(M_double_prime)
            Total_Matching.update(M_round)
            
        iteration += 1

    # --- Final Centralized Step (The O(n) Bottleneck) ---
    all_final_edges = comm.gather(my_edges, root=0)
    
    if rank == 0:
        final_residual_edges = [edge for chunk in all_final_edges for edge in chunk]
        final_solve_workload = len(final_residual_edges)
        
        G_current = nx.Graph(final_residual_edges)
        M_final = nx.maximal_matching(G_current)
        Total_Matching.update(M_final)
    else:
        final_solve_workload = 0

    # --- Final Report ---
    all_workload_data = comm.gather(my_workloads_per_round, root=0)
    
    if rank == 0:
        print(f"---FINAL_RESULT_START---,n={N_NODES}")
        for machine_id in range(size):
            if all_workload_data[machine_id] is None: continue # Should not happen, but safe
            for round_num in range(len(all_workload_data[machine_id])):
                workload = all_workload_data[machine_id][round_num]
                print(f"---DATA_POINT---,{N_NODES},{machine_id},parallel,{round_num+1},{workload}")
        print(f"---DATA_POINT---,{N_NODES},0,final_solve,1,{final_solve_workload}")
        print("---FINAL_RESULT_END---")

if __name__ == "__main__":
    main()