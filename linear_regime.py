import networkx as nx
import random
import math
import time
from mpi4py import MPI
import argparse

# --- Helper Functions (no changes) ---
def generate_graph_data(n, m_edges):
    print(f"[Master] Generating graph with {n} nodes...")
    G = nx.barabasi_albert_graph(n, m_edges, seed=42)
    all_nodes = list(G.nodes())
    all_edges = list(G.edges())
    degrees = [d for n, d in G.degree()]
    delta = max(degrees) if degrees else 1
    print(f"[Master] Graph generated. Max Degree (Delta) = {delta}")
    return G, all_nodes, all_edges, delta

def create_arbitrary_chunks(all_edges, num_chunks):
    random.shuffle(all_edges)
    chunk_size = len(all_edges) // num_chunks
    chunks = [all_edges[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    remainder = len(all_edges) % num_chunks
    for i in range(remainder):
        chunks[i].append(all_edges[-(i + 1)])
    return chunks

# --- Algorithm 1 Function (MODIFIED to return workload) ---
def run_parallel_algorithm_1_iteration(comm, all_nodes, all_edges, delta):
    """
    Runs one full parallel iteration of Algorithm 1.
    All processes must enter this function.
    Returns:
    - (Rank 0): (M_Algorithm1, my_workload)
    - (Others): (None, my_workload)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        k = size
        p = math.pow(delta, -0.77)
        vertex_map = {node: random.randint(0, k - 1) for node in all_nodes}
        global_edge_order = all_edges[:]
        random.shuffle(global_edge_order)
        arbitrary_edge_chunks = create_arbitrary_chunks(all_edges, size)
        params_to_bcast = {
            'p': p,
            'vertex_map': vertex_map,
            'global_edge_order': global_edge_order
        }
    else:
        arbitrary_edge_chunks = None
        params_to_bcast = None

    local_edges = comm.scatter(arbitrary_edge_chunks, root=0)
    params = comm.bcast(params_to_bcast, root=0)
    p = params['p']
    vertex_map = params['vertex_map']
    global_edge_order = params['global_edge_order']
    sampled_local_edges = [edge for edge in local_edges if random.random() <= p]

    send_buckets = [[] for _ in range(size)]
    for u, v in sampled_local_edges:
        if u not in vertex_map or v not in vertex_map:
            continue
        owner_u = vertex_map[u]
        owner_v = vertex_map[v]
        if owner_u == owner_v:
            send_buckets[owner_u].append((u, v))
    
    received_buckets = comm.alltoall(send_buckets)
    my_subgraph_edges = [edge for bucket in received_buckets for edge in bucket]
    
    # --- THIS IS THE METRIC ---
    my_workload = len(my_subgraph_edges) # Measured by ALL processes

    local_matching = set()
    matched_nodes = set()
    subgraph_edge_set = set(my_subgraph_edges)
    for u, v in global_edge_order:
        if (u, v) in subgraph_edge_set:
            if u not in matched_nodes and v not in matched_nodes:
                local_matching.add((u, v))
                matched_nodes.add(u)
                matched_nodes.add(v)
    
    all_local_matchings = comm.gather(local_matching, root=0)
    
    if rank == 0:
        M_Algorithm1 = set()
        for local_set in all_local_matchings:
            M_Algorithm1.update(local_set)
        return M_Algorithm1, my_workload
    else:
        return None, my_workload # Others return their workload

# --- Helper for residual graph (no changes) ---
def get_residual_graph(G, matching):
    nodes_to_remove = set()
    for u, v in matching:
        nodes_to_remove.add(u)
        nodes_to_remove.add(v)
    
    G_res = G.copy()
    G_res.remove_nodes_from(nodes_to_remove)
    
    new_delta = 0
    if G_res.nodes():
        degrees = [d for n, d in G_res.degree()]
        if degrees:
            new_delta = max(degrees)
            
    return G_res, new_delta

# --- Main "Master Loop" (Modified for detailed workload experiment) ---
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

    # --- NEW: All processes track their own workload per round ---
    my_workloads_per_round = []

    # --- Variables for the main loop ---
    G_current = None
    all_nodes = None
    all_edges = None
    delta_current = 0
    Total_Matching = set()
    
    if rank == 0:
        G_current, all_nodes, all_edges, delta_current = generate_graph_data(N_NODES, M_EDGES_PER_NODE)
    
    iteration = 1
    while True:
        # All processes get the current delta to decide if loop continues
        delta_current = comm.bcast(delta_current, root=0)
        
        if delta_current <= 2:
            break
            
        if rank == 0:
            print(f"[Master] --- Iteration {iteration} (Delta = {delta_current}) ---")
            
        # --- 1. Call Algorithm 1 (Parallel) ---
        # ALL processes MUST enter this function
        # M_parallel is only set on Rank 0
        # my_round_workload is set on ALL ranks
        M_parallel, my_round_workload = run_parallel_algorithm_1_iteration(comm, all_nodes, all_edges, delta_current)
        
        # --- NEW: All processes record their workload for this round ---
        my_workloads_per_round.append(my_round_workload)
        
        # --- The rest of the loop is CENTRALIZED (Rank 0 only) ---
        if rank == 0:
            G_res_1, _ = get_residual_graph(G_current, M_parallel)
            
            q = math.pow(delta_current, -0.91)
            G_prime_edges = [e for e in G_res_1.edges() if random.random() <= q]
            G_prime = nx.Graph(G_prime_edges)
            M_prime = nx.maximal_matching(G_prime)
            
            G_res_2, _ = get_residual_graph(G_res_1, M_prime)
            
            threshold = math.pow(delta_current, 0.92)
            nodes_to_check = list(G_res_2.nodes())
            U = [n for n in nodes_to_check if G_res_2.degree(n) >= threshold]
            
            G_double_prime = G_res_2.subgraph(U)
            M_double_prime = nx.maximal_matching(G_double_prime)
            
            M_round = M_parallel.union(M_prime).union(M_double_prime)
            Total_Matching.update(M_round)
            
            # --- Master prepares graph data for the *next* iteration ---
            G_current, delta_current = get_residual_graph(G_res_2, M_double_prime)
            all_nodes = list(G_current.nodes())
            all_edges = list(G_current.edges())
            iteration += 1
        
        # --- NEW: Master must bcast new graph info to workers ---
        # (This is needed so workers have the right all_edges for the *next* round)
        if rank != 0:
             # Workers need placeholders for the bcast
             all_nodes = None
             all_edges = None
        all_nodes = comm.bcast(all_nodes, root=0)
        all_edges = comm.bcast(all_edges, root=0)


    # --- Final Centralized Step (The O(n) Bottleneck) ---
    if rank == 0:
        M_final = nx.maximal_matching(G_current)
        Total_Matching.update(M_final)

    # --- NEW: Final Report ---
    # All processes send their workload history to the master
    all_workload_data = comm.gather(my_workloads_per_round, root=0)
    
    if rank == 0:
        # Master now has a list of lists:
        # e.g., [[m0_r1, m0_r2], [m1_r1, m1_r2], ...]
        
        # This print line is specifically formatted for the
        # orchestrator script to read.
        result_str = f"---FINAL_RESULT---,n={N_NODES}"
        for machine_id in range(size):
            for round_num in range(len(all_workload_data[machine_id])):
                workload = all_workload_data[machine_id][round_num]
                result_str += f",m{machine_id}_r{round_num+1}={workload}"
        print(result_str)

if __name__ == "__main__":
    main()