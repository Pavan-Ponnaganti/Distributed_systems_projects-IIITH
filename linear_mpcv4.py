import networkx as nx
import random
import math
import struct
import hashlib
from mpi4py import MPI
import argparse
from collections import defaultdict
import sys
import os

# =============================================================================
# GLOBAL METRICS TRACKER
# =============================================================================
METRICS = {
    "rounds": 0,
    "local_msg_volume": 0 
}

def track(comm_type, volume=0):
    if comm_type == "round_end":
        METRICS["rounds"] += 1
    else:
        METRICS["local_msg_volume"] += volume

def get_edge_priority(u, v, seed):
    u, v = min(u, v), max(u, v)
    packed = struct.pack('>QQQ', u, v, seed)
    hash_bytes = hashlib.sha256(packed).digest()
    hash_int = int.from_bytes(hash_bytes[:8], 'big')
    return hash_int / (2**64 - 1)

def get_vertex_owner(node, size, iteration):
    packed = struct.pack('>QQ', node, iteration)
    hash_int = int.from_bytes(hashlib.sha256(packed).digest()[:8], 'big')
    return hash_int % size

# =============================================================================
# GRAPH GENERATION
# =============================================================================
def load_or_generate_graph(args):
    n = args.n
    m_edges = args.m
    graph_type = args.type
    
    if graph_type == 'barabasi':
        if m_edges >= n: m_edges = n - 1
        G = nx.barabasi_albert_graph(n, m_edges, seed=42)
    elif graph_type == 'star': G = nx.star_graph(n - 1)
    elif graph_type == 'dense': G = nx.erdos_renyi_graph(n, 0.1, seed=42)
    elif graph_type == 'sparse': G = nx.random_tree(n, seed=42)
    else: sys.exit(1)

    all_edges = list(G.edges())
    degrees = [d for n, d in G.degree()]
    delta = max(degrees) if degrees else 1
    return G, all_edges, delta, args.n

def get_local_residual_edges(local_edges, matching):
    if not matching: return local_edges
    nodes_to_remove = set()
    for u, v in matching:
        nodes_to_remove.add(u)
        nodes_to_remove.add(v)
    return [(u, v) for u, v in local_edges if u not in nodes_to_remove and v not in nodes_to_remove]

# =============================================================================
# STRICT VALIDATION LOGIC
# =============================================================================
def perform_strict_check(G_original, matching_edges):
    """Verifies Independence and Maximality in O(E)"""
    print("[Master] Performing STRICT Validation...", flush=True)
    matched_vertices = set()
    
    # 1. Check Independence
    for u, v in matching_edges:
        if u in matched_vertices or v in matched_vertices:
            print(f"[FAIL] Independence Violated at edge ({u}, {v})")
            return False
        matched_vertices.add(u)
        matched_vertices.add(v)
        
    # 2. Check Maximality
    for u, v in G_original.edges():
        if u not in matched_vertices and v not in matched_vertices:
            print(f"[FAIL] Maximality Violated. Edge ({u}, {v}) is not covered.")
            return False
            
    print("[PASS] Strict Validation Successful.")
    return True

# =============================================================================
# MAIN
# =============================================================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--type', type=str, default='barabasi')
    parser.add_argument('--check', action='store_true') # Validation flag
    args = parser.parse_args()
    
    Total_Matching = set()
    N_NODES = args.n 
    G_initial = None # Kept only if check is True
    
    if rank == 0:
        G_initial, all_edges, initial_delta, N_NODES = load_or_generate_graph(args)
        random.seed(42)
        random.shuffle(all_edges)
        chunks = [all_edges[i*(len(all_edges)//size): (i+1)*(len(all_edges)//size)] for i in range(size)]
        for i in range(len(all_edges)%size): chunks[i].append(all_edges[-(i+1)])
        arbitrary_edge_chunks = chunks
        
        # Free memory if we don't need to validate later
        if not args.check:
            del G_initial
            G_initial = None
    else:
        arbitrary_edge_chunks = None
        
    N_NODES = comm.bcast(N_NODES, root=0)
    my_edges = comm.scatter(arbitrary_edge_chunks, root=0)
    
    comm.Barrier()
    start_time = MPI.Wtime()
    
    iteration = 1
    while True:
        phase_seed = iteration * 1000 
        track("round_end")
        
        local_m = len(my_edges)
        local_degrees = defaultdict(int)
        for u, v in my_edges:
            local_degrees[u]+=1; local_degrees[v]+=1
        
        track("gather", len(local_degrees)) 
        all_local_degrees = comm.gather(local_degrees, root=0)
        track("allreduce", 1) 
        global_m = comm.allreduce(local_m, op=MPI.SUM)
        
        global_delta = 0
        if rank == 0:
            deg_map = defaultdict(int)
            for l_map in all_local_degrees:
                for n, c in l_map.items(): deg_map[n]+=c
            if deg_map: global_delta = max(deg_map.values())
            
            # --- PHASE LOGGING ---
            current_rounds = METRICS["rounds"]
            print(f"[PHASE] iter={iteration} delta={global_delta} active_edges={global_m} cum_rounds={current_rounds}")
            loads = comm.gather(local_m, root=0)
        else:
            comm.gather(local_m, root=0)

        if rank == 0:
            load_str = ",".join(map(str, loads))
            print(f"[LOAD] iter={iteration} loads={load_str}")

        track("bcast", 1) 
        global_delta = comm.bcast(global_delta, root=0)
        
        # Safety & Stop
        if global_delta > size ** 8.33: comm.Abort(1)
        if global_m < 10 * N_NODES: break
            
        # --- Phase 1 ---
        p = global_delta ** -0.77
        send = [[] for _ in range(size)]
        for u, v in my_edges:
            if get_edge_priority(u, v, phase_seed+1) <= p:
                if get_vertex_owner(u, size, iteration) == get_vertex_owner(v, size, iteration):
                    send[get_vertex_owner(u, size, iteration)].append((u, v))
        track("alltoall", sum(len(b) for b in send))
        recv = comm.alltoall(send)
        induced = [e for b in recv for e in b]
        
        local_match = set()
        matched = set()
        prio = [(get_edge_priority(u,v,phase_seed+2), u, v) for u, v in induced]
        prio.sort()
        for _, u, v in prio:
            if u not in matched and v not in matched:
                local_match.add((u,v)); matched.add(u); matched.add(v)
        
        track("gather", len(local_match))
        gathered = comm.gather(local_match, root=0)
        M_par = set()
        if rank == 0: 
            for s in gathered: M_par.update(s)
        track("bcast", len(M_par) if rank==0 else 0)
        M_par = comm.bcast(M_par, root=0)
        my_edges = get_local_residual_edges(my_edges, M_par)
        
        # --- Phase 2 ---
        q = global_delta ** -0.91
        samp = [e for e in my_edges if get_edge_priority(e[0], e[1], phase_seed+3) <= q]
        track("allreduce", 1)
        tot = comm.allreduce(len(samp), op=MPI.SUM)
        M_prime = set()
        if tot < 2 * N_NODES:
            track("gather", len(samp))
            all_samp = comm.gather(samp, root=0)
            if rank == 0:
                flat = [e for c in all_samp for e in c]
                pr = [(get_edge_priority(u,v,phase_seed+4),u,v) for u,v in flat]
                pr.sort()
                m_vs = set()
                for _, u, v in pr:
                    if u not in m_vs and v not in m_vs:
                        M_prime.add((u,v)); m_vs.add(u); m_vs.add(v)
        track("bcast", len(M_prime) if rank==0 else 0)
        M_prime = comm.bcast(M_prime, root=0)
        my_edges = get_local_residual_edges(my_edges, M_prime)
        
        # --- Phase 3 ---
        ld = defaultdict(int)
        for u,v in my_edges: ld[u]+=1; ld[v]+=1
        track("gather", len(ld))
        ald = comm.gather(ld, root=0)
        U = set()
        if rank == 0:
            gd = defaultdict(int)
            for m in ald: 
                for n, c in m.items(): gd[n]+=c
            thresh = global_delta ** 0.92
            for n, c in gd.items(): 
                if c>=thresh: U.add(n)
        track("bcast", len(U) if rank==0 else 0)
        U = comm.bcast(U, root=0)
        
        strag = [e for e in my_edges if e[0] in U or e[1] in U]
        track("allreduce", 1)
        tots = comm.allreduce(len(strag), op=MPI.SUM)
        M_dbl = set()
        if tots < 2 * N_NODES:
            track("gather", len(strag))
            alls = comm.gather(strag, root=0)
            if rank == 0:
                flat = [e for c in alls for e in c]
                if flat: M_dbl = set(nx.maximal_matching(nx.Graph(flat)))
        track("bcast", len(M_dbl) if rank==0 else 0)
        M_dbl = comm.bcast(M_dbl, root=0)
        my_edges = get_local_residual_edges(my_edges, M_dbl)
        
        if rank == 0: Total_Matching.update(M_par.union(M_prime).union(M_dbl))
        iteration += 1

    # Final Solve
    track("gather", len(my_edges))
    final = comm.gather(my_edges, root=0)
    if rank == 0:
        flat = [e for c in final for e in c]
        if flat:
            M_f = nx.maximal_matching(nx.Graph(flat))
            Total_Matching.update(M_f)
            
        # --- LOGGING ---
        print(f"Total Matching Size: {len(Total_Matching)}")
        if args.check:
            if G_initial:
                valid = perform_strict_check(G_initial, Total_Matching)
                if valid: print("STRICT_CHECK_PASSED")
                else: print("STRICT_CHECK_FAILED")
            else:
                print("Strict check skipped (G_initial missing).")
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    g_msg = comm.reduce(METRICS["local_msg_volume"], op=MPI.MAX, root=0)
    g_rds = comm.reduce(METRICS["rounds"], op=MPI.MAX, root=0)
    
    if rank == 0:
        print(f"[BENCHMARK] time={end_time - start_time:.4f} rounds={g_rds} max_msg={g_msg}")

if __name__ == "__main__":
    main()
