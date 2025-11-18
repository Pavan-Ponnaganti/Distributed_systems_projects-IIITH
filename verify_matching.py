import networkx as nx
import argparse
import sys

def generate_graph_data(n, m_edges):
    """
    Re-generates the exact same graph as the main script.
    It's critical that seed=42 is the same.
    """
    print(f"Re-generating original graph (n={n}, m_per_node={m_edges}, seed=42)...")
    G = nx.barabasi_albert_graph(n, m_edges, seed=42)
    # We only need the edge list
    all_edges = list(G.edges())
    print(f"Original graph has {len(all_edges)} edges.")
    return all_edges

def load_matching(matching_path):
    """
    Loads the matching file produced by your algorithm.
    """
    print(f"Loading matching from: {matching_path}")
    matching_edges = set()
    try:
        with open(matching_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                
                u = int(parts[0])
                v = int(parts[1])
                
                # Add both orderings to the set for easy checking
                # (u, v) and (v, u) are the same edge
                if u > v:
                    u, v = v, u
                matching_edges.add((u, v))
                
    except FileNotFoundError:
        print(f"Error: Matching file not found at {matching_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(matching_edges)} edges in the matching file.")
    return matching_edges

def main():
    parser = argparse.ArgumentParser(description="Verify a maximal matching.")
    # Arguments must match the ones used to generate the graph
    parser.add_argument('--n', type=int, required=True, help='Number of nodes (n) used to generate the graph.')
    parser.add_argument('--m', type=int, required=True, help='Edges per node (m) used to generate the graph.')
    parser.add_argument('--matching', type=str, required=True, help="Path to the matching.txt file to verify.")
    args = parser.parse_args()

    # 1. Re-generate the original graph
    original_edges = generate_graph_data(args.n, args.m)
    
    # 2. Load the matching
    matching_edges = load_matching(args.matching)

    # 3. Create a set of all nodes covered by the matching
    matched_nodes = set()
    for u, v in matching_edges:
        matched_nodes.add(u)
        matched_nodes.add(v)

    print(f"\nVerification running... checking {len(original_edges)} original edges.")
    print(f"Total nodes covered by matching: {len(matched_nodes)}")

    # 4. Check every edge in the *original* graph
    found_free_edge = None
    for u, v in original_edges:
        
        # An edge is "free" if it could be added to the matching.
        # This is only possible if *both* its endpoints are free (unmatched).
        if u not in matched_nodes and v not in matched_nodes:
            found_free_edge = (u, v)
            break # Found a failure, no need to check more

    # 5. Report the result
    print("\n--- Verification Result ---")
    if found_free_edge:
        u, v = found_free_edge
        print(f"FAILURE: The matching is NOT maximal.")
        print(f"Found an edge ({u}, {v}) in the original graph where neither")
        print(f"node 'u' nor node 'v' is covered by the matching.")
    else:
        print("SUCCESS: The final matching is maximal.")
        print("All edges in the original graph are covered by the matching.")

if __name__ == "__main__":
    main()