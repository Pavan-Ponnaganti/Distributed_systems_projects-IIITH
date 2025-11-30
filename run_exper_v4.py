import subprocess
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np
import csv

# ==============================================================================
# CLUSTER CONFIGURATION
# ==============================================================================
MPI_SCRIPT = "linear_mpcv4.py" # Uses the merged file above
MPI_CMD = "mpiexec" 
TIMEOUT_SEC = 1200 # 20 mins for heavy checking

# EXPERIMENT RANGES (As requested)
# 1. Scale N (Linearity)
LIN_N_RANGE = list(range(10000, 160001, 20000))
LIN_CONST_M = 150
LIN_CONST_K = 48

# 2. Scale M (Robustness)
ROB_M_RANGE = list(range(20, 201, 20))
ROB_CONST_N = 100000
ROB_CONST_K = 48

# 3. Scale K (Strong Scaling)
STR_K_RANGE = list(range(6, 97, 6))
STR_CONST_N = 100000
STR_CONST_M = 150

# 4. Convergence (Single Heavy Run)
CONV_N = 100000
CONV_M = 150
CONV_K = 48

# ==============================================================================
# EXECUTION ENGINE
# ==============================================================================
def run_job(n, m, k, tag=""):
    print(f"Running {tag}: N={n} M={m} K={k}...", end=" ", flush=True)
    
    # We ENABLE --check for every run as requested
    cmd = [
        MPI_CMD, "-n", str(k),
        "--oversubscribe", 
        sys.executable, MPI_SCRIPT,
        "--n", str(n), "--m", str(m), "--type", "barabasi",
        "--check" 
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=TIMEOUT_SEC)
        
        # Check Validity in Output
        validity = "UNKNOWN"
        if "STRICT_CHECK_PASSED" in res.stdout:
            validity = "PASS"
        elif "STRICT_CHECK_FAILED" in res.stdout:
            validity = "FAIL"
            
        print(f"Done ({validity}).")
        return parse_logs(res.stdout, validity)
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"CRASH. ({e.stderr[:50].strip()})")
        return None

def parse_logs(output, validity):
    data = {
        "summary": {},
        "phases": [],
        "validity": validity
    }
    
    bench = re.search(r"\[BENCHMARK\] time=([0-9\.]+) rounds=([0-9]+) max_msg=([0-9]+)", output)
    if bench:
        data["summary"]["time"] = float(bench.group(1))
        data["summary"]["rounds"] = int(bench.group(2))
        data["summary"]["max_msg"] = int(bench.group(3))
    else:
        return None

    lines = output.splitlines()
    for line in lines:
        if "[PHASE]" in line:
            p = {}
            p["iter"] = int(re.search(r"iter=(\d+)", line).group(1))
            p["delta"] = int(re.search(r"delta=(\d+)", line).group(1))
            p["edges"] = int(re.search(r"active_edges=(\d+)", line).group(1))
            p["cum_rounds"] = int(re.search(r"cum_rounds=(\d+)", line).group(1))
            data["phases"].append(p)
    return data

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def add_assumptions(ax, text):
    ax.text(0.95, 0.05, text, verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='black', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

def plot_linearity(results):
    ns = [r["n"] for r in results]
    c_factors = [r["data"]["summary"]["max_msg"] / r["n"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ns, c_factors, 'bo-', linewidth=2, markersize=8)
    plt.title("2.1 Load Factor Stability (Linearity Proof)")
    plt.xlabel("Input Size (N)")
    plt.ylabel("Linear Factor C (MaxLoad / N)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    assumptions = (f"Assumptions:\nConst M={LIN_CONST_M}\nConst K={LIN_CONST_K}")
    add_assumptions(plt.gca(), assumptions)
    plt.savefig("cluster_plot_2_1_linearity.png")

def plot_robustness(results):
    ms = [r["m"] for r in results]
    rounds = [r["data"]["summary"]["rounds"] for r in results]
    times = [r["data"]["summary"]["time"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Density (Edges per Node M)')
    ax1.set_ylabel('Total Rounds', color='tab:blue')
    ax1.plot(ms, rounds, color='tab:blue', marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Time (s)', color='tab:red') 
    ax2.plot(ms, times, color='tab:red', marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title("Robustness Check: Impact of Graph Density")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    assumptions = (f"Assumptions:\nConst N={ROB_CONST_N}\nConst K={ROB_CONST_K}")
    add_assumptions(ax1, assumptions)
    plt.savefig("cluster_plot_robustness_density.png")

def plot_strong_scaling(results):
    ks = [r["k"] for r in results]
    times = [r["data"]["summary"]["time"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, times, 'go-', linewidth=2, markersize=8)
    plt.title("2.2 Strong Scaling (Time vs Machines)")
    plt.xlabel("MPI Ranks (K)")
    plt.ylabel("Execution Time (s)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    assumptions = (f"Assumptions:\nConst N={STR_CONST_N}\nConst M={STR_CONST_M}")
    add_assumptions(plt.gca(), assumptions)
    plt.savefig("cluster_plot_2_2_strong_scaling.png")

def plot_convergence(data):
    phases = data["phases"]
    x = [p["iter"] for p in phases]
    deltas = [p["delta"] for p in phases]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, deltas, 'mo-', linewidth=2, markersize=8)
    plt.yscale('log')
    plt.title("1.1 Degree Decay (Double Exponential)")
    plt.xlabel("Phase")
    plt.ylabel("Global Max Degree (Log Scale)")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    
    assumptions = (f"Assumptions:\nN={CONV_N}, M={CONV_M}\nK={CONV_K}")
    add_assumptions(plt.gca(), assumptions)
    plt.savefig("cluster_plot_1_1_degree_decay.png")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(MPI_SCRIPT):
        print(f"Error: {MPI_SCRIPT} not found.")
        sys.exit(1)
        
    print("=== STARTING CLUSTER EXPERIMENTS (WITH VALIDITY CHECKS) ===")
    
    # 1. CONVERGENCE
    print("\n[1/4] Running Convergence Deep Dive...")
    conv_data = run_job(CONV_N, CONV_M, CONV_K, "Convergence")
    if conv_data: plot_convergence(conv_data)
        
    # 2. LINEARITY
    print("\n[2/4] Running Linearity Check (Scale N)...")
    lin_results = []
    for n in LIN_N_RANGE:
        d = run_job(n, LIN_CONST_M, LIN_CONST_K, f"Lin N={n}")
        if d: lin_results.append({"n": n, "data": d})
    if lin_results: plot_linearity(lin_results)
        
    # 3. ROBUSTNESS
    print("\n[3/4] Running Robustness Check (Scale M)...")
    rob_results = []
    for m in ROB_M_RANGE:
        d = run_job(ROB_CONST_N, m, ROB_CONST_K, f"Rob M={m}")
        if d: rob_results.append({"m": m, "data": d})
    if rob_results: plot_robustness(rob_results)
        
    # 4. STRONG SCALING
    print("\n[4/4] Running Strong Scaling (Scale K)...")
    str_results = []
    for k in STR_K_RANGE:
        d = run_job(STR_CONST_N, STR_CONST_M, k, f"Str K={k}")
        if d: str_results.append({"k": k, "data": d})
    if str_results: plot_strong_scaling(str_results)
        
    print("\n=== ALL CLUSTER PLOTS GENERATED ===")

if __name__ == "__main__":
    main()
