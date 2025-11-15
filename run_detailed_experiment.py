import subprocess
import csv
import time

# --- Configuration ---
MPI_EXECUTABLE = "mpiexec"
NUM_PROCESSES = 10  # This *MUST* match the number of CPUs you want to test
SCRIPT_TO_RUN = "linear_regime.py"
RESULTS_FILE = "detailed_workload_results.csv"

# --- Define the graph sizes you want to test ---
N_VALUES_TO_TEST = [1000, 5000, 10000, 20000, 40000, 80000, 120000, 160000]

# --- Run the Experiment ---
print("--- Starting Detailed Workload Experiment ---")
print(f"Will write results to {RESULTS_FILE}")

# Create header for the CSV file
csv_headers = ["n", "machine_id", "round", "workload"]

with open(RESULTS_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers) # Write the header

    for n in N_VALUES_TO_TEST:
        print(f"\nRunning experiment for n = {n}...")
        
        command = [
            MPI_EXECUTABLE,
            "-n", str(NUM_PROCESSES),
            "python",
            SCRIPT_TO_RUN,
            "--n", str(n)
        ]
        
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"Finished in {duration:.2f} seconds.")

        # --- Parse the output to find our special result line ---
        found_result = False
        for line in result.stdout.splitlines():
            if "---FINAL_RESULT---" in line:
                # Line: "---FINAL_RESULT---,n=1000,m0_r1=50,m0_r2=20,m1_r1=48,..."
                parts = line.split(',')
                
                n_val = int(parts[1].split('=')[1])
                
                # Parse all other key-value pairs
                for part in parts[2:]:
                    # key = "m0_r1", val = "50"
                    key, val = part.split('=')
                    workload = int(val)
                    
                    # key_parts = ["m0", "r1"]
                    key_parts = key.split('_')
                    machine_id = int(key_parts[0][1:]) # "m0" -> 0
                    round_num = int(key_parts[1][1:])  # "r1" -> 1
                    
                    # Write one row in the CSV for this data point
                    writer.writerow([n_val, machine_id, round_num, workload])
                
                print(f"Saved results for n={n_val}.")
                found_result = True
                break
        
        if not found_result:
            print(f"ERROR: Could not find '---FINAL_RESULT---' in output for n={n}.")

print("\n--- Experiment Complete ---")