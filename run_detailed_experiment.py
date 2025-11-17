import subprocess
import csv
import time

# --- Configuration ---
MPI_EXECUTABLE = "mpiexec"
NUM_PROCESSES = 10  # This *MUST* match the number of CPUs you want to test
SCRIPT_TO_RUN = "linear_regime.py"
RESULTS_FILE = "detailed_workload_results.csv"

# --- Define the graph sizes you want to test ---
N_VALUES_TO_TEST = [1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]

# --- Run the Experiment ---
print("--- Starting Detailed Workload Experiment ---")
print(f"Will write results to {RESULTS_FILE}")

# Create header for the CSV file
csv_headers = ["n", "machine_id", "round_type", "round_num", "workload"]

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

        # --- Parse the output to find our data lines ---
        found_data = False
        in_results_block = False
        
        for line in result.stdout.splitlines():
            if "---FINAL_RESULT_START---" in line:
                in_results_block = True
                continue
            if "---FINAL_RESULT_END---" in line:
                in_results_block = False
                print(f"Saved results for n={n}.")
                break
                
            if in_results_block and "---DATA_POINT---" in line:
                # Line: "---DATA_POINT---,1000,0,parallel,1,50"
                # Line: "---DATA_POINT---,1000,0,final_solve,1,10"
                parts = line.split(',')
                n_val = int(parts[1])
                machine_id = int(parts[2])
                round_type = parts[3]
                round_num = int(parts[4])
                workload = int(parts[5])
                
                # Write the row in the CSV
                writer.writerow([n_val, machine_id, round_type, round_num, workload])
                found_data = True
        
        if not found_data:
            print(f"ERROR: Could not find '---DATA_POINT---' lines in output for n={n}.")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)

print("\n--- Experiment Complete ---")