import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FILE = "detailed_workload_results.csv"

try:
    df = pd.read_csv(RESULTS_FILE)
except FileNotFoundError:
    print(f"Error: Could not find {RESULTS_FILE}")
    print("Please run the 'run_detailed_experiment.py' script first.")
    exit()

if df.empty:
    print("Error: The results file is empty.")
    exit()

# --- Create directories for plots ---
plot_dir_1 = "plots_per_machine"
plot_dir_2 = "plots_peak_workload"
os.makedirs(plot_dir_1, exist_ok=True)
os.makedirs(plot_dir_2, exist_ok=True)

# --- 1. Generate Plot Set 1 (Per-Machine, Per-Round - PARALLEL ONLY) ---
print(f"Generating Plot Set 1 (Parallel Rounds) in '{plot_dir_1}'...")
num_machines = df['machine_id'].max() + 1

# Filter for parallel rounds only
parallel_df = df[df['round_type'] == 'parallel'].copy()

if not parallel_df.empty:
    max_rounds = parallel_df['round_num'].max()
    palette = sns.color_palette("husl", max_rounds)

    for i in range(num_machines):
        plt.figure(figsize=(12, 7))
        machine_df = parallel_df[parallel_df['machine_id'] == i]
        
        if machine_df.empty:
            plt.close() # Close empty plot
            continue
            
        ax = sns.lineplot(
            data=machine_df,
            x='n',
            y='workload',
            hue='round_num', # Use round_num now
            palette=palette,
            marker='o',
            legend='full'
        )
        
        ax.set_title(f'Parallel Workload vs. Graph Size for Machine {i}')
        ax.set_xlabel('Total Vertices in Graph (n)')
        ax.set_ylabel('Edges Processed by Machine (for Algorithm 1)')
        ax.legend(title='Round')
        plt.grid(True)
        plt.tight_layout()
        
        plot_file = os.path.join(plot_dir_1, f"machine_{i}_parallel_workload.png")
        plt.savefig(plot_file)
        plt.close()

print("Plot Set 1 Complete.")

# --- 2. Generate Plot Set 2 (Peak Workload Per Machine - INCLUDES FINAL SOLVE) ---
print(f"Generating Plot Set 2 (Peak Workload) in '{plot_dir_2}'...")

# This plot includes *all* workload types
# We find the peak workload for each machine at each 'n'
peak_workloads = df.groupby(['n', 'machine_id'])['workload'].max().reset_index()

plt.figure(figsize=(12, 7))
peak_palette = sns.color_palette("tab10", num_machines)

ax_peak = sns.lineplot(
    data=peak_workloads,
    x='n',
    y='workload',
    hue='machine_id',
    palette=peak_palette,
    marker='o',
    legend='full'
)

ax_peak.set_title('Peak Per-Round Workload (All Types) vs. Graph Size')
ax_peak.set_xlabel('Total Vertices in Graph (n)')
ax_peak.set_ylabel('Peak Edges Processed in Any Round or Final Solve')
ax_peak.legend(title='Machine ID', bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()

plot_file_peak = os.path.join(plot_dir_2, "peak_workload_all_machines.png")
plt.savefig(plot_file_peak)
plt.close()

print("Plot Set 2 Complete.")
print(f"\nAll plots saved to '{plot_dir_1}' and '{plot_dir_2}'.")
