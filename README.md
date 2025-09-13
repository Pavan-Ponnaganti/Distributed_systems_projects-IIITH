# 🧠 Distributed KNN with MapReduce and MPI

This project implements a scalable **K-Nearest Neighbors (KNN)** algorithm using **MapReduce-style parallelism** with **MPI**. It is designed for high-performance clusters and supports automated scaling experiments via SLURM.

---

## 🚀 Overview

The pipeline follows a classic MapReduce structure:

1. **Setup Phase**: Computes spatial grid parameters and bounding box.
2. **Map Phase**: Each MPI rank maps its local data to grid cells.
3. **Shuffle Phase**: Redistributes data across ranks based on cell ownership.
4. **Reduce Phase**: Each rank computes KNN for queries in its assigned cells.
5. **Gather Phase**: Results are collected and merged at rank 0.
6. **Output Phase**: Final results and timing metrics are saved.

---

## 📁 File Structure

├── knn_main.py # Main driver script 
├── map.py # Setup and mapper functions 
├── shuffle.py # Shuffler logic using MPI alltoall
├── reduce.py # Reducer and result gathering 
├── output.py # Final output and timing logs 
├── run_knn_mpr.sh # SLURM job script for scaling experiments 
├── auto_run_knn.sh # Interactive launcher with input prompts 
├── data/ # Contains points.csv and queries.csv 
├── output/ # Stores results and timings


---

## ⚙️ Requirements

- Python 3.8+
- `mpi4py`
- SLURM-compatible cluster
- Shared filesystem (for distributed reads)

---

## 📦 Installation

```bash
conda create -n mpr_env python=3.10 mpi4py
conda activate mpr_env


## ⚙️ Execution

Start Experiment
 bash auto_run_knn.sh

check output folder for outpunt or errors

