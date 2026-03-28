# Neural-Net-work-cache-in-Deep-Learning-using-GEM5

This project studies how cache organization affects a simple neural-network style workload in `gem5`. It builds a small food-image classifier for the Pizza / Steak / Sushi dataset, converts the learned computation into matrix form, runs a RISC-V matrix-multiplication workload in `gem5`, and compares cache behavior across direct-mapped, set-associative, and fully associative configurations.

## What This Project Does

- Downloads and prepares the Pizza / Steak / Sushi image dataset
- Resizes images to `32x32` RGB and converts them into matrix inputs
- Trains a deterministic linear classifier in Python
- Exports matrices and weights for a RISC-V workload
- Compiles a static RISC-V binary from `matmul_food.cpp`
- Runs three `gem5` simulations with different cache associativities
- Parses `gem5` stats and generates comparison plots
- Optionally launches a live PyQt5 dashboard during execution

## Pipeline Overview

The main workflow is driven by [`run_all.sh`](/home/wincode/nn_cache_project/run_all.sh):

1. Validate the Python virtual environment and required packages
2. Download the Pizza / Steak / Sushi dataset
3. Extract image matrices and train the classifier
4. Compile the RISC-V matrix-multiplication program
5. Run `gem5` for:
   `Direct-Mapped` (`assoc=1`)
   `4-way Set-Associative` (`assoc=4`)
   `Fully Associative` (`assoc=512`)
6. Parse stats and generate the final cache-analysis plot

## Main Files

- [`run_all.sh`](/home/wincode/nn_cache_project/run_all.sh): end-to-end automation script
- [`download_dataset.py`](/home/wincode/nn_cache_project/download_dataset.py): downloads and unpacks the dataset
- [`extract_matrices.py`](/home/wincode/nn_cache_project/extract_matrices.py): builds matrices and trains the classifier
- [`model_utils.py`](/home/wincode/nn_cache_project/model_utils.py): preprocessing, training, prediction, and evaluation helpers
- [`matmul_food.cpp`](/home/wincode/nn_cache_project/matmul_food.cpp): RISC-V matrix-multiplication workload used by `gem5`
- [`gem5_food_config.py`](/home/wincode/nn_cache_project/gem5_food_config.py): `gem5` cache and memory configuration
- [`parse_and_plot.py`](/home/wincode/nn_cache_project/parse_and_plot.py): parses stats and generates plots
- [`stats_utils.py`](/home/wincode/nn_cache_project/stats_utils.py): shared helpers for reading `gem5` stats
- [`live_monitor.py`](/home/wincode/nn_cache_project/live_monitor.py): PyQt5 live dashboard

## Requirements

### System tools

- `gem5` built for RISC-V, for example at `~/gem5/build/RISCV/gem5.opt`
- `riscv64-linux-gnu-g++`
- Bash

### Python packages

The pipeline expects a project virtual environment at `venv/` with:

- `numpy`
- `matplotlib`
- `pillow`
- `requests`
- `PyQt5`

## Setup

Create and populate a virtual environment:

```bash
python3 -m venv venv
./venv/bin/pip install numpy matplotlib pillow requests PyQt5
```

Build `gem5` if needed:

```bash
cd ~/gem5
scons build/RISCV/gem5.opt -j2
```

Install the RISC-V cross compiler if it is not already available:

```bash
riscv64-linux-gnu-g++ --version
```

## Run The Full Workflow

From the project root:

```bash
chmod +x run_all.sh
./run_all.sh
```

If `gem5` is installed somewhere else, override the path:

```bash
GEM5=/path/to/gem5.opt ./run_all.sh
```

To disable the live GUI monitor:

```bash
ENABLE_LIVE_MONITOR=0 ./run_all.sh
```

## Outputs

After a successful run, the main outputs are:

- [`results/model_metrics.json`](/home/wincode/nn_cache_project/results/model_metrics.json): classifier metrics
- [`results/direct/stats.txt`](/home/wincode/nn_cache_project/results/direct/stats.txt): direct-mapped cache stats
- [`results/set4way/stats.txt`](/home/wincode/nn_cache_project/results/set4way/stats.txt): 4-way set-associative cache stats
- [`results/fullassoc/stats.txt`](/home/wincode/nn_cache_project/results/fullassoc/stats.txt): fully associative cache stats
- [`plots/cache_analysis.png`](/home/wincode/nn_cache_project/plots/cache_analysis.png): final comparison chart
- [`results/live_status.json`](/home/wincode/nn_cache_project/results/live_status.json): status file for the live monitor

## Current Experiment Structure

- Dataset classes: `pizza`, `steak`, `sushi`
- Image resolution: `32x32`
- Model: deterministic linear classifier
- GEM5 workload: matrix multiply `X_sub @ W`
- CPU model: `TimingSimpleCPU`
- Cache study focus: L1 data-cache associativity

## Notes

- The repository currently includes generated outputs such as dataset files, plots, matrices, and `gem5` result folders.
- The local virtual environment and cache folders are excluded from Git with [`.gitignore`](/home/wincode/nn_cache_project/.gitignore).
- The live dashboard depends on a reachable GUI display; without one, the pipeline still runs in non-GUI mode.
