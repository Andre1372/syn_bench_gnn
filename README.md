# Synthetic Benchmark for Graph Neural Networks

## Table of Contents
- [Synthetic Benchmark for Graph Neural Networks](#synthetic-benchmark-for-graph-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Standard Command for a Full Benchmark Run](#standard-command-for-a-full-benchmark-run)
    - [Command Line Arguments](#command-line-arguments)
      - [Dataset \& Generation Settings](#dataset--generation-settings)
      - [GNN Training \& Evaluation Settings](#gnn-training--evaluation-settings)
      - [Execution \& Infrastructure Settings](#execution--infrastructure-settings)
  - [Results Analysis and Visualization](#results-analysis-and-visualization)
  - [Auxiliary Scripts \& Utilities](#auxiliary-scripts--utilities)

## Installation

Clone the repository and install the dependencies via `pip`. 
Some dependencies like `torch` need to be installed manually beforehand as noted in the `requirements.txt`.

```bash
# Create and activate the virtual environment
python3 -m venv venv_sbg
source venv_sbg/bin/activate  # On Windows: venv_sbg\Scripts\activate

# Install manually 
pip install torch
pip install torch-geometric
pip install torch-scatter torch-sparse

# Install the required packages
pip install -r requirements.txt
```

## Project Structure

The repository code is organized into three main spaces:
- **`notebooks/`**: Post-hoc analysis, statistical evaluations, and interactive visualizations.
- **`src/`**: Core library modules containing GNN trainers, dataset samplers, and generation engines.
- **Root directory scripts**: Entry points for downloading datasets, running experiments, and utility analysis.

```text
syn_bench_gnn/
├── data/                       # Original datasets folder (TUDatasets)
├── synthetic_data/             # Generated synthetic datasets from different methods
├── logs/                       # Log files for tracking and debugging runs
├── notebooks/                  # Post-hoc analysis and visualizations
│   ├── embedding_dataset.ipynb         # Graph embedding and reconstruction analysis
│   ├── generation_results_analysis.ipynb # Evaluation of different synthetic generators
│   ├── gnn_result_analysis.ipynb       # Comparative GNN performance evaluations
│   ├── single_generation_analysis.ipynb # In-depth diagnostic of single graph fitting runs
│   └── visualization_utils.py          # Shared plotting utilities
├── src/                        # Core source code of the project
│   ├── anndg/                  # ANNDG and eccentricity statistical generator modules
│   ├── ergm/                   # Exponential Random Graph Model (ERGM) modules
│   ├── padma/                  # Degree moments generator modules
│   ├── data_utils.py           # Unified data loading and GNN preprocessing helpers
│   ├── enc_dec_dataset.py      # Statistical encoders/decoders (Moments, Percentile, GMCM)
│   ├── generate_datasets.py    # Pipeline entry point for synthetic dataset generation
│   ├── graph_analysis.py       # Metrics, Wasserstein distances, and feature analysis
│   ├── log_utils.py            # Unified logging config and helpers
│   └── train_gnn.py            # GNN model training and validation loops
├── add_distributional_metadata.py # script to enrich dataset files with distributional details
├── benchmark_gnn.py            # Main test suite to benchmark GNN architectures
├── download_tudatasets.py      # Downloader script for source TUDatasets
├── inspect_log_bins.py         # Utility to inspect log-degree distributions
├── main_experiment.py          # Full experimental pipeline (Generation -> GNN Training -> Evaluation)
├── test_zero_deg.py            # Helper utility to check for zero-degree nodes
├── train_original_only.py      # Baseline runner to train GNN models on original datasets only
├── requirements.txt            # Python dependencies
└── README.md                   # This README file
```

## Usage

`main_experiment.py` orchestrates the complete benchmark pipeline, which is divided into two phases:
- **Phase A (Generation)**: Downloads datasets (if not present) and generates multiple synthetic datasets (variants) for the selected TUDatasets using various graph generator algorithms.
- **Phase B (GNN Evaluation)**: Trains and evaluates GNN models on the original and synthetic datasets to assess structural similarity.

### Standard Command for a Full Benchmark Run
```bash
nohup python main_experiment.py --cut_datasets 500 --features_BinLogDeg --process_original > output.log 2>&1 &
```

---

### Command Line Arguments

`main_experiment.py` accepts the following arguments:

#### Dataset & Generation Settings
- `--dataset [DATASET ...]`  
  One or more TUDataset names to process (e.g., `MUTAG`, `BZR`, `DHFR`, `Mutagenicity`).  
  *Default:* `["BZR", "DHFR", "Mutagenicity", "MUTAG"]`
- `--cut_datasets N`  
  If set, each dataset is down-sampled to at most `N` graphs before generation. The down-sampling is stratified by label and preserves node/edge distributions.
- `--distribution_sampler SAMPLER`  
  Encoder-decoder to use for distributional statistic sampling. If omitted, per-graph statistics are replicated directly.  
  *Choices:* `gmcm`, `moments`, `percentile`, `percentile_corr`
- `--methods METHOD [METHOD ...]`  
  Graph generation methods to run.  
  *Choices:* `padma`, `pdd`, `ergm`, `dummyEdges`, `dummyNodes`, `anndg`, `anndgE`, `nextGen`  
  *Default:* `["dummyNodes", "dummyEdges", "padma", "anndg", "anndgE"]`
- `--num_synth_datasets V`, `-V V`  
  Number of independent synthetic variants *V* to generate per (dataset, method) pair.  
  *Default:* `20`.

#### GNN Training & Evaluation Settings
- `--process_original`  
  Pre-processes (and optionally down-samples) the original dataset even if it already exists on disk. Also enables GNN baseline evaluation on original data.
- `--gnn_runs R`, `-R R`  
  Number of independent GNN training runs per dataset for robust variance estimation.  
  *Default:* `10`
- `--features_BinLogDeg`  
  Uses log-binned degree one-hot vectors as node features instead of the default constant all-ones dummy vectors.

#### Execution & Infrastructure Settings
- `--seed SEED`  
  RNG seed for synthetic graph generation to ensure reproducibility. (Note: GNN training is always independently randomized). Pass `-1` for a fully stochastic generation run.  
  *Default:* `-1`
- `--skip_generation`  
  Skip Phase A (generation) and use existing synthetic datasets stored in `synthetic_data/`.
- `--skip_evaluation`  
  Skip Phase B (GNN evaluation).
- `--num_workers N`  
  Number of worker processes for parallel graph generation.  
  *Default:* `90%` of available CPU cores.
- `--quick_test`  
  Runs a rapid functional validation (1 epoch, 1 GNN run, smaller hidden dimensions).


## Results Analysis and Visualization

All notebooks have a synchronized `.py` copy in the same directory. These Python files are used to facilitate version control and integration with AI agents, and should be kept in sync with the corresponding `.ipynb` notebooks.

The most important notebooks are:
- `notebooks/generation_results_analysis.ipynb`: Analyzes the quality of the generated graphs and of the encoding-decoding scheme.
- `notebooks/gnn_result_analysis.ipynb`: Analysis of GNN performance on original and synthetic datasets.


## Auxiliary Scripts & Utilities

The root directory contains several auxiliary scripts and diagnostic tools:
- **`download_tudatasets.py`**: A helper script to download and locally cache raw TUDatasets.
- **`train_original_only.py`**: A baseline script to train and evaluate GNN architectures exclusively on original graphs.
- **`add_distributional_metadata.py`**, **`benchmark_gnn.py`**, **`inspect_log_bins.py`**, and **`test_zero_deg.py`**: Specialized tools for data-preparation, localized profiling, and graph sanity checks.

> [!WARNING]  
> These auxiliary scripts are currently **outdated and may not function correctly** due to recent codebase and API updates. They are **not** necessary to run the primary benchmarking pipeline (`main_experiment.py`), and it is highly recommended **not to use them** for the time being.