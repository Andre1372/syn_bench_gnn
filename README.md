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
│   ├── distribution_matching_analysis.ipynb # Statistical encoder-decoders (GMCM, Moments, Percentile) matching analysis
│   ├── explain_performances.ipynb      # GNN performance factors, topological attributes
│   ├── gen_flexibility_analysis.ipynb  # Synthetic generator parameters and structural flexibility analysis
│   ├── gnn_result_analysis.ipynb       # Comparative GNN performance evaluation on original vs synthetic datasets
│   ├── single_generation_analysis.ipynb # Visual diagnostic of single graph structure generations (2x3 subplot grids)
│   ├── target_fidelity_analysis.ipynb  # Evaluation of structural fidelity and emergent metrics
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
├── add_distributional_metadata.py # Script to enrich dataset files with distributional details [DEPRECATED]
├── benchmark_gnn.py            # Main test suite to benchmark GNN architectures [DEPRECATED]
├── download_tudatasets.py      # Downloader script for source TUDatasets (Optional helper)
├── inspect_log_bins.py         # Utility to inspect log-degree distributions [DEPRECATED]
├── main_experiment.py          # Full experimental pipeline (Generation -> GNN Training -> Evaluation) [PRIMARY ENTRY POINT]
├── test_zero_deg.py            # Helper utility to check for zero-degree nodes [DEPRECATED]
├── train_original_only.py      # Baseline runner to train GNN models on original datasets only [DEPRECATED]
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

The available analysis notebooks are:
- **`notebooks/distribution_matching_analysis.ipynb`**: Analyzes the quality of the encoding-decoding schemes and sampling strategies (GMCM, moments, percentile, percentile_corr) for reproducing distributions.
- **`notebooks/explain_performances.ipynb`**: Investigates how topological graph features influence class separability (PCA/t-SNE) and how feature strategies (e.g., log-binning degree vs. constant dummy features) affect GNN performance (GCN, GIN).
- **`notebooks/gen_flexibility_analysis.ipynb`**: Evaluates the adaptability and structural flexibility of different synthetic graph generators under varying configurations.
- **`notebooks/gnn_result_analysis.ipynb`**: Provides comparative GNN evaluation on original and synthetic datasets, showcasing performance preservation and downstream task utility.
- **`notebooks/single_generation_analysis.ipynb`**: Offers visual diagnostics of single-graph structure generations, featuring a 2x3 grid to compare target original structures against synthetic methods (`dummyNodes`, `dummyEdges`, `padma`, `anndg`, `anndgE`).
- **`notebooks/target_fidelity_analysis.ipynb`**: Rigorously evaluates structural fidelity (MAE of nodes, edges, degree moments, ANND, eccentricity) and implicit emergent properties (clustering, assortativity, modularity, efficiency, diameter) using scatter plots, CDF error distributions, KDE density curves, and performance heatmaps.


## Auxiliary Scripts & Utilities

The root directory contains several secondary scripts, though their use is highly restricted:
- **`download_tudatasets.py`**: A helper script that can be used to download, filter, and pre-cache raw TUDatasets structurally and through initial GNN evaluations.
  > [!NOTE]
  > Using this script is **entirely optional**. `main_experiment.py` is fully autonomous and will automatically download, pre-process, and split the required datasets if they are not already present in the `data/` folder.

- **`train_original_only.py`**, **`add_distributional_metadata.py`**, **`benchmark_gnn.py`**, **`inspect_log_bins.py`**, and **`test_zero_deg.py`**: Outdated baseline, data-preparation, and localized profiling tools.

> [!WARNING]  
> Apart from `download_tudatasets.py` (which is optional), **all other auxiliary root scripts are deprecated and it is strongly recommended NOT to use them**. They may not function correctly due to recent API refactorings, are completely redundant, and are not needed for any part of the workflow. The full pipeline is completely handled by `main_experiment.py`.