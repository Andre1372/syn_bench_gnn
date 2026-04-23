"""
This script is the python replica of a notebook, so it is not meant to be run as a script.
"""

# Cell 0 - Imports
from pathlib import Path
import sys

import torch
import numpy as np
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import TUDataset

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import pytorch_to_networkx, networkx_to_igraph, DatasetPT
from src.graph_analysis import analyze_single_graph, calculate_annd_error
from src.generate_datasets import generate_graph
from notebooks.visualization_utils import plot_graph, plot_annd



# Cell 1 - Global Variables & Data Loading
DATASET = "MUTAG"
METHOD = "anndg"  # Options: "padma", "anndg", etc.
IDX = 0

try:
    dataset = DatasetPT(PROJECT_ROOT / "data" / DATASET / f"{DATASET}_original.pt")
except:
    dataset = TUDataset(PROJECT_ROOT / "data", name=DATASET)

GRAPH_NX = pytorch_to_networkx(dataset[IDX])
GRAPH_IG = networkx_to_igraph(GRAPH_NX)



# Cell 2 - Original Graph Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_graph(graph=GRAPH_NX, ax=ax1, dataset_name=f"{DATASET} ({METHOD})", graph_index=IDX)
plot_annd(graph=GRAPH_NX, ax=ax2, title=f"ANND - {DATASET} ({METHOD})")

# Calculate and print assortativity
target_stats = analyze_single_graph(GRAPH_IG)

plt.tight_layout()
plt.show()


# Cell 3 - Synthetic Graph Generation (PADMA)
# Generate synthetic graph
rng = np.random.default_rng(seed=42)
GRAPH_SYNTH_NX = generate_graph(target_stats, method=METHOD, rng=rng)
GRAPH_SYNTH_IG = networkx_to_igraph(GRAPH_SYNTH_NX)

# Plot synthetic results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_graph(graph=GRAPH_SYNTH_NX, ax=ax1, dataset_name=f"{DATASET} (Synthetic)", graph_index=IDX)
plot_annd(graph=GRAPH_SYNTH_NX, ax=ax2, title=f"ANND - Synthetic", label='Synthetic $k_{nn}(k)$')

# Calculate and print assortativity
obtained_stats = analyze_single_graph(GRAPH_SYNTH_IG)

plt.tight_layout()
plt.show()

print("\n" + "="*30)
print("STRUCTURAL ERROR ANALYSIS")
print("="*30)

# Calculate ANND and degree sequences for both graphs
deg_seq_original = np.array(GRAPH_IG.degree())
deg_seq_synthetic = np.array(GRAPH_SYNTH_IG.degree())

# Compute the structural error between the ANND of the synthetic and original graphs
annd_error = calculate_annd_error(
    obtained_annd=obtained_stats['annd'], 
    obtained_degree_sequence=deg_seq_synthetic, 
    target_annd=target_stats['annd'], 
    target_degree_sequence=deg_seq_original
)

print(f"ANND Structural Error (Synthetic vs {METHOD.capitalize()}): {annd_error:.4f}")
print(f"Original Assortativity:  {target_stats['assortativity']:.4f}")
print(f"Synthetic Assortativity: {obtained_stats['assortativity']:.4f}")
print(f"Original ANND: {target_stats['annd']}")
print(f"Synthetic ANND: {obtained_stats['annd']}")
print(f"Original Degree Moments: {target_stats['normalized_degree_moments']}")
print(f"Synthetic Degree Moments: {obtained_stats['normalized_degree_moments']}")
print("="*30)