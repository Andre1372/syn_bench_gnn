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
from src.graph_analysis import analyze_single_graph, calculate_annd_error, calculate_moments_error, calculate_eccentricity_error
from notebooks.visualization_utils import plot_graph, plot_annd, plot_eccentricity
from src.anndg.graph_generator import generate_graph
from src.generate_datasets import generate_graph as generate_graph_by_method



# Cell 1 - Global Variables & Data Loading
DATASET = "BZR"
IDX = 330

try:
    dataset = DatasetPT(PROJECT_ROOT / "data" / DATASET / f"{DATASET}_original.pt")
except:
    dataset = TUDataset(PROJECT_ROOT / "data", name=DATASET)

GRAPH_NX = pytorch_to_networkx(dataset[IDX])
GRAPH_IG = networkx_to_igraph(GRAPH_NX)



# Cell 2 - Comparative Analysis
# Analyze Original Graph
target_stats = analyze_single_graph(GRAPH_IG)

# Generate and Analyze Synthetic Graph
rng = np.random.default_rng(seed=40689061)
GRAPH_SYNTH_NX, info = generate_graph(target_stats, rng=rng, debug=True, replicate_eccentricity=False)
GRAPH_SYNTH_IG = networkx_to_igraph(GRAPH_SYNTH_NX)
obtained_stats = analyze_single_graph(GRAPH_SYNTH_IG)

# Combined Visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Row 1: Graph Structures
plot_graph(graph=GRAPH_NX, ax=axes[0, 0], dataset_name=f"{DATASET} (Original)", graph_index=IDX)
plot_graph(graph=GRAPH_SYNTH_NX, ax=axes[0, 1], dataset_name=f"{DATASET} (Synthetic)", graph_index=IDX)

# Row 2: Topological Metrics
plot_annd(annd_values=info['best_annd'], ax=axes[1, 0], title="ANND Profile", target_graph=GRAPH_NX)
if 'best_eccentricity' in info:
    plot_eccentricity(ecc_values=info['best_eccentricity'], ax=axes[1, 1], title="Eccentricity Profile", target_graph=GRAPH_NX)
else:
    axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\n" + "="*70)
print(f"{'TOPOLOGICAL STATISTIC':<25} | {'ORIGINAL':<12} | {'SYNTHETIC':<12} | {'DELTA':<10}")
print("-" * 70)

# Scalar metrics comparison
scalar_metrics = ['assortativity', 'modularity', 'clustering', 'efficiency', 'diameter']
for m in scalar_metrics:
    orig = target_stats.get(m, 0.0)
    synth = obtained_stats.get(m, 0.0)
    delta = abs(orig - synth)
    print(f"{m.capitalize():<25} | {orig:>12.4f} | {synth:>12.4f} | {delta:>10.4f}")

# ANND comparison
print("-" * 70)
print(f"{'ANND Bin 0':<25} | {target_stats['annd'][0]:>12.4f} | {info['best_annd'][0]:>12.4f} | {abs(target_stats['annd'][0] - info['best_annd'][0]):>10.4f}")
print(f"{'ANND Bin 1':<25} | {target_stats['annd'][1]:>12.4f} | {info['best_annd'][1]:>12.4f} | {abs(target_stats['annd'][1] - info['best_annd'][1]):>10.4f}")
print(f"{'ANND Bin 2':<25} | {target_stats['annd'][2]:>12.4f} | {info['best_annd'][2]:>12.4f} | {abs(target_stats['annd'][2] - info['best_annd'][2]):>10.4f}")
print(f"{'ANND Bin 3':<25} | {target_stats['annd'][3]:>12.4f} | {info['best_annd'][3]:>12.4f} | {abs(target_stats['annd'][3] - info['best_annd'][3]):>10.4f}")
print("-" * 70)

annd_error = calculate_annd_error(info['best_annd'], target_stats['annd'])
if 'best_eccentricity' in info:
    ecc_error = calculate_eccentricity_error(info['best_eccentricity'], target_stats['eccentricity'])
else:
    ecc_error = np.nan
moments_error = calculate_moments_error(target_stats['degree_moments'], obtained_stats['degree_moments'])
diameter_error = abs(obtained_stats['diameter'] - target_stats['diameter']) / target_stats['diameter'] if target_stats['diameter'] > 0 else 0.0

print(f"\n{'STRUCTURAL ERROR SUMMARY':<25} | {'VALUE':<12}")
print("-" * 40)
print(f"{'Moments Error':<25} | {moments_error:>12.4f}")
print(f"{'ANND Error':<25} | {annd_error:>12.4f}")
print(f"{'Eccentricity Error':<25} | {ecc_error:>12.4f}")
print(f"{'Diameter Error':<25} | {diameter_error:>12.4f}")
print("="*70 + "\n")



# Cell 3 - Multi-Generator Graph Comparison
# Generate one synthetic graph per method and display alongside the original.
METHODS = ["dummyNodes", "dummyEdges", "padma", "anndg", "anndgE"]
METHOD_LABELS = {
    "dummyNodes": "Dummy Nodes (ER-p)",
    "dummyEdges": "Dummy Edges (GNM)",
    "padma":      "PADMA",
    "anndg":      "ANNDG",
    "anndgE":     "ANNDG+Ecc",
}

rng_cmp = np.random.default_rng(seed=99887766)

# Build target_stats once (reuse the already-computed one from Cell 2)
target_stats_for_gen = dict(target_stats)
target_stats_for_gen["observed_nx"] = GRAPH_NX  # needed by pdd / ergm (not used here)

generated = {}  # method -> nx.Graph
for method in METHODS:
    try:
        G_gen, _ = generate_graph_by_method(target_stats_for_gen, method=method, rng=rng_cmp)
        generated[method] = G_gen
    except Exception as exc:
        print(f"[WARNING] {method} failed: {exc}")
        generated[method] = nx.empty_graph(target_stats_for_gen.get("n_nodes", 1))

# 2 rows x 3 cols: row 0 → dummyNodes, dummyEdges, padma | row 1 → anndg, anndgE, original
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.suptitle(
    f"{DATASET} #{IDX} — Generator Comparison",
    fontsize=14, fontweight="bold", y=0.98,
)

plot_order = ["dummyNodes", "dummyEdges", "padma", "anndg", "anndgE"]
for ax, method in zip(axes.flat, plot_order):
    G = generated[method]
    label = METHOD_LABELS[method]
    n, m = G.number_of_nodes(), G.number_of_edges()
    plot_graph(graph=G, ax=ax, dataset_name=label, graph_index=IDX)
    ax.set_title(f"{label}\n(n={n}, m={m})", fontsize=10)
    
    # Enclose graph in a beautiful frame (riquadro) and add margin/padding
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.18)  # Leave room around nodes so graph looks smaller/well-framed
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#d3d3d3')
        spine.set_linewidth(1.0)

# Last cell: original graph
ax_orig = axes.flat[-1]
n_orig = GRAPH_NX.number_of_nodes()
m_orig = GRAPH_NX.number_of_edges()
plot_graph(graph=GRAPH_NX, ax=ax_orig, dataset_name=f"{DATASET} — Original", graph_index=IDX)
ax_orig.set_title(f"Original\n(n={n_orig}, m={m_orig})", fontsize=10)

# Enclose original graph in the same frame style
ax_orig.axis("on")
ax_orig.set_xticks([])
ax_orig.set_yticks([])
ax_orig.margins(0.18)
for spine in ax_orig.spines.values():
    spine.set_visible(True)
    spine.set_color('#d3d3d3')
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.show()