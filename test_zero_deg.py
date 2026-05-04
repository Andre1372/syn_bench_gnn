
from pathlib import Path
import sys

import numpy as np
import igraph as ig
from torch_geometric.datasets import TUDataset
from tqdm import tqdm # Opzionale, per vedere il progresso

# Ensure project root is in path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import pytorch_to_networkx, networkx_to_igraph, DatasetPT
from src.graph_analysis import analyze_single_graph
from src.generate_datasets import generate_graph



DATASET = "Mutagenicity"
METHOD = "padma"
try:
    dataset = DatasetPT(PROJECT_ROOT / "data" / DATASET / f"{DATASET}_original.pt")
except:
    dataset = TUDataset(PROJECT_ROOT / "data", name=DATASET)

N_VARIANTS = 5  # Numero di varianti sintetiche per ogni grafo originale
N_GRAPHS = 400

results = {
    "unexpected_zeros": [],
    "delta_existing_zeros": [],
    "out_of_range": []
}
normalized_results = {
    "unexpected_zeros": [],
    "delta_existing_zeros": [],
    "out_of_range": []
}

def _get_degree_distribution(graph: ig.Graph) -> np.ndarray:
    """
    Calculates the normalized degree distribution of the graph using igraph.
    
    Args:
        graph: The igraph.Graph instance.
    Returns:
        np.ndarray: An array where entry i is the fraction of nodes with degree i.
    """
    n = graph.vcount()
    if n == 0: 
        return np.array([], dtype=float)
    
    degrees = graph.degree()
    counts = np.bincount(degrees)
    return counts

for idx in tqdm(range(min(N_GRAPHS, len(dataset)))):
    data = dataset[idx]
    GRAPH_NX = pytorch_to_networkx(data)
    GRAPH_IG = networkx_to_igraph(GRAPH_NX)
    
    degrees_orig_list = GRAPH_IG.degree()
    dist_orig = _get_degree_distribution(GRAPH_IG)
    min_deg = min(degrees_orig_list)
    max_deg = max(degrees_orig_list)
    
    target_stats = analyze_single_graph(GRAPH_IG)
    n_nodes = target_stats["n_nodes"]
    cnt_zero_orig = dist_orig[0] if len(dist_orig) > 0 else 0

    v_unexpected = []
    v_delta = []
    v_range = []
    for v in range(N_VARIANTS):
        GRAPH_SYNTH_NX = generate_graph(target_stats, method=METHOD, rng=None)
        GRAPH_SYNTH_IG = networkx_to_igraph(GRAPH_SYNTH_NX)
        dist_obt = _get_degree_distribution(GRAPH_SYNTH_IG)
        
        cnt_zero_obt = dist_obt[0] if len(dist_obt) > 0 else 0

        if min_deg > 0:
            v_unexpected.append(cnt_zero_obt)
        else:
            v_delta.append(abs(cnt_zero_orig - cnt_zero_obt))
        lower_than_min = np.sum(dist_obt[:min_deg]) if min_deg > 0 else 0
        higher_than_max = np.sum(dist_obt[max_deg + 1:]) if len(dist_obt) > max_deg + 1 else 0
        v_range.append(lower_than_min + higher_than_max)

    avg_unexpected = np.mean(v_unexpected) if v_unexpected else 0
    avg_delta = np.mean(v_delta) if v_delta else 0
    avg_range = np.mean(v_range)

    if avg_unexpected > 0:
        results["unexpected_zeros"].append(avg_unexpected)
        normalized_results["unexpected_zeros"].append(avg_unexpected / n_nodes)
    if avg_delta > 0:
        results["delta_existing_zeros"].append(avg_delta)
        normalized_results["delta_existing_zeros"].append(avg_delta / n_nodes)
    if avg_range > 0:
        results["out_of_range"].append(avg_range)
        normalized_results["out_of_range"].append(avg_range / n_nodes)
    if avg_unexpected > 0 or avg_delta > 0 or avg_range > 0:
        print(f"IDX: {idx:3d} | Avg Unexpected Isolated: {avg_unexpected:4.1f} | Avg Delta Isolated: {avg_delta:4.1f} | Avg Out of range: {avg_range:4.1f}")

print("\n" + "="*50)
for key in results.keys():
    vals = results[key]
    norm_vals = normalized_results[key]
    if vals:
        print(f"\n{key.replace('_', ' ').upper()}:")
        print(f"  Graphs with error: {len(vals)}/{N_GRAPHS}")
        print(f"  Mean error (nodes): {np.mean(vals):.2f} | Normalized: {np.mean(norm_vals):.4f}")
        print(f"  Max error  (nodes): {np.max(vals):.2f} | Normalized: {np.max(norm_vals):.4f}")
    else:
        print(f"\n{key.replace('_', ' ').upper()}: No errors.")
