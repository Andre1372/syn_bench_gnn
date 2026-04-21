import logging
from collections import defaultdict

import igraph as ig
import numpy as np
from scipy import stats
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Any

from src.data_utils import pytorch_to_igraph


logger = logging.getLogger(__name__)


def aggregate_statistics(per_graph_stats: list[dict[str, float]]) -> dict[str, float]:
    """Computes the mean for network statistics across the dataset.

    Args:
        per_graph_stats: A list of dictionaries, where each dict contains absolute statistics for a graph.
    Returns:
        A dictionary mapping statistic names to their mean value.
    Raises:
        ValueError: If the input list is empty.
    """
    if not per_graph_stats:
        raise ValueError("Input per_graph_stats list cannot be empty.")

    keys = per_graph_stats[0].keys()
    mean_stats: dict[str, float] = {}
    for key in keys:
        mean_stats[key] = float(np.mean([stat[key] for stat in per_graph_stats]))

    return mean_stats


def aggregate_statistics_per_class(data_list: list[Data], per_graph_stats: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Computes the mean for network statistics across the dataset, grouped by class label.

    Args:
        data_list: List of PyG Data graphs.
        per_graph_stats: List of dictionaries containing absolute statistics for each graph.
    Returns:
        A dictionary mapping class labels (as strings) to a dictionary of aggregated statistics.
    """
    class_groups = defaultdict(list)
    for data, stats in zip(data_list, per_graph_stats):
        label = int(data.y.item())
        class_groups[label].append(stats)
        
    class_stats = {}
    for label, stats_list in class_groups.items():
        if stats_list:
            class_stats[str(label)] = aggregate_statistics(stats_list)
            
    return class_stats


def per_graph_statistics(data_list: list[Data], show_progress: bool = False) -> list[dict[str, float]]:
    """Calculates absolute topological and motif statistics for each graph in a list.

    Args:
        data_list: List of PyG Data graphs to analyze.
        show_progress: Whether to show a progress bar.
    Returns:
        A list of dictionaries containing absolute statistics for each graph.
    """
    all_stats: list[dict[str, float]] = []

    for data in tqdm(data_list, desc="Analyzing graph statistics", disable=not show_progress):
        ig_graph = pytorch_to_igraph(data)
        stats_vals = analyze_single_graph(ig_graph)
        all_stats.append(stats_vals)

    return all_stats


def analyze_single_graph(graph: ig.Graph) -> dict[str, float]:
    """Computes a comprehensive set of network statistics for a single graph.

    Args:
        graph: The igraph.Graph to analyze.
    Returns:
        A dictionary containing absolute values for modularity, clustering,
        assortativity, degree moments (1-4), and motif counts (size 3-4).
    """
    deg_moments = count_deg_moments(graph)
    motifs = count_motifs(graph, k=4)
    modularity = calculate_modularity(graph)
    clustering = calculate_clustering_coefficient(graph)
    assortativity = calculate_degree_assortativity(graph)
    efficiency = calculate_global_efficiency(graph)

    stats_dict: dict[str, float] = {
        "n_nodes": float(graph.vcount()),
        "n_edges": float(graph.ecount()),
        "modularity": modularity,
        "clustering": clustering,
        "assortativity": assortativity,
        "efficiency": efficiency,
    }

    for i, val in enumerate(deg_moments):
        stats_dict[f"deg_moment_{i+1}"] = float(val)

    # for i, val in enumerate(motifs):
    #     stats_dict[f"motif_count_{i+1}"] = float(val)

    return stats_dict


def count_motifs(graph: ig.Graph, k: int, sampling_probs: list[float] | None = None) -> np.ndarray:
    """Counts the induced occurrences of connected motifs up to size k.
    
    Args:
        graph: The input igraph.Graph.
        k: Maximum motif size to investigate (3, 4, or 5).
        sampling_probs: Optional list of probabilities for sampling at each size 
            from 3 to k. Used to estimate motif counts in large graphs.
    Returns:
        A NumPy array containing the frequency of each valid, connected, 
        non-isomorphic motif.
    """
    if sampling_probs is not None:
        if len(sampling_probs) != k:
            raise ValueError(f"sampling_probs must have length {k}")
        if not all(0 <= p <= 1 for p in sampling_probs):
            raise ValueError("All sampling probabilities must be between 0 and 1")

    number_motifs: list[float] = []
    num_nodes = graph.vcount()

    for s in range(3, k + 1):
        # Determine the number of isoclass for the current size
        probe = ig.Graph.Full(s, directed=False)
        num_classes = len(probe.motifs_randesu(size=s))

        # Handle NaNs (non-existent or unconnected isomorphism classes)
        if num_nodes < s:
            counts: list[int | float | None] = [0] * num_classes
        else:
            if sampling_probs is None:
                # Compute exact motif counts
                counts = graph.motifs_randesu(size=s)
            else:
                # Set up probabilities for current size s
                # Note: igraph expects cut_prob = 1 - p
                cur_probs = sampling_probs[:s]
                not_probs = [1.0 - p for p in cur_probs]
                raw_counts = graph.motifs_randesu(size=s, cut_prob=not_probs)
                prob_prod = float(np.prod(cur_probs))
                counts = [(c / prob_prod) if c is not None else None for c in raw_counts]

        # Match with theoretical compendium
        for i, count in enumerate(counts):
            if ig.Graph.Isoclass(s, i, directed=False).is_connected():
                val = count if (count is not None and not np.isnan(count)) else 0.0
                number_motifs.append(float(val))

    return np.array(number_motifs)


def count_deg_moments(graph: ig.Graph) -> np.ndarray:
    """Computes the first n moments of the graph's degree distribution.
    
    Returns:
        NumPy array [mean, variance, skewness, kurtosis].
    """
    n = graph.vcount()
    degrees = np.array(graph.degree(), dtype=float)
    
    if n == 0: return np.zeros(4)

    # Normalize by size
    degrees = degrees / max(n - 1, 1)

    m1 = float(np.mean(degrees))
    m2 = float(np.var(degrees, ddof=0))

    if m2 > 1e-10:
        m3 = float(stats.skew(degrees, bias=False))
        m4 = float(stats.kurtosis(degrees, bias=False)) + 3.0
    else:
        m3 = 0.0
        m4 = 3.0

    return np.array([m1, m2, m3, m4])


def calculate_modularity(graph: ig.Graph) -> float:
    """Calculates the modularity score using the Leiden algorithm for community detection.

    Args:
        graph: The input igraph.Graph.
    Returns:
        The computed modularity score. Returns 0.0 if the graph has no edges.
    """
    if graph.ecount() == 0:
        return 0.0

    # The Leiden algorithm is used to find an optimal partition (membership)
    partition = graph.community_leiden(objective_function="modularity")
    return float(graph.modularity(partition.membership))


def calculate_clustering_coefficient(graph: ig.Graph) -> float:
    """Calculates the global clustering coefficient (transitivity) of the graph.

    Args:
        graph: The input igraph.Graph.
    Returns:
        The global transitivity as a float. Returns 0.0 if not defined (e.g., NaN).
    """
    val = graph.transitivity_undirected()
    return 0.0 if np.isnan(val) else float(val)


def calculate_degree_assortativity(graph: ig.Graph) -> float:
    """Calculates the degree assortativity of the undirected graph.

    Args:
        graph: The input igraph.Graph.
    Returns:
        The assortativity coefficient. Returns 0.0 if the graph has no edges or 
        if all nodes have the same degree (which would otherwise result in NaN).
    """
    if graph.ecount() == 0:
        return 0.0

    # The calculation is performed for undirected graphs
    assortativity = graph.assortativity_degree(directed=False)
    
    return 0.0 if np.isnan(assortativity) else float(assortativity)


def calculate_global_efficiency(graph: ig.Graph) -> float:
    """Calculates the global efficiency of the graph.

    Global efficiency is defined as the average of the inverse of the shortest 
    path lengths between all pairs of nodes.

    Args:
        graph: The input igraph.Graph.
    Returns:
        The global efficiency as a float. Returns 0.0 if the graph has 
        less than two nodes or no edges.
    """
    n = graph.vcount()
    if n < 2 or graph.ecount() == 0: return 0.0

    # Get all-pairs shortest path distances.
    dist = np.array(graph.distances())
    
    # Mask to extract only off-diagonal elements (i != j)
    off_diag_dist = dist[~np.eye(n, dtype=bool)]
    
    # Calculate inverse distances. 1/inf is treated as 0 (disconnected nodes).
    inv_dist = np.divide(1.0, off_diag_dist, out=np.zeros_like(off_diag_dist, dtype=float), where=(off_diag_dist > 0) & (off_diag_dist != np.inf))
    
    return float(np.sum(inv_dist) / (n * (n - 1)))
