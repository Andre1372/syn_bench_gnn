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


def aggregate_statistics(per_graph_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes the mean for network statistics across the dataset.
    
    Handles both scalar values and vectors (like ANND) of inhomogeneous lengths.
    Safely ignores missing keys in individual graph dictionaries.

    Args:
        per_graph_stats: A list of dictionaries, where each dict contains absolute statistics for a graph.
    Returns:
        A dictionary mapping statistic names to their aggregated mean value.
    Raises:
        ValueError: If the input list is empty.
    """
    if not per_graph_stats:
        raise ValueError("Input per_graph_stats list cannot be empty.")

    keys = per_graph_stats[0].keys()
    mean_stats: dict[str, Any] = {}

    for key in keys:
        values = [stat[key] for stat in per_graph_stats if key in stat]
        
        if not values: continue

        # Handle numeric arrays (ANND, degree moments)
        if isinstance(values[0], np.ndarray):

            # For variable length vectors (ANND)
            max_len = max(len(v) for v in values)
            # Create a padded matrix with NaNs for alignment
            padded = np.full((len(values), max_len), np.nan)
            for i, v in enumerate(values):
                padded[i, :len(v)] = v
            
            # Compute element-wise mean across graphs, ignoring NaNs
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_vec = np.nanmean(padded, axis=0)
            mean_stats[key] = np.nan_to_num(mean_vec, nan=0.0)

        else:
            # Standard scalar mean calculation
            mean_stats[key] = float(np.mean(values))

    return mean_stats


def per_graph_statistics(data_list: list[Data], show_progress: bool = False) -> list[dict[str, Any]]:
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
    diameter = calculate_diameter(graph)
    annd = calculate_annd(graph)
    ecc_moments = calculate_eccentricity_moments(graph, k=4)

    stats_dict: dict[str, float] = {
        "n_nodes": int(graph.vcount()),
        "n_edges": int(graph.ecount()),
        "modularity": modularity,
        "clustering": clustering,
        "assortativity": assortativity,
        "efficiency": efficiency,
        "diameter": diameter,
        "annd": annd,
        "normalized_degree_moments": deg_moments,
        "ecc_moments": ecc_moments
    }

    # for i, val in enumerate(motifs):
    #     stats_dict[f"motif_count_{i+1}"] = float(val)

    return stats_dict


# ------------------------------------------------------------------
# Graph target stats
# ------------------------------------------------------------------

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


def calculate_annd(graph: ig.Graph, bins: int = 4) -> np.ndarray:
    """
    Computes the Average Nearest Neighbor Degree (ANND) of the graph, normalizes it and bins it into percentiles.

    ANND characterizes degree-degree correlations. The function normalizes neighbor degrees 
    by the maximum possible degree (N-1) and aggregates nodes into 'bins' percentile groups 
    based on their degree rank.

    Args:
        graph: The igraph.Graph object to analyze.
        bins: The number of percentile bins to aggregate into.
    Returns:
        np.ndarray: Array of length `bins` containing the mean normalized ANND per bin.
    """
    n_nodes = graph.vcount()
    if n_nodes == 0:
        return np.zeros(bins, dtype=float)

    knn_nodes, _ = graph.knn()
    
    if not knn_nodes:
        return np.zeros(bins, dtype=float)

    # Normalize by (N-1) and handle NaNs from isolated nodes
    annd_raw = np.array(knn_nodes, dtype=float)
    annd_norm = np.nan_to_num(annd_raw / (n_nodes - 1), nan=0.0)

    # Sort ANND by degree rank to enable grouping into percentile bins
    node_degrees = np.array(graph.degree(), dtype=int)
    sorted_annd = annd_norm[np.argsort(node_degrees)]

    # Partition and average using array_split to handle non-divisible N_nodes gracefully
    return np.array([
        group.mean() if group.size > 0 else 0.0 
        for group in np.array_split(sorted_annd, bins)
    ], dtype=float)


def calculate_eccentricity_moments(graph: ig.Graph, k: int = 4) -> np.ndarray:
    """Calculates the raw eccentricity moments of the graph.
    
    Eccentricity is the maximum shortest path distance from a node to any 
    other node in the graph: e(u) = max_{v \in V} d(u, v). 
        
    Args:
        graph: The input igraph.Graph object.
        k: The number of moments to calculate. Must be strictly positive.
    Returns:
        An array of length `k` containing the eccentricity moments.        
    Raises:
        ValueError: If `k` is less than 1 or greater than 4.
    """
    if k < 1 or k > 4: raise ValueError(f"Number of moments 'k' must be between 1 and 4, got {k}.")
    
    n = graph.vcount()
    if n == 0: return np.zeros(k, dtype=np.float64)

    # Leverage igraph's highly optimized C backend for shortest path calculations
    eccentricities = np.array(graph.eccentricity(), dtype=np.float64)

    m1 = float(np.mean(eccentricities))
    m2 = float(np.var(eccentricities, ddof=0))

    if m2 > 1e-10:
        m3 = float(stats.skew(eccentricities, bias=False))
        m4 = float(stats.kurtosis(eccentricities, bias=False)) + 3.0
    else:
        m3 = 0.0
        m4 = 3.0

    return np.array([m1, m2, m3, m4])[:k]


def count_motifs(graph: ig.Graph, k: int, sampling_probs: list[float] | None = None) -> np.ndarray:
    """Counts the induced occurrences of connected motifs up to size k.
    
    Args:
        graph: The input igraph.Graph.
        k: Maximum motif size to investigate (3, 4, or 5).
        sampling_probs: Optional list of probabilities for sampling at each size 
            from 3 to k. Used to estimate motif counts in large graphs.
    Returns:
        A NumPy array containing the frequency of each valid, connected, non-isomorphic motif.
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


# ------------------------------------------------------------------
# Topological metrics
# ------------------------------------------------------------------

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
        The global efficiency as a float. Returns 0.0 if the graph has less than two nodes or no edges.
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


def calculate_diameter(graph: ig.Graph) -> float:
    """Calculates the diameter of the graph.

    For disconnected graphs, returns the diameter of the largest connected component.

    Args:
        graph: The input igraph.Graph.
    Returns:
        The diameter of the graph as a float. Returns 0.0 if the graph is empty.
    """
    if graph.vcount() == 0: return 0.0
    
    # igraph.diameter() returns the length of the longest shortest path.
    # By default, unconn=True returns the max diameter among all components.
    return float(graph.diameter(directed=False, unconn=True))    


def calculate_moments_error(obtained_moments: np.ndarray, target_moments: np.ndarray) -> float:
    """Calculates the structural error between obtained degree moments and target moments.
    
    Inspired by Padma's maxent objective:
    - Uses relative error with symmetric scaling: scale = min(|actual|, |target|) + 1e-6.
    - Error metric: log1p(abs_diff / scale) ** 2.5.
    - Aggregates variance, skewness, and kurtosis losses using an L2 norm.
    
    Args:
        obtained_moments: NumPy array [mean, variance, skewness, kurtosis].
        target_moments: NumPy array [mean, variance, skewness, kurtosis].
    Returns:
        A scalar error value representing the discrepancy in degree moments.
    """
    moment_losses = []

    def _compute_metric_loss(actual_value, target_value):
        a = np.arcsinh(actual_value)
        b = np.arcsinh(target_value)
        loss = np.abs(a - b)**1.5

        return loss

    # Variance loss
    var_actual, var_target = obtained_moments[1], target_moments[1]
    moment_losses.append(_compute_metric_loss(var_actual, var_target))
    
    # Skewness loss (evaluated only when variance is stable)
    if var_actual > 1e-12:
        skew_actual, skew_target = obtained_moments[2], target_moments[2]
        moment_losses.append(_compute_metric_loss(skew_actual, skew_target))
        
    # Kurtosis loss
    kurt_actual, kurt_target = obtained_moments[3], target_moments[3]
    moment_losses.append(_compute_metric_loss(kurt_actual, kurt_target))
    
    if not moment_losses: return 0.0
        
    losses = np.asarray(moment_losses, dtype=float)
    # L2 norm aggregation
    penalty = float(np.mean(losses ** 2.0) ** 0.5)
    
    return penalty


def calculate_annd_error(
    obtained_annd: np.ndarray, obtained_degree_sequence: np.ndarray, 
    target_annd: np.ndarray, target_degree_sequence: np.ndarray) -> float:
    """Calculates the structural error between obtained ANND and target ANND.
    
    Args:
        obtained_annd: NumPy array of obtained ANND.
        obtained_degree_sequence: NumPy array of obtained degree sequence.
        target_annd: NumPy array of target ANND.
        target_degree_sequence: NumPy array of target degree sequence.
    Returns:
        A scalar error value representing the discrepancy in ANND.
    """    
    bins = len(obtained_annd)

    # Target distribution
    # Slicing from 1 to max_k+1 pulls degrees [1, ..., max_k]
    p_target = np.bincount(target_degree_sequence, minlength=bins + 1)[1:bins + 1] / len(target_degree_sequence)
        
    # Obtained distribution
    p_obtained = np.bincount(obtained_degree_sequence, minlength=bins + 1)[1:bins + 1] / len(obtained_degree_sequence)

    # Weights w(k) = (P1(k) + P2(k)) / 2
    weights = (obtained_annd + p_obtained) * 0.5

    loss = np.log1p(40 * np.abs(obtained_annd - target_annd)**1.5)
    error = np.dot(weights, loss)
    
    return float(error)


def calculate_eccentricity_error(obtained_ecc_moments: np.ndarray, target_ecc_moments: np.ndarray) -> float:
    """Calculates the structural error between obtained eccentricity moments and target eccentricity moments.
    
    Args:
        obtained_ecc_moments: NumPy array of obtained eccentricity moments.
        target_ecc_moments: NumPy array of target eccentricity moments.
    Returns:
        A scalar error value representing the discrepancy in eccentricity moments.
    """    
    k = len(target_ecc_moments)
    if k < 1: raise ValueError("Target eccentricity moments must have at least one element.")
    if len(obtained_ecc_moments) != k:
        raise ValueError("Obtained and target eccentricity moments must have the same length.")
    
    moment_losses = np.zeros(k)

    def _compute_metric_loss(actual_value, target_value):
        a = np.arcsinh(actual_value)
        b = np.arcsinh(target_value)
        loss = np.abs(a - b)**1.5

        return loss

    # Mean loss
    mean_actual, mean_target = obtained_ecc_moments[0], target_ecc_moments[0]
    moment_losses[0] = _compute_metric_loss(mean_actual, mean_target)
    
    # Variance loss
    if k > 1:
        var_actual, var_target = obtained_ecc_moments[1], target_ecc_moments[1]
        moment_losses[1] = _compute_metric_loss(var_actual, var_target)
    
    # Skewness loss (evaluated only when variance is stable)
    if k > 2 and var_actual > 1e-12:
        skew_actual, skew_target = obtained_ecc_moments[2], target_ecc_moments[2]
        moment_losses[2] = _compute_metric_loss(skew_actual, skew_target)
        
    # Kurtosis loss
    if k > 3:
        kurt_actual, kurt_target = obtained_ecc_moments[3], target_ecc_moments[3]
        moment_losses[3] = _compute_metric_loss(kurt_actual, kurt_target)
            
    penalty = float(np.mean(moment_losses ** 2.0) ** 0.5)
    
    return penalty