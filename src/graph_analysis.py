"""Graph analysis utilities: per-graph and aggregate topology statistics."""

import logging

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
    
    Handles both scalar values and vectors.
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

        # Handle numeric arrays
        if isinstance(values[0], np.ndarray):
            # Compute element-wise mean across graphs, ignoring NaNs
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_vec = np.nanmean(values, axis=0)
            mean_stats[key] = np.nan_to_num(mean_vec, nan=0.0)

        else:
            # Standard scalar mean calculation
            mean_stats[key] = float(np.mean(values))

    return mean_stats


def per_graph_statistics(data_list: list[Data], precomputed_stats: list[dict[str, Any]] | None = None, show_progress: bool = False) -> list[dict[str, Any]]:
    """Calculates absolute topological and motif statistics for each graph in a list.

    Args:
        data_list: List of PyG Data graphs to analyze.
        precomputed_stats: Optional list of dictionaries containing precomputed statistics
            to be merged (and potentially override) the analyzed values.
        show_progress: Whether to show a progress bar.
    Returns:
        A list of dictionaries containing absolute statistics for each graph.
    """
    all_stats: list[dict[str, Any]] = []

    for i, data in enumerate(tqdm(data_list, desc="Analyzing graph statistics", disable=not show_progress)):
        ig_graph = pytorch_to_igraph(data)
        stats_vals = analyze_single_graph(ig_graph)
        
        # Merge precomputed stats if available for this specific graph
        if precomputed_stats is not None:
            stats_vals.update(precomputed_stats[i])
            
        all_stats.append(stats_vals)

    return all_stats


def analyze_single_graph(graph: ig.Graph) -> dict[str, float]:
    """Computes a comprehensive set of topological statistics for a single graph.

    Args:
        graph: The igraph.Graph to analyze.
    Returns:
        A dictionary with keys: ``'n_nodes'``, ``'n_edges'``, ``'modularity'``,
        ``'clustering'``, ``'assortativity'``, ``'efficiency'``, ``'diameter'``,
        ``'degree_moments'`` (array of 4 normalised moments), ``'annd'``
        (binned average nearest-neighbour degree), ``'eccentricity'``
        (binned normalised eccentricity), and ``'motifs'`` (array of normalised
        connected-motif counts for sizes 3 and 4, divided by ``C(n, s)`` so
        values are in ``[0, 1]`` and comparable across graphs of different sizes).
    """
    deg_moments = count_deg_moments(graph)
    modularity = calculate_modularity(graph)
    clustering = calculate_clustering_coefficient(graph)
    assortativity = calculate_degree_assortativity(graph)
    efficiency = calculate_global_efficiency(graph)
    diameter = calculate_diameter(graph)
    annd, bin_indices = calculate_annd(graph)
    eccentricity, _ = calculate_eccentricity(graph, bin_indices=bin_indices)

    stats_dict: dict[str, float] = {
        "n_nodes": int(graph.vcount()),
        "n_edges": int(graph.ecount()),
        "modularity": modularity,
        "clustering": clustering,
        "assortativity": assortativity,
        "efficiency": efficiency,
        "diameter": diameter,
        "degree_moments": deg_moments,
        "annd": annd,
        "eccentricity": eccentricity,
    }

    stats_dict["motifs"] = count_motifs_normalized(graph, k=4)

    return stats_dict


# ------------------------------------------------------------------
# Graph target stats
# ------------------------------------------------------------------

def count_deg_moments(graph: ig.Graph) -> np.ndarray:
    """Computes the first four moments of the graph's normalised degree distribution.

    Node degrees are normalised by ``max(N-1, 1)`` before moment computation.
    Skewness and kurtosis are set to 0 and 3 respectively when the degree
    variance is negligible (< 1e-10), avoiding numerical instability.

    Args:
        graph: The igraph.Graph to analyze.
    Returns:
        A NumPy array ``[mean, variance, skewness, Pearson_kurtosis]`` of the
        normalised degree distribution. Returns ``np.zeros(4)`` for empty graphs.
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


def calculate_annd(graph: ig.Graph, bins: int = 4, bin_indices: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Computes the binned, normalised Average Nearest-Neighbour Degree (ANND).

    Nodes are ranked by degree and split into ``bins`` equal-sized groups.
    Within each bin the mean raw ANND is computed and then divided by ``N-1``
    to normalise into [0, 1].  Isolated nodes (degree 0) are excluded from
    binning.  If ``bin_indices`` is supplied the binning step is skipped.

    Args:
        graph: The igraph.Graph object to analyze.
        bins: Number of degree-rank percentile bins to aggregate into.
        bin_indices: Optional pre-computed list of node-index arrays (one per
            bin) as returned by a previous call to this function.  When
            provided the degree-sorting and array-split steps are skipped.
    Returns:
        A tuple ``(annd_bins, bin_indices)`` where:

        - ``annd_bins`` — float array of length ``bins`` with the mean
          normalised ANND per bin.
        - ``bin_indices`` — list of index arrays (one per bin) of the active
          nodes that were used to compute each bin.
    """
    n_nodes = graph.vcount()
    if n_nodes == 0:
        return np.zeros(bins, dtype=float), bin_indices if bin_indices is not None else [np.array([], dtype=int)] * bins

    knn_nodes, _ = graph.knn()
    
    if not knn_nodes:
        return np.zeros(bins, dtype=float), bin_indices if bin_indices is not None else [np.array([], dtype=int)] * bins

    if bin_indices is None:
        node_degrees = np.array(graph.degree(), dtype=int)
        n_active = np.count_nonzero(node_degrees)

        if n_active <= 1:
            return np.zeros(bins, dtype=float), [np.array([], dtype=int)] * bins

        if n_active == n_nodes:
            # Fast-path: all nodes are active
            fixed_active_indices = np.argsort(node_degrees)
        else:
            # Consider only active nodes
            active_mask = node_degrees > 0
            active_indices = np.where(active_mask)[0]
            # Sort active indices by their degree
            fixed_active_indices = active_indices[np.argsort(node_degrees[active_mask])]
        
        # Partition using array_split to handle non-divisible N_active gracefully
        bin_indices = np.array_split(fixed_active_indices, bins)
    else:
        # If provided, we assume it already contains indices of nodes with degree >= 1
        n_active = sum(len(b) for b in bin_indices)

        if n_active <= 1:
            return np.zeros(bins, dtype=float), bin_indices

    annd_raw = np.array(knn_nodes, dtype=float)
    norm_factor = (n_nodes - 1)

    # Calculate mean for each group of nodes using indices
    annd_bins = np.array([
        annd_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0 
        for indices in bin_indices
    ], dtype=float)

    return annd_bins, bin_indices


def calculate_eccentricity(graph: ig.Graph, bins: int = 4, bin_indices: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Computes the binned, normalised node eccentricity.

    Eccentricity ``e(u) = max_{v in V} d(u, v)`` is computed for every node
    via the all-pairs distance matrix.  Unreachable pairs (infinite distance)
    are treated as ``-1`` so they do not inflate the maximum.  Nodes are then
    ranked by degree, split into ``bins`` equal groups, and the mean raw
    eccentricity within each bin is divided by ``N-1`` to normalise into [0, 1].
    Isolated nodes (degree 0) are excluded from binning.  If ``bin_indices``
    is supplied the binning step is skipped.

    Args:
        graph: The igraph.Graph object to analyze.
        bins: Number of degree-rank percentile bins to aggregate into.
        bin_indices: Optional pre-computed list of node-index arrays (one per
            bin) as returned by :func:`calculate_annd`.  When provided the
            degree-sorting and array-split steps are skipped.
    Returns:
        A tuple ``(ecc_bins, bin_indices)`` where:
        - ``ecc_bins`` — float array of length ``bins`` with the mean
          normalised eccentricity per bin.
        - ``bin_indices`` — list of index arrays (one per bin) of the active
          nodes that were used to compute each bin.
    """
    n_nodes = graph.vcount()
    if n_nodes == 0:
        return np.zeros(bins, dtype=float), bin_indices if bin_indices is not None else [np.array([], dtype=int)] * bins

    dists = np.array(graph.distances(), dtype=float)
    # Mask out infinity (unreachable paths) by replacing them with -1.0
    finite_dists = np.where(np.isinf(dists), -1.0, dists)
    eccentricities = np.max(finite_dists, axis=1)

    if bin_indices is None:
        node_degrees = np.array(graph.degree(), dtype=int)
        n_active = np.count_nonzero(node_degrees)

        if n_active <= 1:
            return np.zeros(bins, dtype=float), [np.array([], dtype=int)] * bins

        if n_active == n_nodes:
            # Fast-path: all nodes are active
            fixed_active_indices = np.argsort(node_degrees)
        else:
            # Consider only active nodes
            active_mask = node_degrees > 0
            active_indices = np.where(active_mask)[0]
            # Sort active indices by their degree
            fixed_active_indices = active_indices[np.argsort(node_degrees[active_mask])]
        
        # Partition using array_split to handle non-divisible N_active gracefully
        bin_indices = np.array_split(fixed_active_indices, bins)
    else:
        # If provided, we assume it already contains indices of nodes with degree >= 1
        n_active = sum(len(b) for b in bin_indices)

        if n_active <= 1:
            return np.zeros(bins, dtype=float), bin_indices

    ecc_raw = np.array(eccentricities, dtype=float)
    norm_factor = (n_nodes - 1)

    # Calculate mean for each group of nodes using indices
    ecc_bins = np.array([
        ecc_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0 
        for indices in bin_indices
    ], dtype=float)

    return ecc_bins, bin_indices


def count_motifs(graph: ig.Graph, k: int, sampling_probs: list[float] | None = None) -> np.ndarray:
    """Counts the induced occurrences of connected motifs up to size k.
    
    Args:
        graph: The input igraph.Graph.
        k: Maximum motif size to investigate (3, 4, or 5).
        sampling_probs: Optional list of probabilities for sampling at each size 
            from 3 to k. Used to estimate motif counts in large graphs.
    Returns:
        A NumPy array containing the raw count of each valid, connected,
        non-isomorphic motif (one entry per isomorphism class, ordered by
        igraph's isoclass index).
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


def count_motifs_normalized(graph: ig.Graph, k: int = 4) -> np.ndarray:
    """Returns normalised connected-motif frequencies for sizes 3 up to k.

    Each raw motif count is divided by the number of size-*s* node subsets
    ``C(n, s) = n! / (s! * (n-s)!)`` so that the resulting values are in
    ``[0, 1]`` and comparable across graphs of different sizes.  Graphs with
    fewer than ``s`` nodes receive a count of 0 for all size-``s`` motifs.

    Args:
        graph: The input igraph.Graph.
        k: Maximum motif size to count (3, 4, or 5).  Defaults to 4.
    Returns:
        A NumPy float array of length equal to the number of connected
        non-isomorphic undirected motifs up to size *k*.  For ``k=4`` this
        yields 8 values (2 for size-3 + 6 for size-4).
    """
    from math import comb

    raw = count_motifs(graph, k=k)
    n = graph.vcount()

    # Build a normalisation denominator per motif entry, matching the order
    # produced by count_motifs (connected isoclasses, size 3 then size 4 ...).
    denominators: list[float] = []
    for s in range(3, k + 1):
        probe = ig.Graph.Full(s, directed=False)
        num_classes = len(probe.motifs_randesu(size=s))
        denom = float(comb(n, s)) if n >= s else 1.0
        for i in range(num_classes):
            if ig.Graph.Isoclass(s, i, directed=False).is_connected():
                denominators.append(denom)

    denom_arr = np.array(denominators, dtype=float)
    # Avoid division by zero for degenerate graphs
    denom_arr = np.where(denom_arr > 0, denom_arr, 1.0)

    return (raw / denom_arr).astype(float)


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
    """Calculates the structural error between obtained and target degree moments.

    The loss for each moment is ``|arcsinh(actual) - arcsinh(target)| ** 1.5``.
    Skewness is included only when the obtained variance exceeds 1e-12 (to
    avoid numerical instability on near-constant degree sequences).  The final
    scalar is the RMS (L2 norm / sqrt(n)) of the individual moment losses.

    Args:
        obtained_moments: NumPy array ``[mean, variance, skewness, kurtosis]``
            of the generated graph's degree distribution.
        target_moments: NumPy array ``[mean, variance, skewness, kurtosis]``
            of the reference graph's degree distribution.
    Returns:
        A non-negative scalar representing the moment discrepancy.
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


def calculate_annd_error(obtained_annd: np.ndarray, target_annd: np.ndarray) -> float:
    """Calculates the error between the obtained and target binned ANND vectors.

    Uses the element-wise loss ``log1p(40 * |obtained - target| ** 1.5)``
    averaged across all bins.

    Args:
        obtained_annd: NumPy array of binned ANND values for the generated graph.
        target_annd: NumPy array of binned ANND values for the reference graph.
    Returns:
        A non-negative scalar representing the mean ANND discrepancy across bins.
    """    
    loss = np.log1p(40 * np.abs(obtained_annd - target_annd)**1.5)
    error = np.mean(loss)
    
    return float(error)


def calculate_eccentricity_error(obtained_ecc: np.ndarray, target_ecc: np.ndarray) -> float:
    """Calculates the error between the obtained and target binned eccentricity vectors.

    Uses the element-wise loss ``log1p(40 * |obtained - target| ** 1.5)``
    averaged across all bins — identical in structure to :func:`calculate_annd_error`.

    Args:
        obtained_ecc: NumPy array of binned eccentricity values for the generated graph.
        target_ecc: NumPy array of binned eccentricity values for the reference graph.
    Returns:
        A non-negative scalar representing the mean eccentricity discrepancy across bins.
    """    
    loss = np.log1p(40 * np.abs(obtained_ecc - target_ecc)**1.5)
    error = np.mean(loss)
    
    return float(error)