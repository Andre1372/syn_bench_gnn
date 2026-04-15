"""Module defining a centralized manager for ERGM sufficient statistics."""

import itertools

import igraph as ig
import numpy as np

from src.ergm.graph_state import GraphChange, GraphState
from src.graph_analysis import count_motifs
from src.data_utils import igraph_to_networkx, networkx_to_igraph
import networkx as nx


class StatisticsManager:
    """Manager class for ERGM statistics.
    
    All logic is centralized here for maximum efficiency. Individual statistic classes 
    have been removed to reduce overhead and enable shared computations.
    """

    def __init__(self, k: int) -> None:
        """Initializes the manager with a fixed set of supported statistics.

        Args:
            k: The dimension of the motifs (must be 3, 4, or 5).
        """
        if k not in {3, 4, 5}:
            raise ValueError("k must be 3, 4, or 5")

        self._k = k
        self._num_statistics = {3: 2, 4: 8, 5: 29}[self._k]

        self._current_values = np.zeros(self._num_statistics, dtype=float)
        self._cache: dict[int, set[int]] = {}
        self._names = [f"M{i+1}" for i in range(self._num_statistics)]

    @property
    def current_values(self) -> np.ndarray:
        """Returns the current values of all tracked statistics."""
        return self._current_values

    @property
    def names(self) -> list[str]:
        """Returns the names of all tracked statistics."""
        return self._names

    @property
    def num_statistics(self) -> int:
        """Returns the total number of tracked statistics."""
        return self._num_statistics

    def _get_neighborhood(self, graph_state: GraphState, node_idx: int) -> set[int]:
        """Lazy lookup and caching of node neighborhoods."""
        if node_idx not in self._cache:
            self._cache[node_idx] = set(graph_state.neighbors(node_idx))
        return self._cache[node_idx]

    def calculate_initial_values(self, graph_state: GraphState) -> np.ndarray:
        """Calculates initial values for the supported statistics using the motif counting logic."""
        self._current_values = count_motifs(graph_state.sync_igraph(), self._k)
        return self._current_values

    def calculate_deltas(self, graph_state: GraphState, change: GraphChange) -> np.ndarray:
        """Computes deltas for all statistics given a change.
        
        This method uses a virtual state approach: it calculates deltas for each 
        edge in sequence, updating a local cache to reflect the state after each 
        edge event without modifying the physical graph.
        """
        self._cache.clear()
        total_deltas = np.zeros(self._num_statistics, dtype=float)

        for u, v in change.edges_to_remove:
            nu = self._get_neighborhood(graph_state, u)
            nv = self._get_neighborhood(graph_state, v)
            nu.discard(v)
            nv.discard(u)
            self._apply_edge_deltas(u, v, nu, nv, False, total_deltas, graph_state)

        for u, v in change.edges_to_add:
            nu = self._get_neighborhood(graph_state, u)
            nv = self._get_neighborhood(graph_state, v)
            self._apply_edge_deltas(u, v, nu, nv, True, total_deltas, graph_state)
            nu.add(v)
            nv.add(u)

        return total_deltas

    def _apply_edge_deltas(self, u: int, v: int, nu: set[int], nv: set[int], is_addition: bool, total_deltas: np.ndarray, graph_state: GraphState) -> None:
        """Unified calculation of all deltas for a single edge event."""
        sign = 1 if is_addition else -1
        common = nu & nv
        union = nu | nv | {u,v}
        unique_u = nu - nv
        unique_v = nv - nu
        
        if self._k >= 3:
            n_unique = len(unique_u) + len(unique_v)
            n_common = len(common)

            # update deltas
            total_deltas[0] += sign * (n_unique-n_common)
            total_deltas[1] += sign * n_common
        if self._k >= 4:
            n_unique_con, n_unique_disc = self._count_interconnections(unique_u, unique_v, graph_state)
            n_unique_u_con, n_unique_u_disc = self._count_intraconnections(unique_u, graph_state)
            n_unique_v_con, n_unique_v_disc = self._count_intraconnections(unique_v, graph_state)
            n_common_unique_u_con, n_common_unique_u_disc = self._count_interconnections(common, unique_u, graph_state) #
            n_common_unique_v_con, n_common_unique_v_disc = self._count_interconnections(common, unique_v, graph_state) #
            n_common_con, n_common_disc = self._count_intraconnections(common, graph_state) #

            n_2_uniques = 0
            for w in unique_u:
                n_2_uniques += len(self._get_neighborhood(graph_state, w) - union)
            for w in unique_v:
                n_2_uniques += len(self._get_neighborhood(graph_state, w) - union)
            
            n_2_common = 0
            for w in common:
                n_2_common += len(self._get_neighborhood(graph_state, w) - union)

            temp_n_common_unique_uv_disc = n_common_unique_u_disc + n_common_unique_v_disc
            temp_n_common_unique_uv_con = n_common_unique_u_con + n_common_unique_v_con
            
            # update deltas
            total_deltas[2] += sign * (n_unique_u_disc + n_unique_v_disc - n_2_common)
            total_deltas[3] += sign * (n_2_uniques + n_unique_disc - n_unique_con - temp_n_common_unique_uv_disc)
            total_deltas[4] += sign * (temp_n_common_unique_uv_disc + n_2_common + n_unique_u_con + n_unique_v_con - temp_n_common_unique_uv_con)
            total_deltas[5] += sign * (n_unique_con - n_common_disc)
            total_deltas[6] += sign * (temp_n_common_unique_uv_con + n_common_disc - n_common_con)
            total_deltas[7] += sign * (n_common_con)
            
    def _count_intraconnections(self, nodes: set[int], graph_state: GraphState) -> tuple[int, int]:
        """Counts the number of present and missing edges within a set of nodes."""
        N = len(nodes)
        if N < 2:
            return 0, 0
        present_edges = sum(len(self._get_neighborhood(graph_state, n) & nodes) for n in nodes) // 2
        missing_edges = N * (N - 1) // 2 - present_edges
        return present_edges, missing_edges
    
    def _count_interconnections(self, set1: set[int], set2: set[int], graph_state: GraphState) -> tuple[int, int]:
        """Counts the number of present and missing edges between two sets of nodes."""
        N1 = len(set1)
        N2 = len(set2)
        if N1 < N2:
            present_edges = sum(len(self._get_neighborhood(graph_state, n) & set2) for n in set1)
        else:
            present_edges = sum(len(self._get_neighborhood(graph_state, n) & set1) for n in set2)
        missing_edges = N1 * N2 - present_edges
        return present_edges, missing_edges

    def _count_sub_cliques(self, nodes: list[int], target: int, graph_state: GraphState) -> int:
        """Counts k-cliques in a subgraph defined by nodes."""
        if target == 1:
            return len(nodes)
        count = 0
        for i, node_u in enumerate(nodes):
            nu = self._get_neighborhood(graph_state, node_u)
            new_candidates = [w for w in nodes[i+1:] if w in nu]
            if len(new_candidates) >= target - 1:
                count += self._count_sub_cliques(new_candidates, target - 1, graph_state)
        return count

    def update_statistics(self, deltas: np.ndarray) -> None:
        """Applies accepted deltas to current values."""
        self._current_values += deltas


def _generate_null_model_samples(graph: ig.Graph, num_samples: int, rng: np.random.Generator | None = None) -> list[ig.Graph]:
    """Generates a set of null model samples using degree-preserving randomisation."""
    if rng is None:
        rng = np.random.default_rng()
            
    graph_nx = igraph_to_networkx(graph)
    n_edges = graph_nx.number_of_edges()

    null_graphs = []
    for _ in range(num_samples):
        null_nx = graph_nx.copy()
        if n_edges >= 2:
            num_target_swaps = n_edges * 10
            nx.double_edge_swap(
                null_nx, 
                nswap=num_target_swaps, 
                max_tries=num_target_swaps * 100, 
                seed=int(rng.integers(0, 2**31))
            )
        null_graphs.append(networkx_to_igraph(null_nx))

    return null_graphs


def evaluate_significance_profile(graph: ig.Graph, k: int, num_samples: int = 25, rng: np.random.Generator | None = None) -> np.ndarray:
    """Calculates the significance profile (SP) of network motifs up to size k.
    
    The significance profile is derived by comparing the motif counts 
    of the given graph against an ensemble of randomized null graphs. The 
    null models are generated using degree-preserving double edge swaps (pdd).

    Args:
        graph: The target igraph.Graph object to analyze.
        k: Maximum motif size to investigate.
        num_samples: Number of randomized null samples. Default is 25.
        rng: Random number generator (np.random.Generator).
    Returns:
        A NumPy array representing the normalized significance profile of motifs.
    """
    observed_motifs = count_motifs(graph, k=k)
    
    null_graphs = _generate_null_model_samples(graph, num_samples, rng)
    null_motifs = np.array([count_motifs(null_g, k=k) for null_g in null_graphs])
    
    mean_null = np.mean(null_motifs, axis=0)
    std_null = np.std(null_motifs, axis=0)
    
    # Avoid division by zero to ensure numerical stability
    std_null[std_null == 0] = 1e-10
    
    z_scores = (observed_motifs - mean_null) / std_null
    norm = np.linalg.norm(z_scores)
    
    return z_scores / norm if norm > 0 else np.zeros_like(z_scores)