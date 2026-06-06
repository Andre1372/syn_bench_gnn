"""Module defining foundational data structures for graph manipulation."""

from dataclasses import dataclass
import numpy as np
import igraph as ig


@dataclass(frozen=True)
class GraphChange:
    """Represents an atomic transition between two graph states.

    Attributes:
        edges_to_add: A list of integer tuples representing the edges to insert.
        edges_to_remove: A list of integer tuples representing the edges to delete.
    """
    edges_to_add: list[tuple[int, int]]
    edges_to_remove: list[tuple[int, int]]


class GraphState:
    """Capsulates the graph state allowing O(1) local operations."""

    def __init__(self, graph: ig.Graph, bins: int) -> None:
        """Initializes the GraphState.

        Args:
            graph: The initial igraph.Graph object.
            bins: The number of bins to use for ANND aggregation.
        Raises:
            ValueError: If the graph is directed or not simple.
        """
        if graph.is_directed():
            raise ValueError("Only undirected graphs are supported.")
        if not graph.is_simple():
            raise ValueError("Only simple graphs (no multiple edges or self-loops) are supported.")

        self._num_nodes: int = graph.vcount()
        self._num_edges: int = graph.ecount()

        self._adj_list: list[set[int]] = [set() for _ in range(self._num_nodes)]
        self._edges: list[tuple[int, int]] = []
        self._edge_to_idx: dict[tuple[int, int], int] = {} # map from edge to index in _edges (used for O(1) edge removal)
        
        for u, v in graph.get_edgelist():
            self._adj_list[u].add(v)
            self._adj_list[v].add(u)
            edge = (u, v) if u < v else (v, u)
            if edge not in self._edge_to_idx:
                self._edge_to_idx[edge] = len(self._edges)
                self._edges.append(edge)

        degrees = np.array(graph.degree(), dtype=int)
        self._num_active_nodes: int = int(np.count_nonzero(degrees))
        
        # Fix the binning permutation at initialization for consistency.
        # We only bin nodes that are active (degree > 0).
        if self._num_active_nodes > 0:
            active_mask = degrees > 0
            active_indices = np.where(active_mask)[0]
            # Sort active indices by their initial degree
            self._fixed_active_indices = active_indices[np.argsort(degrees[active_mask])]
        else:
            self._fixed_active_indices = np.array([], dtype=int)
        
        self._bins = bins
        self._bin_indices = np.array_split(self._fixed_active_indices, bins)

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Returns the current number of edges in the graph."""
        return self._num_edges
    
    def has_edge(self, u: int, v: int) -> bool:
        """Checks if an edge exists between two nodes."""
        return v in self._adj_list[u]

    def degree(self, u: int) -> int:
        """Returns the degree of a given node."""
        return len(self._adj_list[u])

    def neighbors(self, u: int) -> frozenset[int]:
        """Returns a safe, read-only view of the neighbors of a node."""
        return frozenset(self._adj_list[u])

    def get_random_edge(self, rng: np.random.Generator) -> tuple[int, int]:
        """Returns a uniformly sampled edge in O(1) time."""
        if not self._edges:
            raise ValueError("Graph has no edges to sample.")
        idx = rng.integers(0, len(self._edges))
        return self._edges[idx]

    def get_random_node_from_bin(
        self,
        bin_idx: int,
        rng: np.random.Generator,
        current_value: float | None = None,
        target_value: float | None = None,
        knn_normalized: np.ndarray | None = None,
    ) -> int | None:
        """
        Samples a node from a specific bin in O(1) time, with optional KNN-based filtering.

        Args:
            bin_idx: The index of the bin to sample from.
            rng: The random number generator to use.
            current_value: Current ANND value for this bin (normalized).
            target_value: Target ANND value for this bin (normalized).
            knn_normalized: Pre-computed per-node normalized KNN values, as returned
                by ``get_annd()``. When provided together with ``current_value`` and
                ``target_value``, avoids a redundant KNN computation.
        Returns:
            The index of the sampled node, or None if the bin is empty or no node satisfies the condition.
        Raises:
            ValueError: If the bin index is out of range.
        """
        if bin_idx < 0 or bin_idx >= self._bins:
            raise ValueError(f"Bin index {bin_idx} out of range [0, {self._bins}).")

        bin_nodes = self._bin_indices[bin_idx]

        # Apply conditional filtering if all required parameters are provided.
        if current_value is not None and target_value is not None and knn_normalized is not None:
            bin_knn = knn_normalized[bin_nodes]
            if current_value < target_value:
                # current < target → sample nodes that pull the average down
                mask = bin_knn <= current_value
            else:
                # current > target → sample nodes that pull the average up
                mask = bin_knn >= current_value

            bin_nodes = bin_nodes[mask]

        if bin_nodes.size == 0:
            return None

        idx = rng.integers(0, bin_nodes.size)
        return int(bin_nodes[idx])

    def get_annd(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes the ANND profile for the graph.

        Returns a pair ``(binned_annd, knn_normalized)`` where:
        - ``binned_annd`` is a per-bin mean of the normalized ANND (shape ``(bins,)``).
        - ``knn_normalized`` is the per-node normalized KNN value (shape ``(num_nodes,)``).

        Returning both avoids a second KNN computation when the caller also needs
        per-node values (e.g. for ``get_random_node_from_bin`` filtering).
        """
        if self._num_nodes == 0: return np.zeros(self._bins, dtype=float), np.zeros(0, dtype=float)
        if self._num_active_nodes <= 1: return np.zeros(self._bins, dtype=float), np.zeros(self._num_nodes, dtype=float)

        knn_nodes, _ = self.get_graph().knn()
        knn_raw = np.array(knn_nodes, dtype=float)
        norm_factor = self._num_nodes - 1
        knn_normalized = knn_raw / norm_factor if norm_factor > 0 else knn_raw

        # Per-bin mean of normalized KNN values
        binned_annd = np.array([
            knn_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0
            for indices in self._bin_indices
        ], dtype=float)

        return binned_annd, knn_normalized

    def get_eccentricity(self) -> np.ndarray:
        """
        Computes the eccentriciy value for each node in the graph, normalizes it and bins it into percentiles.
        """
        if self._num_nodes == 0: return np.zeros(self._bins, dtype=float)
        if self._num_active_nodes <= 1: return np.zeros(self._bins, dtype=float)
        
        dists = np.array(self.get_graph().distances(), dtype=float)
        # Mask out infinity (unreachable paths) by replacing them with -1.0
        finite_dists = np.where(np.isinf(dists), -1.0, dists)
        eccentricities = np.max(finite_dists, axis=1)

        ecc_raw = np.array(eccentricities, dtype=float)
        norm_factor = (self._num_nodes - 1)

        # Calculate mean for each fixed group of nodes using pre-calculated indices
        return np.array([
            ecc_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0 
            for indices in self._bin_indices
        ], dtype=float)

    def copy(self) -> 'GraphState':
        """Creates a deep copy of the current state."""
        # Use simple creation to avoid GraphState.__init__ overhead
        new_state = object.__new__(GraphState)
        new_state._num_nodes = self._num_nodes
        new_state._num_edges = self._num_edges
        new_state._adj_list = [set(adj) for adj in self._adj_list]
        new_state._edges = list(self._edges)
        new_state._edge_to_idx = dict(self._edge_to_idx)
        new_state._num_active_nodes = self._num_active_nodes
        new_state._fixed_active_indices = self._fixed_active_indices.copy()
        new_state._bins = self._bins
        new_state._bin_indices = [idx.copy() for idx in self._bin_indices]
        return new_state

    def apply_change(self, change: GraphChange) -> None:
        """Applies a graph change to the internal state updating all internal variables.

        Args:
            change: The GraphChange object defining edges to add and remove.
        Raises:
            ValueError: If attempting to add an existing edge or remove a non-existent edge.
        """
        if change is None: return

        # Validate consistency before applying
        for u, v in change.edges_to_add:
            if u == v:
                raise ValueError(f"Self-loops are not supported: ({u}, {u}).")
            if self.has_edge(u, v):
                raise ValueError(f"Edge ({u}, {v}) already exists.")
        for u, v in change.edges_to_remove:
            if not self.has_edge(u, v):
                raise ValueError(f"Edge ({u}, {v}) does not exist.")

        # Apply changes
        for u, v in change.edges_to_add:
            self._adj_list[u].add(v)
            self._adj_list[v].add(u)
            self._num_edges += 1
            
            edge = (u, v) if u < v else (v, u)
            self._edge_to_idx[edge] = len(self._edges)
            self._edges.append(edge)

        for u, v in change.edges_to_remove:
            self._adj_list[u].remove(v)
            self._adj_list[v].remove(u)
            self._num_edges -= 1
            
            edge = (u, v) if u < v else (v, u)
            idx_to_remove = self._edge_to_idx.pop(edge)
            last_edge = self._edges[-1]
            if idx_to_remove < len(self._edges) - 1:
                # Update last edge position
                self._edges[idx_to_remove] = last_edge
                self._edge_to_idx[last_edge] = idx_to_remove
            self._edges.pop()

    def revert_change(self, change: GraphChange) -> None:
        """Reverts a previously applied graph change.
        
        Args:
            change: The GraphChange object to revert.
        """
        if change is None: return

        # Simple revert by swapping add/remove
        inverted_change = GraphChange(
            edges_to_add=change.edges_to_remove,
            edges_to_remove=change.edges_to_add
        )
        
        self.apply_change(inverted_change)

    def get_graph(self) -> ig.Graph:
        """Returns the current graph as an igraph.Graph object."""
        return ig.Graph(n=self._num_nodes, edges=self._edges, directed=False)
