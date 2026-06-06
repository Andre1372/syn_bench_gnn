"""Foundational data structures for graph state management in ANNDG."""

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
    """Maintains the current graph state and supports O(1) local operations.

    Internally the class keeps a canonical edge list alongside an adjacency
    list and a reverse index so that random-edge sampling and edge removal are
    both O(1).  The active nodes are sorted by initial degree and split into
    a fixed set of bins used for ANND aggregation.
    """

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
        
        # Build adjacency list and canonical (min, max) edge list with a
        # reverse index so that edge removal is O(1) via swap-and-pop.
        for u, v in graph.get_edgelist():
            self._adj_list[u].add(v)
            self._adj_list[v].add(u)
            edge = (u, v) if u < v else (v, u)
            if edge not in self._edge_to_idx:
                self._edge_to_idx[edge] = len(self._edges)
                self._edges.append(edge)

        degrees = np.array(graph.degree(), dtype=int)
        self._num_active_nodes: int = int(np.count_nonzero(degrees))
        
        # Fix the binning permutation at initialization so that bin membership
        # is stable throughout the optimization.  Only active nodes (degree > 0)
        # are assigned to bins; isolated nodes are excluded.
        if self._num_active_nodes > 0:
            active_mask = degrees > 0
            active_indices = np.where(active_mask)[0]
            # Sort active nodes by their initial degree so that bins group
            # nodes of similar degree together.
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
        Sample a node uniformly at random from the specified bin in O(1) time.

        When *current_value*, *target_value*, and *knn_normalized* are all
        provided, the sample is restricted to nodes whose individual normalized
        KNN value would move the bin mean in the direction of the target:
        - If ``current_value < target_value``, only nodes with KNN ≤
          ``current_value`` are eligible (they pull the bin average up).
        - If ``current_value >= target_value``, only nodes with KNN ≥
          ``current_value`` are eligible (they pull the bin average down).
        If the resulting candidate set is empty, ``None`` is returned.

        Args:
            bin_idx: The index of the bin to sample from.
            rng: The random number generator to use.
            current_value: Current normalized ANND mean for this bin.
            target_value: Target normalized ANND mean for this bin.
            knn_normalized: Per-node normalized KNN array as returned by
                ``get_annd()``.  Required for conditional filtering.
        Returns:
            The index of the sampled node, or ``None`` if the bin is empty or
            no node satisfies the filtering condition.
        Raises:
            ValueError: If the bin index is out of range.
        """
        if bin_idx < 0 or bin_idx >= self._bins:
            raise ValueError(f"Bin index {bin_idx} out of range [0, {self._bins}).")

        bin_nodes = self._bin_indices[bin_idx]

        # Apply conditional filtering if all three parameters are provided.
        if current_value is not None and target_value is not None and knn_normalized is not None:
            bin_knn = knn_normalized[bin_nodes]
            if current_value < target_value:
                # current < target → restrict to nodes whose KNN is below the
                # current mean; including them in a swap will raise the average.
                mask = bin_knn <= current_value
            else:
                # current > target → restrict to nodes whose KNN is above the
                # current mean; including them in a swap will lower the average.
                mask = bin_knn >= current_value

            bin_nodes = bin_nodes[mask]

        if bin_nodes.size == 0:
            return None

        idx = rng.integers(0, bin_nodes.size)
        return int(bin_nodes[idx])

    def get_annd(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the binned ANND profile and per-node normalized KNN values.

        Returns a pair ``(binned_annd, knn_normalized)`` where:
        - ``binned_annd`` is the per-bin mean of the normalized KNN values
          (shape ``(bins,)``).  Each entry is the mean of
          ``knn_raw[bin_nodes] / (num_nodes - 1)`` over the nodes in that bin.
        - ``knn_normalized`` is the per-node normalized KNN array
          (shape ``(num_nodes,)``), equal to ``knn_raw / (num_nodes - 1)``.

        Returning both values avoids a redundant KNN computation when the
        caller (e.g. ``get_random_node_from_bin``) also needs per-node values.
        """
        if self._num_nodes == 0: return np.zeros(self._bins, dtype=float), np.zeros(0, dtype=float)
        if self._num_active_nodes <= 1: return np.zeros(self._bins, dtype=float), np.zeros(self._num_nodes, dtype=float)

        knn_nodes, _ = self.get_graph().knn()
        knn_raw = np.array(knn_nodes, dtype=float)
        norm_factor = self._num_nodes - 1
        knn_normalized = knn_raw / norm_factor if norm_factor > 0 else knn_raw

        # Per-bin mean: compute knn_raw[bin_nodes].mean() and then normalise.
        binned_annd = np.array([
            knn_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0
            for indices in self._bin_indices
        ], dtype=float)

        return binned_annd, knn_normalized

    def get_eccentricity(self) -> np.ndarray:
        """Compute the normalized, binned eccentricity profile of the graph.

        Eccentricities are derived from the all-pairs shortest-path matrix.
        Unreachable node pairs (infinite distance) are treated as if the
        maximum finite distance were zero, by replacing ``inf`` with ``-1``
        before taking the row-wise maximum — effectively making those nodes
        contribute a negative eccentricity that is still dominated by any
        positive finite distance.  Each bin value is the mean eccentricity of
        its member nodes divided by ``(num_nodes - 1)``.

        Returns:
            Array of shape ``(bins,)`` with the normalized mean eccentricity
            per bin.  Returns an all-zero array when the graph has no nodes or
            fewer than two active nodes.
        """
        if self._num_nodes == 0: return np.zeros(self._bins, dtype=float)
        if self._num_active_nodes <= 1: return np.zeros(self._bins, dtype=float)
        
        dists = np.array(self.get_graph().distances(), dtype=float)
        # Replace inf (unreachable pairs) with -1 so they don't inflate
        # eccentricities for disconnected graphs.
        finite_dists = np.where(np.isinf(dists), -1.0, dists)
        eccentricities = np.max(finite_dists, axis=1)

        ecc_raw = np.array(eccentricities, dtype=float)
        norm_factor = (self._num_nodes - 1)

        # Calculate the mean eccentricity for each bin using the fixed indices.
        return np.array([
            ecc_raw[indices].mean() / norm_factor if indices.size > 0 else 0.0 
            for indices in self._bin_indices
        ], dtype=float)

    def copy(self) -> 'GraphState':
        """Return a deep copy of the current graph state.

        Uses ``object.__new__`` to bypass ``__init__`` and copies each
        internal field directly, avoiding the overhead of rebuilding the
        adjacency structures from an igraph object.
        """
        # Bypass __init__ to avoid redundant validation and reconstruction.
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
        """Apply a graph change, updating all internal data structures.

        Edges are added and removed from the adjacency list, the canonical
        edge list, and the reverse index.  Removal uses swap-and-pop to keep
        the edge list compact and the reverse index consistent.

        Args:
            change: The ``GraphChange`` object defining edges to add and remove.
                Passing ``None`` is a no-op.
        Raises:
            ValueError: If a self-loop is attempted, if an edge to add already
                exists, or if an edge to remove does not exist.
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
                # Swap-and-pop: replace the removed edge's slot with the last edge
                # and update the reverse index accordingly.
                self._edges[idx_to_remove] = last_edge
                self._edge_to_idx[last_edge] = idx_to_remove
            self._edges.pop()

    def revert_change(self, change: GraphChange) -> None:
        """Revert a previously applied graph change.

        Internally builds an inverted ``GraphChange`` (swapping
        ``edges_to_add`` and ``edges_to_remove``) and delegates to
        ``apply_change``.

        Args:
            change: The ``GraphChange`` object to revert.  Passing ``None``
                is a no-op.
        """
        if change is None: return

        # Invert the change by swapping add/remove lists, then reuse apply_change.
        inverted_change = GraphChange(
            edges_to_add=change.edges_to_remove,
            edges_to_remove=change.edges_to_add
        )
        
        self.apply_change(inverted_change)

    def get_graph(self) -> ig.Graph:
        """Returns the current graph as an igraph.Graph object."""
        return ig.Graph(n=self._num_nodes, edges=self._edges, directed=False)
