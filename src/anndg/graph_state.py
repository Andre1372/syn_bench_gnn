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

    def __init__(self, graph: ig.Graph) -> None:
        """Initializes the GraphState.

        Populates the fast adjacency list structure for O(1) lookups.

        Args:
            graph: The initial igraph.Graph object.
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

        self._node_degrees: np.ndarray = np.array([len(adj) for adj in self._adj_list], dtype=np.int32)
        self._nodes_by_degree: dict[int, np.ndarray] = {}
        for k in np.unique(self._node_degrees):
            self._nodes_by_degree[k] = np.where(self._node_degrees == k)[0]

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

    def get_random_node_by_degree(self, k: int, rng: np.random.Generator) -> int | None:
        """Sample a node uniformly at random among those with degree k."""
        nodes = self._nodes_by_degree.get(k)
        if nodes is None or len(nodes) == 0:
            return None
        
        idx = rng.integers(0, len(nodes))
        return int(nodes[idx])

    def get_neighbor_proportional_to_degree(self, u: int, rng: np.random.Generator, inverse: bool = False) -> int | None:
        """Sample a node v from the neighborhood of u with probability proportional to d(v) (or 1/d(v) if inverse is True)."""
        if not self._adj_list[u]:
            return None
        
        neighbors = list(self._adj_list[u])
        weights = self._node_degrees[neighbors].astype(float)

        if inverse:
            weights = 1.0 / weights
        
        cum_weights = np.cumsum(weights)
        total_weight = cum_weights[-1]
        
        r = rng.random() * total_weight
        idx = np.searchsorted(cum_weights, r)
        
        return int(neighbors[idx])

    def get_annd(self) -> np.ndarray:
        """Calculates the Average Nearest Neighbor Degree (ANND) for each degree.
        
        Returns:
            A numpy array where entry k is the average degree of neighbors of nodes with degree k.
        """
        if self._num_nodes == 0: return np.array([], dtype=float)

        _, knnk = self.get_graph().knn()

        if not knnk or len(knnk) <= 1:
            # This occurs if max_degree is 0 or if the graph is empty.
            return np.array([], dtype=float)

        annd = np.array(knnk, dtype=float)

        # Handle NaNs for degrees present in the range [1, max_k] but absent in the graph.
        return np.nan_to_num(annd, nan=0.0)

    def copy(self) -> 'GraphState':
        """Creates a deep copy of the current state."""
        # Use simple creation to avoid GraphState.__init__ overhead
        new_state = object.__new__(GraphState)
        new_state._num_nodes = self._num_nodes
        new_state._num_edges = self._num_edges
        new_state._adj_list = [set(adj) for adj in self._adj_list]
        new_state._edges = list(self._edges)
        new_state._edge_to_idx = dict(self._edge_to_idx)
        new_state._node_degrees = self._node_degrees
        new_state._nodes_by_degree = self._nodes_by_degree
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
