"""Module defining foundational data structures for graph manipulation."""

from dataclasses import dataclass
import numpy as np
import igraph as ig


@dataclass(frozen=True)
class GraphChange:
    """Represents an atomic transition between two graph states.

    This immutable data transfer object stores the edges that need to be
    added and removed to transition from one state to another.

    Attributes:
        edges_to_add: A list of integer tuples representing the edges to insert.
        edges_to_remove: A list of integer tuples representing the edges to delete.
    """
    edges_to_add: list[tuple[int, int]]
    edges_to_remove: list[tuple[int, int]]


class GraphState:
    """Capsulates the graph state allowing O(1) local operations.

    Provides a fast adjacency list for MCMC steps and a synchronized
    igraph.Graph object on demand.
    """

    def __init__(self, graph: ig.Graph) -> None:
        """Initializes the GraphState.

        Creates a deep copy of the original graph and populates the
        fast adjacency list structure for O(1) lookups.

        Args:
            graph: The initial igraph.Graph object.
        Raises:
            ValueError: If the graph is directed or not simple.
        """
        if graph.is_directed():
            raise ValueError("Only undirected graphs are supported.")
        if not graph.is_simple():
            raise ValueError("Only simple graphs (no multiple edges or self-loops) are supported.")

        self._graph: ig.Graph = graph.copy()
        self._num_nodes: int = self._graph.vcount()
        self._num_edges: int = self._graph.ecount()
        self._pending_changes: list[GraphChange] = []

        self._adj_list: list[set[int]] = [set() for _ in range(self._num_nodes)]
        self._edges: list[tuple[int, int]] = []
        self._edge_to_idx: dict[tuple[int, int], int] = {} # map from edge to index in _edges (used for O(1) edge removal)
        
        for u, v in self._graph.get_edgelist():
            self._adj_list[u].add(v)
            self._adj_list[v].add(u)
            edge = (u, v) if u < v else (v, u)
            if edge not in self._edge_to_idx:
                self._edge_to_idx[edge] = len(self._edges)
                self._edges.append(edge)

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

    def apply_change(self, change: GraphChange) -> None:
        """Applies a proposed graph change to the internal state updating all internal variables.

        Args:
            change: The GraphChange object defining edges to add and remove.
        Raises:
            ValueError: If attempting to add an existing edge or remove a non-existent edge.
        """
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

        self._pending_changes.append(change)

    def sync_igraph(self, threshold: int = 200) -> ig.Graph:
        """Synchronizes the internal igraph cache with all pending changes.

        If the number of pending changes exceeds the ``threshold``, the igraph object
        is reconstructed from the current edge list for better performance. Otherwise,
        changes are applied sequentially.

        Sequential application is required for small change sets because a set-based
        collapse (``added - removed``) would incorrectly cancel edges that were toggled
        an even number of times across MCMC steps, causing ``delete_edges`` to
        reference edges absent from ``self._graph``.

        Args:
            threshold: The maximum number of pending changes before full reconstruction.
        Returns:
            The updated igraph.Graph object.
        """
        if not self._pending_changes:
            return self._graph

        if len(self._pending_changes) > threshold:
            # Full reconstruction is faster for large change sets
            self._graph = ig.Graph(n=self._num_nodes, edges=self._edges, directed=False)
        else:
            # Sequential application for small change sets
            for change in self._pending_changes:
                if change.edges_to_remove:
                    self._graph.delete_edges(change.edges_to_remove)
                if change.edges_to_add:
                    self._graph.add_edges(change.edges_to_add)

        self._pending_changes.clear()
        return self._graph
