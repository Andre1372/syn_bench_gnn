"""Module defining foundational data structures for graph manipulation."""

from dataclasses import dataclass
import numpy as np
import igraph as ig
from collections import deque


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

        self._degrees = np.array(graph.degree(), dtype=int)

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Returns the current number of edges in the graph."""
        return self._num_edges
    
    @property
    def exact_diameter(self) -> int:
        """
        Returns the exact diameter of the graph using igraph.
        For disconnected graphs, returns the diameter of the largest connected component.
        """
        if self._num_nodes <= 1 or self._num_edges == 0:
            return 0
        d = self.get_graph().diameter(directed=False)
        # Avoid inf if possible, though igraph usually returns max component diameter
        return int(d) if d != float('inf') else 0

    @property
    def approximate_diameter(self) -> int:
        """
        Returns the approximate diameter of the graph using a two-sweep BFS.
        This provides a lower bound and is exact for trees. For general graphs, 
        it is highly efficient but may under-estimate the true diameter by a small margin.
        """
        if self._num_nodes <= 1 or self._num_edges == 0:
            return 0
        
        visited_global = [False] * self._num_nodes
        max_diam = 0
        
        def bfs_furthest(start_node: int) -> tuple[int, int]:
            distances = [-1] * self._num_nodes
            distances[start_node] = 0
            queue = deque([start_node])
            furthest_node = start_node
            max_dist = 0
            
            while queue:
                curr = queue.popleft()
                visited_global[curr] = True
                curr_dist = distances[curr]
                
                for neighbor in self._adj_list[curr]:
                    if distances[neighbor] == -1:
                        d = curr_dist + 1
                        distances[neighbor] = d
                        queue.append(neighbor)
                        if d > max_dist:
                            max_dist = d
                            furthest_node = neighbor
                            
            return furthest_node, max_dist

        for i in range(self._num_nodes):
            if not visited_global[i]:
                # 1st sweep: find a peripheral node
                u, _ = bfs_furthest(i)
                # 2nd sweep: find the distance from that peripheral node
                _, comp_diam = bfs_furthest(u)
                
                if comp_diam > max_diam:
                    max_diam = comp_diam
                    
        return max_diam

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

    def get_annd(self, bins:int = 5) -> np.ndarray:
        """
        Computes the Average Nearest Neighbor Degree (ANND) of the graph, normalizes it and bins it into percentiles.

        ANND characterizes degree-degree correlations. The function normalizes neighbor degrees 
        by the maximum possible degree (N-1) and aggregates nodes into 'bins' percentile groups 
        based on their degree rank.

        Args:
            bins: The number of percentile bins to aggregate into.
        Returns:
            np.ndarray: Array of length `bins` containing the mean normalized ANND per bin.
        """
        if self._num_nodes == 0: return np.zeros(bins, dtype=float)

        knn_nodes, _ = self.get_graph().knn()

        if not knn_nodes: return np.zeros(bins, dtype=float)

        # Normalize by (N-1) and handle NaNs from isolated nodes
        annd_raw = np.array(knn_nodes, dtype=float)
        annd_norm = np.nan_to_num(annd_raw / (self._num_nodes - 1), nan=0.0)

        # Sort ANND by degree rank to enable grouping into percentile bins
        sorted_annd = annd_norm[np.argsort(self._degrees)]

        # Partition and average using array_split to handle non-divisible N_nodes gracefully
        return np.array([
            group.mean() if group.size > 0 else 0.0 
            for group in np.array_split(sorted_annd, bins)
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
        new_state._degrees = self._degrees.copy()
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
