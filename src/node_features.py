"""Node feature assignment strategies for synthetic graph benchmarking.

Two families of assigners are provided:

* **Global assigners** (``Constant``, ``LogBinDeg``): computed once from the whole
  dataset; serialisable via :meth:`to_metadata` / :meth:`from_metadata`.
* **Per-graph assigners** (``RandomSample``, ``DegreeOrdered``,
  ``NeighborDegreeOrdered``): instantiated fresh for each source graph using the
  graph's own feature matrix; *not* serialisable from metadata alone.

Use the registry helpers to avoid hard-coded ``isinstance`` / ``if feature_type``
chains in downstream code::

    if is_per_graph_strategy(feature_type):
        ...  # sort source_x; build per-graph assigner at accumulation time
    else:
        assigner = node_feature_assigner_from_metadata(metadata)

Or via the factory::

    assigner = node_feature_assigner_from_metadata(metadata)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import igraph as ig
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------

class GlobalFeatureAssigner(ABC):
    """Base for assigners that are dataset-wide and fully serialisable."""

    @property
    @abstractmethod
    def in_dim(self) -> int:
        """Dimensionality of the feature vector assigned to each node."""

    @abstractmethod
    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        """Creates a PyG Data object from an igraph graph."""

    def assign_pyg(self, data: Data) -> Data:
        """Re-assigns node features on an existing PyG Data object."""
        g = _pyg_to_igraph(data)
        return self.assign_igraph(g, y=data.y)

    @abstractmethod
    def to_metadata(self) -> dict[str, Any]:
        """Returns a serialisable dict (must contain ``'feature_type'``)."""

    @classmethod
    @abstractmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "GlobalFeatureAssigner":
        """Reconstructs the assigner from a metadata dict."""


class PerGraphFeatureAssigner(ABC):
    """Base for assigners that depend on a single source graph's feature matrix.

    These assigners are instantiated fresh per source graph and are *not*
    serialisable from metadata alone.
    """

    @property
    @abstractmethod
    def in_dim(self) -> int:
        """Dimensionality of the feature vector assigned to each node."""

    @abstractmethod
    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        """Creates a PyG Data object, transplanting features from ``source_x``."""

    def assign_pyg(self, data: Data) -> Data:
        """Re-assigns features on an existing PyG Data object."""
        g = _pyg_to_igraph(data)
        return self.assign_igraph(g, y=data.y)

    def to_metadata(self) -> dict[str, Any]:
        """Returns minimal metadata (``feature_type`` + ``in_dim`` only)."""
        return {"feature_type": self.FEATURE_TYPE, "in_dim": self.in_dim}  # type: ignore[attr-defined]

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "PerGraphFeatureAssigner":
        raise NotImplementedError(
            f"{cls.__name__} cannot be reconstructed from metadata alone. "
            "Use from_graph() with the original graph's Data object."
        )


# Keep a single public name for type hints in the rest of the codebase.
NodeFeatureAssigner = GlobalFeatureAssigner | PerGraphFeatureAssigner


# ---------------------------------------------------------------------------
# Global assigners
# ---------------------------------------------------------------------------

class ConstantFeatureAssigner(GlobalFeatureAssigner):
    """Assigns a scalar all-ones feature vector to every node (``x = [1.0]``)."""

    FEATURE_TYPE = "constant"

    @property
    def in_dim(self) -> int:
        return 1

    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        num_nodes = g.vcount()
        edge_index = _igraph_edge_index(g, num_nodes)
        x = torch.ones((num_nodes, 1), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    def assign_pyg(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        x = torch.ones((num_nodes, 1), dtype=torch.float32)
        return Data(
            x=x,
            edge_index=data.edge_index.clone(),
            y=data.y.clone() if data.y is not None else None,
            num_nodes=num_nodes,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {"feature_type": self.FEATURE_TYPE, "in_dim": self.in_dim}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "ConstantFeatureAssigner":
        return cls()


class LogBinDegFeatureAssigner(GlobalFeatureAssigner):
    """Assigns one-hot log-binned degree features to every node.

    Each node's degree is mapped to one of the bins defined by :attr:`bin_edges`
    and encoded as a one-hot vector.  Bin edges are logarithmically spaced
    (geometric progression with a given ``base``).

    Args:
        bin_edges: Sorted bin boundaries; last element must be ``float('inf')``.
    """

    FEATURE_TYPE = "log_bin_deg"

    def __init__(self, bin_edges: list[float]) -> None:
        if len(bin_edges) < 2: raise ValueError("bin_edges must have at least 2 elements.")
        self._bin_edges = list(bin_edges)

    @property
    def bin_edges(self) -> list[float]:
        return list(self._bin_edges)

    @property
    def in_dim(self) -> int:
        return len(self._bin_edges) - 1

    @classmethod
    def from_dataset(cls, dataset: list[Data], base: float = 2.0, min_tail_fraction: float = 0.01) -> "LogBinDegFeatureAssigner":
        """Computes bin edges from a reference dataset and returns a fitted assigner."""
        bin_edges = cls._compute_log_bin_edges(dataset, base=base, min_tail_fraction=min_tail_fraction)
        logger.info(f"LogBinDegFeatureAssigner: {len(bin_edges) - 1} bins computed from {sum(d.num_nodes for d in dataset)} nodes. Edges: {bin_edges[:-1]} + [inf]")
        return cls(bin_edges)

    @staticmethod
    def _compute_log_bin_edges(dataset: list[Data], base: float = 2.0, min_tail_fraction: float = 0.01) -> list[float]:
        all_degrees: list[int] = []
        for data in dataset:
            if data.edge_index is not None and data.edge_index.numel() > 0:
                degrees = torch.zeros(data.num_nodes, dtype=torch.long)
                degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
                all_degrees.extend(degrees.tolist())
            else:
                all_degrees.extend([0] * data.num_nodes)

        total_nodes = len(all_degrees)
        if total_nodes == 0:
            return [0.0, float("inf")]

        max_degree = max(all_degrees)
        min_tail_count = max(1, int(min_tail_fraction * total_nodes))

        edges: list[float] = [0.0]
        power = 0
        while True:
            val = base ** power
            edges.append(float(val))
            if val > max_degree:
                break
            power += 1

        while len(edges) > 2:
            last_lower = edges[-2]
            count_last = sum(1 for d in all_degrees if d >= last_lower)
            if count_last >= min_tail_count:
                break
            edges.pop(-2)

        edges[-1] = float("inf")
        return edges

    @staticmethod
    def _apply_log_bin_features(g: ig.Graph, bin_edges: list[float]) -> torch.Tensor:
        num_bins = len(bin_edges) - 1
        degrees = torch.tensor(g.degree(), dtype=torch.long)
        upper_edges = torch.tensor(bin_edges[1:], dtype=torch.float32)
        bin_indices = torch.bucketize(degrees.float(), upper_edges, right=True).clamp(0, num_bins - 1)
        x = torch.zeros(len(degrees), num_bins, dtype=torch.float32)
        x.scatter_(1, bin_indices.unsqueeze(1), 1.0)
        return x

    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        num_nodes = g.vcount()
        edge_index = _igraph_edge_index(g, num_nodes)
        x = self._apply_log_bin_features(g, self._bin_edges)
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    def to_metadata(self) -> dict[str, Any]:
        return {"feature_type": self.FEATURE_TYPE, "bin_edges": self._bin_edges, "in_dim": self.in_dim}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "LogBinDegFeatureAssigner":
        bin_edges = metadata.get("bin_edges")
        if not bin_edges:
            raise ValueError("Metadata does not contain 'bin_edges' for LogBinDegFeatureAssigner.")
        return cls(bin_edges)


# ---------------------------------------------------------------------------
# Per-graph assigners
# ---------------------------------------------------------------------------

class RandomSampleFeatureAssigner(PerGraphFeatureAssigner):
    """Assigns features by sampling without replacement from a source graph's matrix.

    Surplus nodes (when synthetic graph is larger) are filled by sampling
    with replacement from the full pool, preserving the marginal distribution.

    Args:
        source_x: Float tensor of shape ``(N_orig, F)``.
        rng: NumPy Generator; a fresh unseeded one is created if *None*.
    """

    FEATURE_TYPE = "random_sample"

    def __init__(self, source_x: torch.Tensor, rng: np.random.Generator | None = None) -> None:
        if source_x.ndim != 2 or source_x.size(0) == 0:
            raise ValueError(f"source_x must be a non-empty 2-D tensor, got shape {tuple(source_x.shape)}.")
        self._source_x = source_x
        self._rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_graph(cls, data: Data, rng: np.random.Generator | None = None) -> "RandomSampleFeatureAssigner":
        """Builds the assigner from a single graph's node feature matrix."""
        if data.x is None:
            raise ValueError("Source graph has no node features (data.x is None).")
        return cls(data.x.float(), rng=rng)

    @property
    def in_dim(self) -> int:
        return self._source_x.size(1)

    def _sample(self, num_nodes: int) -> torch.Tensor:
        pool_size = self._source_x.size(0)
        if num_nodes <= pool_size:
            idx = self._rng.choice(pool_size, size=num_nodes, replace=False)
        else:
            full_perm = self._rng.permutation(pool_size)
            extra = self._rng.integers(0, pool_size, size=num_nodes - pool_size)
            idx = np.concatenate([full_perm, extra])
        return self._source_x[torch.from_numpy(np.asarray(idx, dtype=np.int64))]

    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        num_nodes = g.vcount()
        edge_index = _igraph_edge_index(g, num_nodes)
        return Data(x=self._sample(num_nodes), edge_index=edge_index, y=y, num_nodes=num_nodes)

    def assign_pyg(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        return Data(
            x=self._sample(num_nodes),
            edge_index=data.edge_index.clone(),
            y=data.y.clone() if data.y is not None else None,
            num_nodes=num_nodes,
        )


class DegreeOrderedFeatureAssigner(PerGraphFeatureAssigner):
    """Assigns pre-sorted features matched by descending degree rank.

    ``source_x`` must already be row-sorted by descending node degree of the
    original graph. The target synthetic graph's nodes are sorted by degree
    (descending) and assigned feature rows top-to-bottom.

    Surplus nodes (synthetic graph larger than source) are clamped to the last
    row of ``source_x`` (lowest-degree representative).

    Args:
        source_x: Float tensor of shape ``(N_orig, F)``, pre-sorted by descending degree.
    """

    FEATURE_TYPE = "degree_ordered"

    def __init__(self, source_x: torch.Tensor) -> None:
        if source_x.ndim != 2 or source_x.size(0) == 0:
            raise ValueError(f"source_x must be a non-empty 2-D tensor, got shape {tuple(source_x.shape)}.")
        self._source_x = source_x

    @classmethod
    def from_graph(cls, data: Data) -> "DegreeOrderedFeatureAssigner":
        """Builds the assigner. ``data.x`` must already be degree-sorted."""
        if data.x is None:
            raise ValueError("Source graph has no node features (data.x is None).")
        return cls(data.x.float())

    @property
    def in_dim(self) -> int:
        return self._source_x.size(1)

    def _assign(self, degrees: torch.Tensor) -> torch.Tensor:
        num_nodes = degrees.size(0)
        pool_size = self._source_x.size(0)
        
        # Sort nodes by degree descending (stable=True keeps original order for ties)
        sorted_indices = torch.argsort(degrees, descending=True, stable=True)
        
        # Indices in source_x to pick from
        source_indices = torch.arange(num_nodes).clamp(max=pool_size - 1)
        
        x = torch.empty((num_nodes, self._source_x.size(1)), dtype=self._source_x.dtype)
        x[sorted_indices] = self._source_x[source_indices]
        return x

    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        num_nodes = g.vcount()
        edge_index = _igraph_edge_index(g, num_nodes)
        degrees = torch.tensor(g.degree(), dtype=torch.long)
        return Data(x=self._assign(degrees), edge_index=edge_index, y=y, num_nodes=num_nodes)

    def assign_pyg(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        if data.edge_index is not None and data.edge_index.numel() > 0:
            degrees = torch.zeros(num_nodes, dtype=torch.long)
            degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
        else:
            degrees = torch.zeros(num_nodes, dtype=torch.long)
        return Data(
            x=self._assign(degrees),
            edge_index=data.edge_index.clone(),
            y=data.y.clone() if data.y is not None else None,
            num_nodes=num_nodes,
        )


class NeighborDegreeOrderedFeatureAssigner(PerGraphFeatureAssigner):
    """Like :class:`DegreeOrderedFeatureAssigner` but uses avg. neighbor degree as tiebreaker.

    Nodes are ranked by ``(degree DESC, avg_neighbor_degree DESC)``. The ordering
    is computed via two consecutive stable sorts (lexsort pattern) — numerically
    safe at any degree scale, unlike a scaled combined key.

    ``source_x`` must already be row-sorted by the same composite key of the
    original graph (use :func:`sort_source_x_by_neighbor_degree` to prepare it).

    Surplus nodes (synthetic graph larger than source) are clamped to the last
    row of ``source_x``.

    Args:
        source_x: Float tensor of shape ``(N_orig, F)``, pre-sorted by descending ``(degree, avg_neighbor_degree)``.
    """

    FEATURE_TYPE = "neighbor_degree_ordered"

    def __init__(self, source_x: torch.Tensor) -> None:
        if source_x.ndim != 2 or source_x.size(0) == 0:
            raise ValueError(f"source_x must be a non-empty 2-D tensor, got shape {tuple(source_x.shape)}.")
        self._source_x = source_x

    @classmethod
    def from_graph(cls, data: Data) -> "NeighborDegreeOrderedFeatureAssigner":
        """Builds the assigner. ``data.x`` must already be (degree, avg_nbr_deg)-sorted."""
        if data.x is None:
            raise ValueError("Source graph has no node features (data.x is None).")
        return cls(data.x.float())

    @property
    def in_dim(self) -> int:
        return self._source_x.size(1)

    # ------------------------------------------------------------------
    # Sort-key computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_neighbor_degree_igraph(g: ig.Graph, degrees: torch.Tensor) -> torch.Tensor:
        """Returns avg neighbour degree per node (0.0 for isolated nodes)."""
        num_nodes = g.vcount()
        avg_nd = torch.zeros(num_nodes, dtype=torch.float32)
        edge_list = g.get_edgelist()
        if edge_list:
            src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
            dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
            nd_sum = torch.zeros(num_nodes, dtype=torch.float32)
            nd_sum.scatter_add_(0, src, degrees[dst].float())
            nd_sum.scatter_add_(0, dst, degrees[src].float())
            avg_nd = nd_sum / degrees.float().clamp(min=1.0)
        return avg_nd

    @staticmethod
    def _avg_neighbor_degree_pyg(edge_index: torch.Tensor, degrees: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Returns avg neighbour degree per node from a PyG edge_index."""
        avg_nd = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_index is not None and edge_index.numel() > 0:
            nd_sum = torch.zeros(num_nodes, dtype=torch.float32)
            nd_sum.scatter_add_(0, edge_index[0], degrees[edge_index[1]].float())
            avg_nd = nd_sum / degrees.float().clamp(min=1.0)
        return avg_nd

    @staticmethod
    def _lexsort_desc(primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        """Stable descending lexsort: primary first, secondary as tiebreaker.

        Uses the two-pass stable-sort pattern: sort by secondary first, then by
        primary. Because both sorts are stable, ties in primary preserve the
        secondary order.  This avoids the ``key + eps * tiebreaker`` float trick
        which is unsafe at large magnitudes.
        """
        idx = torch.argsort(secondary, descending=True, stable=True)
        idx2 = torch.argsort(primary[idx], descending=True, stable=True)
        return idx[idx2]

    def _assign(self, sorted_indices: torch.Tensor) -> torch.Tensor:
        num_nodes = sorted_indices.size(0)
        pool_size = self._source_x.size(0)
        source_indices = torch.arange(num_nodes).clamp(max=pool_size - 1)
        x = torch.empty((num_nodes, self._source_x.size(1)), dtype=self._source_x.dtype)
        x[sorted_indices] = self._source_x[source_indices]
        return x

    def assign_igraph(self, g: ig.Graph, y: torch.Tensor) -> Data:
        num_nodes = g.vcount()
        edge_index = _igraph_edge_index(g, num_nodes)
        degrees = torch.tensor(g.degree(), dtype=torch.long)
        avg_nd = self._avg_neighbor_degree_igraph(g, degrees)
        sorted_indices = self._lexsort_desc(degrees.float(), avg_nd)
        return Data(x=self._assign(sorted_indices), edge_index=edge_index, y=y, num_nodes=num_nodes)

    def assign_pyg(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        if data.edge_index is not None and data.edge_index.numel() > 0:
            degrees = torch.zeros(num_nodes, dtype=torch.long)
            degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
        else:
            degrees = torch.zeros(num_nodes, dtype=torch.long)
        avg_nd = self._avg_neighbor_degree_pyg(data.edge_index, degrees, num_nodes)
        sorted_indices = self._lexsort_desc(degrees.float(), avg_nd)
        return Data(
            x=self._assign(sorted_indices),
            edge_index=data.edge_index.clone(),
            y=data.y.clone() if data.y is not None else None,
            num_nodes=num_nodes,
        )


# ---------------------------------------------------------------------------
# Registry (single source of truth for downstream smistamento)
# ---------------------------------------------------------------------------

GLOBAL_ASSIGNERS: dict[str, type[GlobalFeatureAssigner]] = {
    ConstantFeatureAssigner.FEATURE_TYPE: ConstantFeatureAssigner,
    LogBinDegFeatureAssigner.FEATURE_TYPE: LogBinDegFeatureAssigner,
}

PER_GRAPH_ASSIGNERS: dict[str, type[PerGraphFeatureAssigner]] = {
    RandomSampleFeatureAssigner.FEATURE_TYPE: RandomSampleFeatureAssigner,
    DegreeOrderedFeatureAssigner.FEATURE_TYPE: DegreeOrderedFeatureAssigner,
    NeighborDegreeOrderedFeatureAssigner.FEATURE_TYPE: NeighborDegreeOrderedFeatureAssigner,
}

ALL_FEATURE_TYPES: frozenset[str] = frozenset(GLOBAL_ASSIGNERS) | frozenset(PER_GRAPH_ASSIGNERS)


def is_per_graph_strategy(feature_type: str) -> bool:
    """Returns ``True`` if *feature_type* requires per-graph feature transplanting."""
    return feature_type in PER_GRAPH_ASSIGNERS


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sort_source_x_by_neighbor_degree(data: Data) -> torch.Tensor:
    """Returns ``data.x`` with rows re-ordered by ``(degree DESC, avg_nbr_deg DESC)``.

    This is the preprocessing step that must be applied to the original graph
    before building a :class:`NeighborDegreeOrderedFeatureAssigner`.

    Args:
        data: A PyG Data object with ``x`` and ``edge_index`` set.
    Returns:
        Float tensor of shape ``(N, F)`` sorted by the composite key.
    """
    num_nodes = data.num_nodes
    if data.edge_index is not None and data.edge_index.numel() > 0:
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
        nd_sum = torch.zeros(num_nodes, dtype=torch.float32)
        nd_sum.scatter_add_(0, data.edge_index[0], degrees[data.edge_index[1]].float())
        avg_nd = nd_sum / degrees.float().clamp(min=1.0)
    else:
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        avg_nd = torch.zeros(num_nodes, dtype=torch.float32)

    sorted_indices = NeighborDegreeOrderedFeatureAssigner._lexsort_desc(degrees.float(), avg_nd)
    return data.x.float()[sorted_indices]


def node_feature_assigner_from_metadata(metadata: dict[str, Any]) -> GlobalFeatureAssigner:
    """Reconstructs a :class:`GlobalFeatureAssigner` from a ``.pt`` metadata dict.

    Only global assigners can be reconstructed from metadata. Calling this with
    a per-graph ``feature_type`` raises :class:`ValueError` with a clear message.

    Args:
        metadata: Dict with at least a ``'feature_type'`` key.
    Returns:
        The appropriate :class:`GlobalFeatureAssigner` instance.
    Raises:
        ValueError: If the feature type is unknown or is a per-graph strategy.
    """
    feature_type = metadata.get("feature_type")

    if feature_type in PER_GRAPH_ASSIGNERS:
        raise ValueError(
            f"Feature type '{feature_type}' is a per-graph strategy and cannot be "
            "reconstructed from metadata alone. Use the assigner's from_graph() classmethod."
        )

    if feature_type in GLOBAL_ASSIGNERS:
        return GLOBAL_ASSIGNERS[feature_type].from_metadata(metadata)

    # Legacy compatibility: datasets saved before the feature_type key existed.
    if metadata.get("use_log_bin_deg", False):
        bin_edges = metadata.get("bin_edges")
        if bin_edges:
            logger.debug("node_feature_assigner_from_metadata: using legacy 'bin_edges' key.")
            return LogBinDegFeatureAssigner(bin_edges)

    if feature_type is not None:
        raise ValueError(f"Unknown feature_type '{feature_type}' in metadata.")

    # Default fallback
    return ConstantFeatureAssigner()


# ---------------------------------------------------------------------------
# Private helpers (module-level)
# ---------------------------------------------------------------------------

def _pyg_to_igraph(data: Data) -> ig.Graph:
    """Converts a PyG Data object to an undirected igraph Graph."""
    edge_index = data.edge_index.numpy()
    g = ig.Graph(
        n=data.num_nodes,
        edges=list(zip(edge_index[0], edge_index[1])),
        directed=False,
    )
    g.simplify(multiple=True, loops=True)
    return g


def _igraph_edge_index(g: ig.Graph, num_nodes: int) -> torch.Tensor:
    """Builds an undirected PyG edge_index tensor from an igraph Graph."""
    edge_list = g.get_edgelist()
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return to_undirected(edge_index, num_nodes=num_nodes)
