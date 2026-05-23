"""Utilities for loading original and synthetic graph datasets."""

import logging
from pathlib import Path
from typing import Any

import torch
import numpy as np
import igraph as ig
import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Load data utilities
# ------------------------------------------------------------------

class DatasetPT(InMemoryDataset):
    """An InMemoryDataset loaded from a .pt file.

    This class wraps a synthetic dataset previously generated and serialized
    by the generation pipeline via `InMemoryDataset.collate`.
    """

    def __init__(self, pt_path: str | Path) -> None:
        """Initializes the dataset by loading a .pt file.

        Args:
            pt_path: Path to the `.pt` file containing a dict with keys
                'data', 'slices', and optionally 'metadata'.
        """
        super().__init__(root=None, transform=None, pre_transform=None)
        
        # Load the serialized payload. weights_only=False is required for PyG objects.
        payload: dict[str, Any] = torch.load(pt_path, weights_only=False)
        
        # Unpack the collated data, slices and metadata
        self.data, self.slices = payload["data"], payload["slices"]
        self.metadata: dict[str, Any] = payload.get("metadata", {})


def get_split_indices(dataset: list[Data], seed: int | None = None) -> tuple[list[int], list[int], list[int]]:
    """Calculates the deterministic train/val/test split indices for a dataset.

    Args:
        dataset: A list of PyG Data objects.
        seed: Random seed for reproducible splitting. If None, no seed is used.
    Returns:
        A tuple containing lists of indices for (train, val, test) splits.
    """
    labels = [int(data.y.item()) for data in dataset if data.y is not None]
    indices = list(range(len(dataset)))

    # First split: 60% for training, 40% for a temporary combined val/test set
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=seed, stratify=labels)
    
    # Extract labels for the temporary set to maintain stratification in the second split
    temp_labels = [labels[i] for i in temp_idx]
    
    # Second split: divide the 40% equally into 20% validation and 20% test
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels)

    return train_idx, val_idx, test_idx


def sample_dataset(dataset: list[Data], max_size: int, rng: np.random.Generator | None = None) -> list[Data]:
    """Samples up to ``max_size`` graphs preserving label and size distributions.

    If the dataset already has at most ``max_size`` graphs it is returned
    unchanged.  Otherwise the sample is built via **stratified** selection:

    1. The per-class quota is proportional to the original class frequencies.
    2. Within each class, candidate graphs are ranked by ascending number of
       nodes. Indices are chosen at uniformly-spaced positions along that
       rank so that the sampled node-count histogram mirrors the full-class
       histogram as closely as possible.

    Args:
        dataset: A list of PyG ``Data`` objects.
        max_size: Maximum number of graphs to keep.
        rng: NumPy random Generator used only to break rank ties deterministically.  If *None* a fresh generator is created.
    Returns:
        A (possibly shorted) list of ``Data`` objects.
    """
    if len(dataset) <= max_size: return dataset

    if rng is None: rng = np.random.default_rng()

    labels = np.array([int(d.y.item()) for d in dataset])
    classes, counts = np.unique(labels, return_counts=True)
    fractions = counts / counts.sum()

    # Compute per-class quotas that sum to max_size
    raw_quotas = fractions * max_size
    quotas = np.floor(raw_quotas).astype(int)
    remainder = max_size - quotas.sum()
    # Distribute remaining slots to classes with largest fractional parts
    fractional_parts = raw_quotas - quotas
    top_classes = np.argsort(-fractional_parts)[:remainder]
    quotas[top_classes] += 1

    selected: list[int] = []
    for cls, quota in zip(classes, quotas):
        cls_indices = np.where(labels == cls)[0]
        if quota >= len(cls_indices):
            selected.extend(cls_indices.tolist())
            continue

        # Sort by num_nodes (add small random jitter to break ties)
        node_counts = np.array([dataset[i].num_nodes for i in cls_indices], dtype=float)
        jitter = rng.uniform(0, 1e-3, size=len(node_counts))
        rank_order = np.argsort(node_counts + jitter)  # ascending node count

        # Pick *quota* evenly-spaced positions along the sorted list
        pick_positions = np.linspace(0, len(rank_order) - 1, quota, dtype=int)
        picked = cls_indices[rank_order[pick_positions]]
        selected.extend(picked.tolist())

    final_fractions = quotas / quotas.sum()

    selected.sort()
    logger.info(
        f"Dataset sampled from {len(dataset)} to {len(selected)} graphs "
        f"{' '.join([f'class {c}: from {fractions[c]:.3f} to {final_fractions[c]:.3f}' for c in classes])}"
    )
    return [dataset[i] for i in selected]


def make_loaders(dataset: list[Data], split_indices: tuple[list[int], list[int], list[int]], batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Splits a dataset into train/val/test loaders based on provided indices.

    Args:
        dataset: A list of PyG Data objects to split.
        split_indices: A tuple of (train_idx, val_idx, test_idx) list of indices.
        batch_size: Number of graphs per batch.
    Returns:
        A tuple of (train_loader, val_loader, test_loader) DataLoaders.
    """
    train_idx, val_idx, test_idx = split_indices

    # Map indices back to Data objects
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    # Create PyG DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def remove_features(data: Data) -> Data:
    """Sets all node features to 1 for a given graph.

    Args:
        data: Input PyG Data object.
    Returns:
        A cloned Data object with node features set to all ones, 
        maintaining the original feature dimensionality.
    """
    num_nodes = data.num_nodes
    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    new_data = Data(
        x=x,
        edge_index=data.edge_index.clone(),
        y=data.y.clone() if data.y is not None else None,
        num_nodes=num_nodes
    )
    return new_data


# ------------------------------------------------------------------
# Log-binned degree feature utilities
# ------------------------------------------------------------------

def compute_log_bin_edges(dataset: list[Data], base: float = 2.0, min_tail_fraction: float = 0.01) -> list[float]:
    """Computes logarithmic bin edges for the global degree distribution of a dataset.

    Bin boundaries follow a geometric progression (``base``^0, ``base``^1, …).
    The algorithm then merges upper bins whose node count falls below
    ``min_tail_fraction`` of the total number of nodes in the dataset,
    preventing near-empty bins for high-degree hubs.

    Args:
        dataset: A list of PyG Data objects from which degrees are collected.
        base: Base of the logarithmic progression. Defaults to 2.
        min_tail_fraction: Minimum fraction of total nodes required in the last bin.
            Bins are merged until this threshold is met. Defaults to 0.01.
    Returns:
        A sorted list of bin-edge values (floats), starting from 0.0.
        The last edge is effectively +inf (represented as ``float('inf')``).
    """
    # Collect all node degrees across the whole dataset
    all_degrees: list[int] = []
    for data in dataset:
        if data.edge_index is not None and data.edge_index.numel() > 0:
            # Count degree from edge_index (undirected: each edge appears twice)
            degrees = torch.zeros(data.num_nodes, dtype=torch.long)
            degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
            all_degrees.extend(degrees.tolist())
        else:
            # Isolated nodes have degree 0
            all_degrees.extend([0] * data.num_nodes)

    total_nodes = len(all_degrees)
    if total_nodes == 0:
        return [0.0, float('inf')]

    max_degree = max(all_degrees)
    min_tail_count = max(1, int(min_tail_fraction * total_nodes))

    # Build candidate edges: 0, 1, base, base^2, ..., first power > max_degree
    edges: list[float] = [0.0]
    power = 0
    while True:
        val = base ** power
        edges.append(float(val))
        if val > max_degree:
            break
        power += 1

    # Merge upper bins until the last bin has enough nodes
    while len(edges) > 2:
        # Count nodes in the last bin [edges[-2], inf)
        last_lower = edges[-2]
        count_last = sum(1 for d in all_degrees if d >= last_lower)
        if count_last >= min_tail_count:
            break
        # Remove the second-to-last edge, effectively merging the two upper bins
        edges.pop(-2)

    # Replace the last finite edge with +inf as a sentinel
    edges[-1] = float('inf')
    logger.debug(
        f"compute_log_bin_edges: {len(edges) - 1} bins, edges={edges[:-1]} + [inf], "
        f"total_nodes={total_nodes}, max_degree={max_degree}"
    )
    return edges


def apply_log_bin_features(g: ig.Graph, bin_edges: list[float]) -> torch.Tensor:
    """Assigns one-hot log-binned degree features to all nodes of an igraph graph.

    Each node's degree is mapped to one of the bins defined by ``bin_edges``
    and encoded as a one-hot vector of length ``len(bin_edges) - 1``.

    Args:
        g: The source igraph Graph.
        bin_edges: Sorted bin boundaries as produced by :func:`compute_log_bin_edges`.
            Must have at least 2 elements.
    Returns:
        A tensor of shape (num_nodes, num_bins) carrying the one-hot feature matrix.
    """
    num_bins = len(bin_edges) - 1
    degrees = torch.tensor(g.degree(), dtype=torch.long)  # shape: (num_nodes,)

    # Map each degree to its bin index using searchsorted on the *right* edges
    # bin_edges = [0, 1, 2, 4, 8, …, inf]; we search within edges[1:]
    upper_edges = torch.tensor(bin_edges[1:], dtype=torch.float32)  # excludes the initial 0
    # searchsorted on upper_edges gives the bin index (0-based)
    bin_indices = torch.bucketize(degrees.float(), upper_edges, right=True).clamp(0, num_bins - 1)

    # One-hot encode
    x = torch.zeros(len(degrees), num_bins, dtype=torch.float32)
    x.scatter_(1, bin_indices.unsqueeze(1), 1.0)
    return x


def flatten_stats(per_graph_statistics: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    """Flattens a dictionary of graph statistics into a 1D NumPy array.

    Args:
        per_graph_statistics: Dictionary containing graph statistics (e.g., n_nodes, n_edges, annd).
    Returns:
        A tuple containing:
            - A 1D float64 NumPy array of all concatenated statistics.
            - A dictionary mapping each key to its size in the flat array.
    """
    parts: list[np.ndarray] = []
    structure: dict[str, int] = {}

    stat_keys: list[str] = ["n_nodes", "n_edges", "degree_moments", "annd", "eccentricity"]
    
    for key in stat_keys:
        val = per_graph_statistics.get(key)
        if val is None:
            structure[key] = 0
            continue

        # Handle both scalars and sequences by converting to at least 1D array
        arr = np.atleast_1d(val).astype(np.float64).flatten()
        parts.append(arr)
        structure[key] = arr.size

    flat_array = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    return flat_array, structure


def unflatten_stats(flat_arr: np.ndarray, structure: dict[str, int]) -> dict[str, Any]:
    """Reconstructs the graph statistics dictionary from a flat array.

    Args:
        flat_arr: A 1D float64 NumPy array containing concatenated statistics.
        structure: A dictionary mapping keys to their original sizes.
    Returns:
        The reconstructed statistics dictionary.
    """
    per_graph_statistics: dict[str, Any] = {}
    current_idx = 0
    
    discrete_keys: set[str] = {"n_nodes", "n_edges"}

    for key, size in structure.items():
        if size == 0:
            per_graph_statistics[key] = None
            continue

        segment = flat_arr[current_idx : current_idx + size]
        if size == 1:
            val = segment[0].item()
            # Cast discrete features back to int
            per_graph_statistics[key] = int(round(val)) if key in discrete_keys else float(val)
        else:
            per_graph_statistics[key] = segment.tolist()

        current_idx += size

    return per_graph_statistics


def get_target_stats(dataset_obj: DatasetPT, idx: int) -> dict[str, Any]:
    """Retrieves target statistics (nodes, edges, moments) for a graph from .pt metadata.
    
    If metadata is unavailable, raises error.
    """
    metadata = dataset_obj.metadata
    orig_per_graph_stats = metadata.get("per_graph_statistics", [])

    if not orig_per_graph_stats:
        raise ValueError(f"No per_graph_statistics found in metadata for dataset {dataset_obj}.")
    
    if idx < len(orig_per_graph_stats):
        pgs = orig_per_graph_stats[idx]
        return {
            "n_nodes": int(pgs.get("n_nodes")),
            "n_edges": int(pgs.get("n_edges")),
            "diameter": pgs.get("diameter"),
            "degree_moments": pgs.get("degree_moments"),
            "annd": pgs.get("annd"),
            "eccentricity": pgs.get("eccentricity"),
        }
    raise ValueError(f"No metadata found for graph {idx}.")


def preprocess_and_save_original_dataset(
    dataset_name: str,
    data_dir: Path,
    max_size: int | None = None,
    rng: np.random.Generator | None = None,
    use_log_bin_deg: bool = False,
    out_filename: str | None = None,
) -> tuple[list[Data], dict[str, Any]]:
    """Preprocesses a TUDataset, applies node features, computes statistics, and saves to .pt.

    If ``max_size`` is set and the dataset is larger, it is down-sampled
    **before** statistics are computed and the file is written.
    
    Args:
        dataset_name: Name of the TUDataset.
        data_dir: Directory where the original dataset will be downloaded and saved.
        max_size: If not None, down-sample to at most this many graphs while preserving label and node/edge-count distributions.
        rng: NumPy Generator forwarded to :func:`sample_dataset`.  If *None* a fresh generator is used when sampling is needed.
        use_log_bin_deg: If ``True``, node features are one-hot log-binned degree vectors
            computed globally over the dataset.  If ``False`` (default), all-ones dummy
            features are used.
        out_filename: Optional filename (including ``.pt`` extension) for the saved file.
            Defaults to ``{dataset_name}_original.pt``.
    Returns:
        The preprocessed (and possibly sampled) list of Data objects and the metadata dictionary.
    """
    _fname = out_filename if out_filename else f"{dataset_name}_original.pt"
    orig_pt_path = data_dir / dataset_name / _fname
    
    feat_tag = "log_bin_deg" if use_log_bin_deg else "dummy"
    logger.info(f"Preprocessing and computing statistics for original dataset: {dataset_name} (features='{feat_tag}')...")
    raw_dataset = TUDataset(root=str(data_dir), name=dataset_name)
    num_classes = raw_dataset.num_classes

    original_data_list = [remove_features(d) for d in raw_dataset]

    # 1. Binarize labels if multi-class using optimal partitioning for balance
    if num_classes > 2:
        all_y = torch.cat([d.y for d in original_data_list if d.y is not None]).view(-1)
        unique_labels, counts = torch.unique(all_y, return_counts=True)
        counts_list = counts.tolist()
        labels_list = unique_labels.tolist()
        
        total_samples = sum(counts_list)
        target = total_samples // 2
        
        # DP to find subset sum closest to target
        reachable = {0: []}
        for i, count in enumerate(counts_list):
            new_reachable = {}
            for s, indices in reachable.items():
                new_s = s + count
                if new_s not in reachable:
                    new_reachable[new_s] = indices + [i]
            reachable.update(new_reachable)
            
        best_sum = min(reachable.keys(), key=lambda s: abs(s - target))
        class1_labels = {labels_list[i] for i in reachable[best_sum]}

        logger.info(
            f"Binarizing {dataset_name} ({num_classes} classes). "
            f"Set 1 labels: {sorted(list(class1_labels))} (size {best_sum}), "
            f"Set 0 labels: {sorted(list(set(labels_list) - class1_labels))} (size {total_samples - best_sum})."
        )
        for d in original_data_list:
            if d.y is not None:
                d.y = torch.tensor([1 if d.y.item() in class1_labels else 0], dtype=torch.long)
        num_classes = 2

    # 2. Sample (down-sample) dataset if needed
    if max_size is not None and len(original_data_list) > max_size:
        original_data_list = sample_dataset(original_data_list, max_size, rng)

    # 3. Compute and apply features
    if use_log_bin_deg:
        # Compute global bin edges from the (binarized and sampled) dataset
        bin_edges = compute_log_bin_edges(original_data_list)
        in_dim = len(bin_edges) - 1
        logger.info(f"Log-binned degree features: {in_dim} bins, edges={bin_edges}")
        # Apply log-bin features to every graph
        final_list = []
        for data in original_data_list:
            g = pytorch_to_igraph(data)
            x = apply_log_bin_features(g, bin_edges)
            edge_index = data.edge_index.clone()
            y = data.y.clone() if data.y is not None else None
            final_list.append(Data(x=x, edge_index=edge_index, y=y, num_nodes=data.num_nodes))
        original_data_list = final_list
    else:
        # Default: dummy (all-ones) features
        bin_edges = []
        in_dim = 1

    from src.graph_analysis import per_graph_statistics, aggregate_statistics
    orig_stats = per_graph_statistics(original_data_list, show_progress=True)
    orig_agg = aggregate_statistics(orig_stats)
    # --- Distributional Sampling Data Extraction ---
    
    if len(orig_stats) > 0:
        _, stat_structure = flatten_stats(orig_stats[0])
    else:
        stat_structure = {}

    is_discrete_list = []
    for k, size in stat_structure.items():
        if k in ["n_nodes", "n_edges"]:
            is_discrete_list.extend([True] * size)
        else:
            is_discrete_list.extend([False] * size)
    is_discrete = np.array(is_discrete_list, dtype=bool)

    per_class_stats = {}
    for data, per_graph_stats in zip(original_data_list, orig_stats):
        y_val = int(data.y.item())

        if y_val not in per_class_stats:
            per_class_stats[y_val] = {"num_samples": 0, "stat_list": []}

        flat_arr, _ = flatten_stats(per_graph_stats)
        per_class_stats[y_val]["num_samples"] += 1
        per_class_stats[y_val]["stat_list"].append(flat_arr)

    for y_val in per_class_stats:
        per_class_stats[y_val]["stat_matrix"] = np.vstack(per_class_stats[y_val].pop("stat_list"))
    
    orig_metadata = {
        "source": "original",
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "use_log_bin_deg": use_log_bin_deg,
        "bin_edges": bin_edges,
        "in_dim": in_dim,
        "per_graph_statistics": orig_stats,
        "aggregate_statistics": orig_agg,
        "is_discrete": is_discrete,
        "per_class_stats": per_class_stats,
        "stat_structure": stat_structure,
    }

    save_synthetic_dataset(
        dataset_list=original_data_list,
        output_dir=orig_pt_path.parent,
        filename=orig_pt_path.name,
        extra_metadata=orig_metadata
    )

    return original_data_list, orig_metadata


def load_all_synthetic_variants(synth_dir: Path, dataset_name: str) -> list[Path]:
    """Scans the directory and returns paths to all synthetic variant datasets.

    Args:
        synth_dir: Directory containing the `.pt` files.
        dataset_name: Name of the original dataset.
    Returns:
        A list of paths to `.pt` files.
    """
    variants: list[Path] = []
    
    # Iteratively find sequential .pt files (v0, v1, ...) until not found
    v = 0
    while True:
        pt_path = synth_dir / f"{dataset_name}_synth_v{v}.pt"
        if not pt_path.exists():
            break

        variants.append(pt_path)
        v += 1
        
    if v == 0:
        logger.warning(f"No synthetic variant files found in {synth_dir}")
        
    return variants


# ------------------------------------------------------------------
# Save data utilities
# ------------------------------------------------------------------

def save_synthetic_dataset(dataset_list: list[Data], output_dir: Path, filename: str | Path, extra_metadata: dict[str, Any] | None = None) -> None:
    """Saves a synthetic dataset (collection of graphs) to a ``.pt`` file.

    Args:
        dataset_list: List of PyG Data objects to save.
        output_dir: Directory where the file will be written.
        filename: Name of the file (including .pt extension).
        extra_metadata: Optional additional metadata embedded in the payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(dataset_list)

    payload = {
        "data": data, 
        "slices": slices, 
        "metadata": extra_metadata if extra_metadata else {}
    }
    
    torch.save(payload, output_dir / filename)
    logger.info(f"Saved synthetic dataset ({len(dataset_list)} graphs) to '{output_dir / filename}'.")


# ------------------------------------------------------------------
# Conversion data utilities
# ------------------------------------------------------------------

def igraph_to_networkx(g: ig.Graph) -> nx.Graph:
    """Converts an :class:`igraph.Graph` to a :class:`networkx.Graph`.

    Args:
        g: The source igraph graph (assumed undirected and simple).
    Returns:
        An equivalent NetworkX graph with the same edge topology.
    """
    nx_graph: nx.Graph = nx.Graph()
    nx_graph.add_nodes_from(range(g.vcount()))
    nx_graph.add_edges_from(g.get_edgelist())
    return nx_graph


def networkx_to_igraph(nx_graph: nx.Graph) -> ig.Graph:
    """Converts a :class:`networkx.Graph` to an :class:`igraph.Graph`.

    Args:
        nx_graph: The source NetworkX graph.
    Returns:
        An equivalent igraph Graph.
    """
    return ig.Graph.from_networkx(nx_graph)


def pytorch_to_igraph(data: Data) -> ig.Graph:
    """Converts a PyG Data object to an igraph Graph.

    Args:
        data: The PyG Data object.
    Returns:
        An undirected igraph Graph instance.
    """
    edge_index = data.edge_index.numpy()
    g = ig.Graph(n=data.num_nodes, edges=list(zip(edge_index[0], edge_index[1])), directed=False)
    g.simplify(multiple=True, loops=True)
    return g


def igraph_to_pytorch(g: ig.Graph, y: torch.Tensor, bin_edges: list[float] | None = None) -> Data:
    """Converts an igraph Graph back to a PyG Data object.

    Args:
        g: The igraph Graph.
        y: Target label for the graph.
        bin_edges: Optional bin boundaries from :func:`compute_log_bin_edges`.  When
            provided, node features are one-hot log-binned degree vectors instead of
            the default all-ones vector.
    Returns:
        A PyG Data object.
    """
    num_nodes = g.vcount()
    edge_list = g.get_edgelist()

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Pytorch Geometric undirected graphs require both [u, v] and [v, u]
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    if bin_edges:
        x = apply_log_bin_features(g, bin_edges)
    else:
        x = torch.ones((num_nodes, 1), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def networkx_to_pytorch(nx_graph: nx.Graph, y: torch.Tensor, bin_edges: list[float] | None = None) -> Data:
    """Converts a NetworkX graph directly to a PyG Data object.

    Args:
        nx_graph: The source NetworkX graph.
        y: Target label for the graph.
        bin_edges: Optional bin boundaries forwarded to :func:`igraph_to_pytorch`.
    Returns:
        A PyG Data object.
    """
    return igraph_to_pytorch(networkx_to_igraph(nx_graph), y, bin_edges=bin_edges)


def pytorch_to_networkx(data: Data) -> nx.Graph:
    """Converts a PyG Data object directly to a NetworkX graph.

    Args:
        data: The source PyG Data object.
    Returns:
        A NetworkX Graph instance.
    """
    return igraph_to_networkx(pytorch_to_igraph(data))
