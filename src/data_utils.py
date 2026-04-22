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
            "normalized_degree_moments": pgs.get("normalized_degree_moments"),
            "annd": pgs.get("annd"),
        }
    raise ValueError(f"No metadata found for graph {idx}.")


def preprocess_and_save_original_dataset(dataset_name: str, data_dir: Path, max_size: int | None = None, rng: np.random.Generator | None = None,) -> tuple[list[Data], dict[str, Any]]:
    """Preprocesses a TUDataset, removes features, computes statistics, and saves to .pt.

    If ``max_size`` is set and the dataset is larger, it is down-sampled
    **before** statistics are computed and the file is written.
    
    Args:
        dataset_name: Name of the TUDataset.
        data_dir: Directory where the original dataset will be downloaded and saved.
        max_size: If not None, down-sample to at most this many graphs while preserving label and node/edge-count distributions.
        rng: NumPy Generator forwarded to :func:`sample_dataset`.  If *None* a fresh generator is used when sampling is needed.
    Returns:
        The preprocessed (and possibly sampled) list of Data objects and the metadata dictionary.
    """
    orig_pt_path = data_dir / dataset_name / f"{dataset_name}_original.pt"
    
    logger.info(f"Preprocessing and computing statistics for original dataset: {dataset_name}...")
    raw_dataset = TUDataset(root=str(data_dir), name=dataset_name)
    num_classes = raw_dataset.num_classes

    # Remove node features completely for topology-only learning
    original_data_list = [remove_features(d) for d in raw_dataset]

    # Down-sample before saving so generate_synthetic_variants reads a cut file
    if max_size is not None and len(original_data_list) > max_size:
        original_data_list = sample_dataset(original_data_list, max_size, rng)

    # Binarize labels if multi-class using optimal partitioning for balance
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

    from src.graph_analysis import per_graph_statistics, aggregate_statistics_per_class
    orig_stats = per_graph_statistics(original_data_list, show_progress=True)
    orig_agg_class = aggregate_statistics_per_class(original_data_list, orig_stats)

    orig_metadata = {
        "source": "original",
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "per_graph_statistics": orig_stats,
        "aggregate_statistics_per_class": orig_agg_class,
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


def igraph_to_pytorch(g: ig.Graph, y: torch.Tensor) -> Data:
    """Converts an igraph Graph back to a PyG Data object.

    Args:
        g: The igraph Graph.
        y: Target label for the graph.
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

    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def networkx_to_pytorch(nx_graph: nx.Graph, y: torch.Tensor) -> Data:
    """Converts a NetworkX graph directly to a PyG Data object.

    Args:
        nx_graph: The source NetworkX graph.
        y: Target label for the graph.
    Returns:
        A PyG Data object.
    """
    return igraph_to_pytorch(networkx_to_igraph(nx_graph), y)


def pytorch_to_networkx(data: Data) -> nx.Graph:
    """Converts a PyG Data object directly to a NetworkX graph.

    Args:
        data: The source PyG Data object.
    Returns:
        A NetworkX Graph instance.
    """
    return igraph_to_networkx(pytorch_to_igraph(data))
