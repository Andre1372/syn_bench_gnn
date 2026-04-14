"""Utilities for loading original and synthetic graph datasets."""

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
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

class SyntheticDataset(InMemoryDataset):
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

    logger.info(f"Splits — train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)} graphs.")

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


def load_single_graph(dataset_name: str, graph_index: int, data_dir: Path | None = None, return_label: bool = False) -> ig.Graph | tuple[ig.Graph, torch.Tensor]:
    """Loads a graph from a TUDataset and sanitizes its topology.

    Downloads the specified dataset (if not cached), extracts the graph at the
    given index, and topologically projects it to a valid simple undirected
    graph by removing self-loops and multiple edges.

    Args:
        dataset_name: The name of the TUDataset to load (e.g., "PROTEINS").
        graph_index: The index of the graph within the dataset.
        data_dir: The directory path for caching the dataset.
        return_label: If True, returns a tuple (graph, label).
    Returns:
        A sanitized ig.Graph object or (ig.Graph, torch.Tensor).
    Raises:
        IndexError: If the graph_index is out of the dataset bounds.
        ValueError: If the selected graph lacks edge indices.
    """
    if data_dir is None: data_dir = Path("data")

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=str(data_dir), name=dataset_name)
    if graph_index < 0 or graph_index >= len(dataset):
        raise IndexError(
            f"Graph index {graph_index} is out of bounds for dataset {dataset_name} "
            f"of size {len(dataset)}."
        )

    data = dataset[graph_index]

    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes

    g = ig.Graph(n=num_nodes, edges=list(zip(edge_index[0], edge_index[1])), directed=False)

    # Sanitize graph geometry: remove self-loops and multi-edges in an optimal way
    g.simplify(multiple=True, loops=True, combine_edges=None)

    return g, data.y if return_label else g


def load_all_synthetic_variants(synth_dir: Path, source_label: str, dataset_name: str) -> list[tuple[list[Data], str, list[int] | None]]:
    """Scans the directory and loads all synthetic variant datasets.

    Args:
        synth_dir: Directory containing the `.pt` files.
        source_label: Label for the data source (e.g., 'padma').
        dataset_name: Name of the original dataset.
    Returns:
        A list of tuples containing:
            - The list of PyG Data objects.
            - The qualified source name (e.g., 'padma_0').
            - The list of seeds (if available, else None).
    """
    variants: list[tuple[list[Data], str, list[int] | None]] = []
    
    # Iteratively find sequential .pt files (v0, v1, ...) until not found
    v = 0
    while True:
        pt_path = synth_dir / f"{dataset_name}_synth_v{v}.pt"
        if not pt_path.exists():
            break

        synth_dataset_obj = SyntheticDataset(pt_path)
        synth_data_list = [synth_dataset_obj[i] for i in range(len(synth_dataset_obj))]
        
        # Try to retrieve seeds from metadata if present
        seeds = synth_dataset_obj.metadata.get("seeds")

        qualified_source = f"{source_label}_{v}"
        variants.append((synth_data_list, qualified_source, seeds))
        
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


def save_synthetic_variants(
    synth_datasets: list[list[Data]],
    seeds: list[list[int]],
    dataset_name: str,
    num_synth_datasets: int,
    method: str,
    project_root: Path,
) -> None:
    """Persists synthetic samples to disk under separate subfolders.

    The output layout is::
        synthetic_data/<dataset_name>/<method>/

    Args:
        synth_datasets: V lists of Data objects from the method generator.
        seeds: V lists of seeds used for each graph.
        dataset_name: Original dataset name.
        num_synth_datasets: Number of variants V.
        method: Method used.
        project_root: Project root directory.
    """
    save_dir = project_root / "synthetic_data" / dataset_name / method

    for v, (synth_list, seed_list) in enumerate(zip(synth_datasets, seeds)):
        filename = f"{dataset_name}_synth_v{v}.pt"
        save_synthetic_dataset(
            synth_list, 
            save_dir, 
            filename,
            extra_metadata={
                "source": method, 
                "dataset_name": dataset_name,
                "variant_idx": v,
                "num_synth_datasets": num_synth_datasets,
                "seeds": seed_list
            }
        )


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
