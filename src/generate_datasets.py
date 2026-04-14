"""Module for orchestrating the graph generation and evaluation experiment."""

import logging
from typing import Any
from pathlib import Path
from contextlib import nullcontext
import csv

import igraph as ig
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data_utils import igraph_to_pytorch, networkx_to_igraph, pytorch_to_networkx, save_synthetic_dataset, remove_features
from src.graph_analysis import analyze_single_graph

from src.padma.graph_generator import generate_graph as padma_generate_graph, compute_graph_stats


logger = logging.getLogger(__name__)


KNOWN_METHODS: frozenset[str] = frozenset({"padma", "pdd", "dummy_nodes", "dummy_edges"})


def generate_graph(target_stats: dict[str, Any], method: str, rng: np.random.Generator) -> nx.Graph:
    """Generates a graph using method.

    Args:
        target_stats: Dictionary containing graph statistics:
            - 'n_nodes': Number of nodes.
            - 'n_edges': Number of edges.
            - 'normalized_degree_moments': List of moments for PADMA.
            - 'observed_nx': (Optional) The original graph for 'pdd'.
        method: The generating method.
        rng: Random number generator.
    Returns:
        A synthetic NetworkX graph.
    Raises:
        ValueError: If the method is not supported or required data is missing.
    """
    if method == "padma":
        # Probabilistic Annealing for Degree Moments Alignment.
        G_nx, _ = padma_generate_graph(target_stats, rng)
        return G_nx
    elif method == "pdd":
        # Preserving degree distribution: using double edge swaps.
        if "observed_nx" not in target_stats: raise ValueError("Method 'pdd' requires 'observed_nx' in target_stats.")
        
        G_nx = target_stats["observed_nx"].copy()
        n_edges = G_nx.number_of_edges()

        # Edge swaps require at least 2 edges
        if n_edges < 2: return G_nx

        num_target_swaps = n_edges * 10
        try:
            nx.double_edge_swap(
                G_nx,
                nswap=num_target_swaps,
                max_tries=num_target_swaps * 100,
                seed=int(rng.integers(0, 2**31))
            )
        except Exception as e:
            logger.warning(
                "nx.double_edge_swap failed for pdd (partially randomized graph). "
                "Swaps requested: %d, Error: %s", num_target_swaps, e
            )

        return G_nx
    elif method == "dummy_nodes":
        # A graph with the same number of nodes as the observed graph, with random number of edges.
        n_nodes = target_stats["n_nodes"]
        p = rng.uniform(0, 0.5)
        G_nx = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=int(rng.integers(0, 2**31)))
        return G_nx
    elif method == "dummy_edges":
        # A graph with the same number of nodes and edges as the observed graph.
        n_nodes = target_stats["n_nodes"]
        n_edges = target_stats["n_edges"]
        G_nx = nx.gnm_random_graph(n=n_nodes, m=n_edges, seed=int(rng.integers(0, 2**31)))
        return G_nx
    else:
        raise ValueError(f"Unknown method: {method}. Choose from {', '.join(KNOWN_METHODS)}.")


def generate_synthetic_variants(
    dataset_name: str,
    method: str,
    num_variants: int,
    rng: np.random.Generator,
    project_root: Path,
    output_dir: Path,
) -> None:
    """Generates V synthetic variants for a dataset using a given method and saves them to disk.

    For every graph in the TUDataset the function runs the requested generation
    method for each of the ``num_variants`` independent variants.  
    Results are accumulated in-memory variant-by-variant and written to 
    ``output_dir`` as ``.pt`` files via :func:`~src.data_utils.save_synthetic_dataset`.

    Args:
        dataset_name: Name of the TUDataset (e.g. ``"PROTEINS"``).
        method: Generation method.  One of ``'padma'``, ``'pdd'``, ``'dummy_nodes'``, ``'dummy_edges'``.
        num_variants: Number of independent synthetic variants ``V`` to produce.
        rng: A seeded (or unseeded) NumPy random generator.
        project_root: Root directory of the project (used to locate configs).
        output_dir: Directory where the variant ``.pt`` files will be written.
            Created automatically if it does not exist.
    Raises:
        ValueError: If ``method`` is not a recognised generation method.
    """
    if method not in KNOWN_METHODS:
        raise ValueError(
            f"Unknown generation method '{method}'. "
            f"Supported values: {sorted(KNOWN_METHODS)}."
        )

    dataset = TUDataset(root=str(project_root / "data"), name=dataset_name)

    # variant_datasets[v] will hold one PyG Data object per original graph
    variant_datasets: list[list[Data]] = [[] for _ in range(num_variants)]
    variant_seeds:    list[list[int]]  = [[] for _ in range(num_variants)]

    with logging_redirect_tqdm():
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Phase A [{dataset_name}/{method}]")
        for i, data in pbar:
            obs_nx = pytorch_to_networkx(data)
            
            # Precompute target statistics
            target_stats = compute_graph_stats(obs_nx)
            if method == "pdd":
                target_stats["observed_nx"] = obs_nx

            for v in range(num_variants):
                current_seed = int(rng.integers(0, 2**31))
                try:
                    synth_nx = generate_graph(target_stats, method, np.random.default_rng(current_seed))
                    synth_ig = networkx_to_igraph(synth_nx)
                    synth_pyg = igraph_to_pytorch(synth_ig, data.y)

                    variant_datasets[v].append(synth_pyg)
                    variant_seeds[v].append(current_seed)

                except Exception as exc:
                    logger.error(f"Generation failed for graph {i} variant {v} (method={method}): {exc}")
                    variant_datasets[v].append(remove_features(data))
                    variant_seeds[v].append(-1)

    # Persist each variant to disk
    for v, (graphs, seeds) in enumerate(zip(variant_datasets, variant_seeds)):
        filename = f"{dataset_name}_synth_v{v}.pt"
        save_synthetic_dataset(
            dataset_list=graphs,
            output_dir=output_dir,
            filename=filename,
            extra_metadata={
                "source": method,
                "dataset_name": dataset_name,
                "variant_idx": v,
                "num_variants": num_variants,
                "seeds": seeds,
            },
        )
        logger.info(f"Saved variant {v + 1}/{num_variants} for {dataset_name}/{method} → {output_dir / filename}")
