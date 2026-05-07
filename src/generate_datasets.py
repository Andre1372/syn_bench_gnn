"""Module for orchestrating the graph generation and evaluation experiment."""

import logging
import multiprocessing as mp
from functools import partial
from typing import Any
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data_utils import (
    DatasetPT,
    preprocess_and_save_original_dataset,
    get_target_stats,
    igraph_to_pytorch,
    networkx_to_igraph,
    igraph_to_networkx,
    pytorch_to_igraph,
    save_synthetic_dataset,
    remove_features,
    compute_log_bin_edges,
    apply_log_bin_features,
    unflatten_stats,
)
from src.graph_analysis import per_graph_statistics, aggregate_statistics
from src.enc_dec_dataset import PercentileEncoderDecoder

from src.padma.graph_generator import generate_graph as padma_generate_graph
from src.ergm.graph_generator import ergm_fit_sample
from src.anndg.graph_generator import generate_graph as anndg_generate_graph

logger = logging.getLogger(__name__)


KNOWN_METHODS: frozenset[str] = frozenset({"padma", "pdd", "ergm", "dummyEdges", "dummyNodes", "anndg", "anndgE", "nextGen"})


def generate_graph(target_stats: dict[str, Any], method: str, rng: np.random.Generator = None) -> tuple[nx.Graph, dict]:
    """Generates a graph using method.

    Args:
        target_stats: Dictionary containing graph statistics:
            - 'n_nodes': Number of nodes.
            - 'n_edges': Number of edges.
            - 'degree_moments': List of moments for PADMA.
            - 'observed_nx': (Optional) The original graph for 'pdd'.
        method: The generating method.
        rng: Random number generator.
    Returns:
        A tuple (G_nx, info) where G_nx is the NetworkX graph and info is a dict.
    Raises:
        ValueError: If the method is not supported or required data is missing.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "nextGen":
        return anndg_generate_graph(target_stats, rng=rng, replicate_eccentricity=True)
    elif method == "padma":
        return padma_generate_graph(target_stats, rng)
    elif method == "anndg":
        return anndg_generate_graph(target_stats, rng=rng, replicate_eccentricity=False)
    elif method == "anndgE":
        return anndg_generate_graph(target_stats, rng=rng, replicate_eccentricity=True)
    elif method == "pdd":
        if "observed_nx" not in target_stats: raise ValueError("Method 'pdd' requires 'observed_nx' in target_stats.")
        
        G_nx = target_stats["observed_nx"].copy()
        n_edges = G_nx.number_of_edges()

        # Edge swaps require at least 2 edges
        if n_edges < 2: return G_nx, {}

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

        return G_nx, {}
    elif method == "ergm":
        # Exponential Random Graph Model estimation and sampling.
        if "observed_nx" not in target_stats:
            raise ValueError("Method 'ergm' requires 'observed_nx' in target_stats.")
        
        # We need a configuration directory or a default config for ERGM.
        # Let's check for a default yaml config in the project root.
        config_path = Path("src/ergm/default_config.yaml")
        if not config_path.exists():
            # Create a basic default config if it doesn't exist
            default_config = {
                "statistics": 4,
                "strategy": {"name": "MultipleEdgeSwapStrategy", "args": {}},
                "estimation": {
                    "updates": 300,
                    "learning_rate": 0.1,
                    "lr_decay": 0.95,
                    "clip_gradient_norm": 10.0,
                    "covariance_update_interval": 5,
                    "covariance_update_alpha": 0.25,
                    "final_samples": 5
                },
                "activation_strategy": {"name": "ErrorThresholdActivationStrategy", "args": {"initial_h": 4}},
                "early_stopping": {"patience": 20, "alpha": 0.75, "acc_rate_target": 0.2}
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            import yaml
            with config_path.open("w") as f:
                yaml.dump(default_config, f)
        
        import yaml
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
        
        # Set experiment name for logging
        experiment_name = f"ergm_gen_{rng.integers(0, 100000)}"
        
        # We use 'padma' for initialisation by default if not specified
        init_method = "padma"

        # project_root is assumed to be current dir in this context.
        project_root = Path(".")
        
        _, synth_igraphs = ergm_fit_sample(
            observed_nx=target_stats["observed_nx"],
            config_dict=config_dict,
            init_method=init_method,
            project_root=project_root,
            experiment_name=experiment_name,
            seed=int(rng.integers(0, 2**31)),
            n_samples=1,
            show_progress=False,
            verbose=False,
            save_results=False
        )
        
        if not synth_igraphs:
            raise RuntimeError("ERGM failed to generate any synthetic samples.")
            
        return igraph_to_networkx(synth_igraphs[0]), {}
    elif method == "dummyNodes":
        # A graph with the same number of nodes as the observed graph, with random number of edges.
        n_nodes = target_stats["n_nodes"]
        n_edges = target_stats["n_edges"]
        p = n_edges / (n_nodes * (n_nodes - 1) / 2) + rng.uniform(-0.05, 0.05)
        p = max(0, min(1, p))
        G_nx = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=int(rng.integers(0, 2**31)))
        return G_nx, {}
    elif method == "dummyEdges":
        # A graph with the same number of nodes and edges as the observed graph.
        n_nodes = target_stats["n_nodes"]
        n_edges = target_stats["n_edges"]
        G_nx = nx.gnm_random_graph(n=n_nodes, m=n_edges, seed=int(rng.integers(0, 2**31)))
        return G_nx, {}
    else:
        raise ValueError(f"Unknown method: {method}. Choose from {', '.join(KNOWN_METHODS)}.")


def generate_synthetic_variants(
    dataset_name: str,
    method: str,
    num_variants: int,
    rng: np.random.Generator,
    project_root: Path,
    output_dir: Path,
    num_workers: int,
    use_distribution_sampling: bool = False,
) -> None:
    """Generates V synthetic variants for a dataset using a given method and saves them to disk.

    For every graph in the TUDataset the function runs the requested generation
    method for each of the ``num_variants`` independent variants.  
    Results are accumulated in-memory variant-by-variant and written to 
    ``output_dir`` as ``.pt`` files via :func:`~src.data_utils.save_synthetic_dataset`.

    Args:
        dataset_name: Name of the TUDataset (e.g. ``"PROTEINS"``).
        method: Generation method.  One of ``'padma'``, ``'pdd'``, ``'dummyEdges'``.
        num_variants: Number of independent synthetic variants ``V`` to produce.
        rng: A seeded (or unseeded) NumPy random generator.
        project_root: Root directory of the project (used to locate configs).
        output_dir: Directory where the variant ``.pt`` files will be written.
            Created automatically if it does not exist.
        num_workers: Number of worker processes for parallel generation.
        use_distribution_sampling: If True, samples stats from the per-class distribution.
    Raises:
        ValueError: If ``method`` is not a recognised generation method.
    """
    if method not in KNOWN_METHODS:
        raise ValueError(
            f"Unknown generation method '{method}'. "
            f"Supported values: {sorted(KNOWN_METHODS)}."
        )

    if use_distribution_sampling and method in ["pdd", "ergm"]:
        raise ValueError(f"Method '{method}' does not support distribution sampling as it requires 'observed_nx'.")

    orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
    if not orig_pt_path.exists():
        preprocess_and_save_original_dataset(dataset_name, project_root / "data")
        
    dataset_obj = DatasetPT(orig_pt_path)
    metadata = dataset_obj.metadata
    
    # original statistics are already saved inside the metadata
    orig_per_graph_stats = metadata.get("per_graph_statistics", [])
    # Whether to apply log-binned degree features (read from the original dataset metadata).
    # If True, each variant will compute its own bin edges after all graphs are generated.
    use_log_bin_deg: bool = metadata.get("use_log_bin_deg", False)
    if use_log_bin_deg:
        logger.info(f"Log-binned degree features enabled for {dataset_name}: bin edges will be computed per-variant after generation.")

    # variant_datasets[v] will hold one PyG Data object per original graph
    variant_datasets: list[list[Data]] = [[] for _ in range(num_variants)]
    variant_seeds:    list[list[int]]  = [[] for _ in range(num_variants)]
    variant_infos:    list[list[dict]] = [[] for _ in range(num_variants)]

    # Pre-generate all seeds to ensure reproducibility and pass them to workers
    # We need a seed for each (graph, variant) pair
    all_seeds = [
        [int(rng.integers(0, 2**31)) for _ in range(num_variants)]
        for _ in range(len(dataset_obj))
    ]

    # Pre-collect all necessary data for workers to avoid passing the whole dataset_obj
    tasks = []
    
    if not use_distribution_sampling:
        for i in range(len(dataset_obj)):
            data = dataset_obj[i]
            target_stats = get_target_stats(dataset_obj, i)
            
            obs_nx = None
            if method == "pdd" or method == "ergm":
                obs_nx = igraph_to_networkx(pytorch_to_igraph(data))
                
            tasks.append({
                'i': i,
                'target_stats': target_stats,
                'y': data.y,
                'obs_nx': obs_nx,
                'seeds': all_seeds[i],
            })
    else:
        # Use distributional sampling
        is_discrete = metadata.get("is_discrete")
        per_class_stats = metadata.get("per_class_stats")
        stat_structure = metadata.get("stat_structure")
        
        if is_discrete is None or per_class_stats is None or stat_structure is None:
            raise ValueError(f"Dataset {dataset_name} is missing distributional metadata. Re-run with --process_original.")
            
        encoder = PercentileEncoderDecoder(num_classes=metadata["num_classes"], is_discrete=is_discrete, rng=rng)
        
        # Encode features for all classes
        for class_id_str, class_info in per_class_stats.items():
            class_id = int(class_id_str)
            stat_matrix = class_info["stat_matrix"]
            encoder.encode_features(stat_matrix, class_id)
            
        for i in range(len(dataset_obj)):
            data = dataset_obj[i]
            y_val = int(data.y.item())
            
            # Sample a single row representing the stats of the i-th synthetic base graph
            sampled_matrix = encoder.sample_features(num_samples=1, class_id=y_val)
            flat_arr = sampled_matrix[0]
            
            # Reconstruct the dictionary format
            target_stats = unflatten_stats(flat_arr, stat_structure)
            
            tasks.append({
                'i': i,
                'target_stats': target_stats,
                'y': data.y,
                'obs_nx': None,
                'seeds': all_seeds[i],
            })

    if num_workers > 1:
        logger.info(f"Generating synthetic variants in parallel using {num_workers} workers...")
        worker_func = partial(
            _worker_generate_variants,
            method=method,
            num_variants=num_variants
        )
        
        # Use a reasonable chunksize for imap
        chunksize = max(1, len(tasks) // (num_workers * 2))

        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_func, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Phase A [{dataset_name}/{method}]"
            ))
            
        # Reconstruct variant_datasets from results
        # results is a list of (graph_idx, list_of_graphs, list_of_seeds)
        results.sort(key=lambda x: x[0])
        for i, graphs, seeds, worker_errors, infos in results:
            for variant_idx, exc_msg in worker_errors:
                logger.error(f"Generation failed for graph {i} variant {variant_idx} (method={method}): {exc_msg}")
            for v in range(num_variants):
                variant_datasets[v].append(graphs[v])
                variant_seeds[v].append(seeds[v])
                variant_infos[v].append(infos[v])
    else:
        with logging_redirect_tqdm():
            pbar = tqdm(tasks, desc=f"Phase A [{dataset_name}/{method}]")
            for task in pbar:
                i, graphs, seeds, worker_errors, infos = _worker_generate_variants(
                    task, method, num_variants
                )
                for variant_idx, exc_msg in worker_errors:
                    logger.error(f"Generation failed for graph {i} variant {variant_idx} (method={method}): {exc_msg}")
                for v in range(num_variants):
                    variant_datasets[v].append(graphs[v])
                    variant_seeds[v].append(seeds[v])
                    variant_infos[v].append(infos[v])

    # Persist each variant to disk
    for v, (graphs, seeds, infos) in enumerate(zip(variant_datasets, variant_seeds, variant_infos)):
        filename = f"{dataset_name}_synth_v{v}.pt"

        # Compute per-variant bin edges and re-encode features if requested.
        # Bin edges are computed from THIS variant's degree distribution, not the original dataset's.
        if use_log_bin_deg:
            variant_bin_edges = compute_log_bin_edges(graphs)
            variant_in_dim = len(variant_bin_edges) - 1
            logger.info(f"Variant {v}: recomputed {variant_in_dim} bins from {len(graphs)} synthetic graphs.")
            re_encoded: list[Data] = []
            for data in graphs:
                g = pytorch_to_igraph(data)
                x = apply_log_bin_features(g, variant_bin_edges)
                re_encoded.append(Data(x=x, edge_index=data.edge_index, y=data.y, num_nodes=data.num_nodes))
            graphs = re_encoded
        else:
            variant_bin_edges = []
            variant_in_dim = 1

        # Map generator-specific info keys to standardized topology metric keys
        precomputed_stats = []
        for info in infos:
            mapped = {}
            if "best_annd" in info: mapped["annd"] = info["best_annd"]
            if "best_eccentricity" in info: mapped["eccentricity"] = info["best_eccentricity"]
            precomputed_stats.append(mapped)

        synth_stats = per_graph_statistics(graphs, precomputed_stats=precomputed_stats, show_progress=False)
        synth_agg = aggregate_statistics(synth_stats)
        
        metadata = {
            "source": method,
            "dataset_name": dataset_name,
            "variant_idx": v,
            "num_variants": num_variants,
            "seeds": seeds,
            "use_log_bin_deg": use_log_bin_deg,
            "bin_edges": variant_bin_edges,
            "in_dim": variant_in_dim,
            "per_graph_statistics": synth_stats,
            "aggregate_statistics": synth_agg,
        }
        
        save_synthetic_dataset(graphs, output_dir, filename, extra_metadata=metadata)
        logger.info(f"Saved variant {v + 1}/{num_variants} for {dataset_name}/{method} → {output_dir / filename}")


def _worker_generate_variants(task, method, num_variants):
    """Worker function for parallel generation."""
    # Ensure workers don't oversubscribe CPUs with internal threading
    torch.set_num_threads(1)
    
    i = task['i']
    target_stats = task['target_stats']
    y = task['y']
    obs_nx = task['obs_nx']
    seeds_list = task['seeds']

    if obs_nx is not None:
        target_stats["observed_nx"] = obs_nx
        
    graphs = []
    seeds = []
    infos = []
    errors = []
    for v in range(num_variants):
        current_seed = seeds_list[v]
        try:
            synth_nx, info = generate_graph(target_stats, method, np.random.default_rng(current_seed))
            synth_ig = networkx_to_igraph(synth_nx)
            synth_pyg = igraph_to_pytorch(synth_ig, y)
            graphs.append(synth_pyg)
            seeds.append(current_seed)
            infos.append(info)
        except Exception as exc:
            errors.append((v, str(exc)))
            # Fallback: minimal 1-node graph with dummy features.
            dummy_pyg = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), y=y, num_nodes=1)
            graphs.append(dummy_pyg)
            seeds.append(-1)
            infos.append({})
            
    return i, graphs, seeds, errors, infos
