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
    unflatten_stats,
)
from src.node_features import (
    NodeFeatureAssigner,
    node_feature_assigner_from_metadata,
    is_per_graph_strategy,
    PER_GRAPH_ASSIGNERS,
    sort_source_x_by_neighbor_degree,
)
from src.graph_analysis import per_graph_statistics, aggregate_statistics
from src.enc_dec_dataset import FeatureEncoderDecoder, KNOWN_SAMPLERS
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

    # Early exit for graphs with 1 or 0 nodes (where no simple edges are possible)
    n_nodes = target_stats.get("n_nodes", 0)
    if n_nodes <= 1:
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(n_nodes))
        return G_nx, {}

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


# ---------------------------------------------------------------------------
# Task-building helpers
# ---------------------------------------------------------------------------

def _build_tasks_direct(dataset_obj: "DatasetPT", method: str, num_variants: int, all_seeds: list[list[int]], include_source_features: bool = False, feature_type: str = "random_sample") -> list[dict]:
    """Builds per-graph tasks using the original (per-graph) statistics.

    Args:
        dataset_obj: The loaded original dataset.
        method: Generation method name.
        num_variants: Number of synthetic variants.
        all_seeds: Pre-generated seeds, shape (n_graphs, num_variants).
        include_source_features: If True, store ``data.x`` in the task dict under ``'source_x'`` for per-graph feature transplanting.
        feature_type: The type of feature assignment.
    Returns:
        List of task dicts, one per graph.
    """
    tasks = []
    needs_obs_nx = method in {"pdd", "ergm"}
    for i in range(len(dataset_obj)):
        data = dataset_obj[i]
        target_stats = get_target_stats(dataset_obj, i)
        obs_nx = igraph_to_networkx(pytorch_to_igraph(data)) if needs_obs_nx else None
        
        source_x = data.x
        if include_source_features and source_x is not None:
            # Sort source features by the strategy's composite key before storing.
            # _build_tasks_direct is only called when is_per_graph is True, so
            # feature_type is guaranteed to be in PER_GRAPH_ASSIGNERS.
            if feature_type == "degree_ordered":
                if data.edge_index is not None and data.edge_index.numel() > 0:
                    degrees = torch.zeros(data.num_nodes, dtype=torch.long)
                    degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
                else:
                    degrees = torch.zeros(data.num_nodes, dtype=torch.long)
                sorted_indices = torch.argsort(degrees, descending=True, stable=True)
                source_x = source_x.float()[sorted_indices]
            elif feature_type == "neighbor_degree_ordered":
                source_x = sort_source_x_by_neighbor_degree(data)
            # random_sample: no pre-sorting needed

        tasks.append({
            "i": i,
            "target_stats": [target_stats] * num_variants,
            "y": data.y,
            "obs_nx": obs_nx,
            "seeds": all_seeds[i],
            "source_x": source_x if include_source_features else None,
        })
    return tasks


def _build_tasks_distributional(dataset_obj: DatasetPT, metadata: dict[str, Any], num_variants: int, all_seeds: list[list[int]], encoder: FeatureEncoderDecoder) -> list[dict[str, Any]]:
    """Builds tasks by sampling class-wise statistics for all variants at once.
    
    Args:
        dataset_obj: The loaded baseline dataset.
        metadata: Dataset metadata containing 'per_class_stats' and 'stat_structure'.
        num_variants: The number of synthetic variants to generate.
        all_seeds: Pre-generated random seeds, shape (n_graphs, num_variants).
        encoder: An initialized and pre-fitted FeatureEncoderDecoder.
    Returns:
        A list of task dictionaries containing target stats per variant.
    Raises:
        ValueError: If 'per_class_stats' or 'stat_structure' are missing from the metadata.
    """
    per_class_stats = metadata.get("per_class_stats")
    stat_structure = metadata.get("stat_structure")

    if per_class_stats is None or stat_structure is None:
        raise ValueError("Dataset metadata is missing distributional fields ('per_class_stats', 'stat_structure'). Re-run with --process_original.")

    # Encode features once for all classes
    for class_id_str, class_info in per_class_stats.items():
        encoder.encode_features(class_info["stat_matrix"], int(class_id_str))

    # Group baseline graph indices by class to perform batch-sampling
    from collections import defaultdict
    indices_by_class = defaultdict(list)
    for i in range(len(dataset_obj)):
        class_id = int(dataset_obj[i].y.item())
        indices_by_class[class_id].append(i)

    # Pre-allocate task structures
    tasks = [{
        "i": i,
        "target_stats": [None] * num_variants,
        "y": dataset_obj[i].y,
        "obs_nx": None,
        "seeds": all_seeds[i],
    } for i in range(len(dataset_obj))]

    # Generate complete dataset statistics per variant to maintain full variance
    for v in range(num_variants):
        for class_id, indices in indices_by_class.items():
            num_samples = len(indices)
            sampled_matrix = encoder.sample_features(num_samples=num_samples, class_id=class_id)
            for j, idx in enumerate(indices):
                tasks[idx]["target_stats"][v] = unflatten_stats(sampled_matrix[j], stat_structure)

    return tasks


def _accumulate_result(
    result: tuple,
    method: str,
    num_variants: int,
    variant_datasets: list[list],
    variant_seeds: list[list],
    variant_infos: list[list],
    y: torch.Tensor,
    assigner: NodeFeatureAssigner,
    source_x: torch.Tensor | None = None,
    feature_type: str = "constant",
) -> None:
    """Merges a single worker result into the per-variant accumulators.

    Args:
        result: Tuple (graph_idx, graphs, seeds, errors, infos) from a worker.
        method: Generation method (used for logging).
        num_variants: Number of variants.
        variant_datasets: Accumulator list indexed by variant.
        variant_seeds: Accumulator list indexed by variant.
        variant_infos: Accumulator list indexed by variant.
        y: Target label for the graph.
        assigner: Feature assigner used to build PyG Data from igraph graphs.
        source_x: If provided, a per-graph feature assigner is built from this tensor; overrides *assigner*.
        feature_type: The type of feature strategy (used to select the correct assigner).
    """
    i, graphs, seeds, errors, infos, captured_logs = result
    for log_msg in captured_logs:
        logger.info(f"[Worker {i}] {log_msg}")
    for variant_idx, exc_msg in errors:
        logger.error(f"Generation failed for graph {i} variant {variant_idx} (method={method}): {exc_msg}")
    for v in range(num_variants):
        ig_g = graphs[v]
        if ig_g is None:
            pyg_data = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), y=y, num_nodes=1)
        elif source_x is not None:
            # Build the per-graph assigner from the registry — no if-else needed.
            assigner_cls = PER_GRAPH_ASSIGNERS[feature_type]
            per_graph_assigner = assigner_cls(source_x)
            pyg_data = igraph_to_pytorch(ig_g, y, assigner=per_graph_assigner)
        else:
            pyg_data = igraph_to_pytorch(ig_g, y, assigner=assigner)
        variant_datasets[v].append(pyg_data)
        variant_seeds[v].append(seeds[v])
        variant_infos[v].append(infos[v])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_variants(
    dataset_name: str,
    method: str,
    num_variants: int,
    rng: np.random.Generator,
    project_root: Path,
    output_dir: Path,
    num_workers: int,
    distribution_sampler: str | None = None,
    feature_type: str | None = None,
) -> None:
    """Generates ``num_variants`` synthetic variants for a dataset and saves them to disk.

    For every graph in the preprocessed dataset the function runs the requested
    generation method for each of the ``num_variants`` independent variants.
    Results are accumulated in-memory, then written to ``output_dir`` as ``.pt``
    files via :func:`~src.data_utils.save_synthetic_dataset`.

    The output file name follows the pattern::
        {dataset_name}_{method}_{sampler}_{feature}_v{v}.pt

    Args:
        dataset_name: Name of the TUDataset (e.g. ``"PROTEINS"``).
        method: Generation method — one of the values in :data:`KNOWN_METHODS`.
        num_variants: Number of independent synthetic variants ``V`` to produce.
        rng: A seeded (or unseeded) NumPy random generator.
        project_root: Root directory of the project (used to locate the data).
        output_dir: Directory where the variant ``.pt`` files will be written.
            Created automatically if it does not exist.
        num_workers: Number of worker processes for parallel generation.
        distribution_sampler: Name of the encoder-decoder to use for distributional
            sampling (one of ``'gmcm'``, ``'moments'``, ``'percentile'``). If ``None``,
            per-graph statistics are replicated directly without sampling.
        feature_type: Node feature strategy override. If ``None``, the value is read
            from the preprocessed dataset metadata.

    Raises:
        ValueError: If ``method`` or ``distribution_sampler`` is not recognised, or if
            a sampler is requested for an observation-dependent method.
    """
    if method not in KNOWN_METHODS:
        raise ValueError(f"Unknown generation method '{method}'. Supported values: {sorted(KNOWN_METHODS)}.")
    if distribution_sampler is not None and distribution_sampler not in KNOWN_SAMPLERS:
        raise ValueError(f"Unknown sampler '{distribution_sampler}'. Supported values: {sorted(KNOWN_SAMPLERS)}.")
    if distribution_sampler is not None and method in {"pdd", "ergm"}:
        raise ValueError(f"Method '{method}' requires 'observed_nx' and is incompatible with distribution sampling.")

    effective_feature_type: str = feature_type if feature_type is not None else "constant"

    if is_per_graph_strategy(effective_feature_type):
        orig_fname = f"{dataset_name}_original_native.pt"
    else:
        orig_fname = f"{dataset_name}_original_{effective_feature_type}.pt"

    orig_pt_path = project_root / "data" / dataset_name / orig_fname
    if not orig_pt_path.exists():
        preprocess_and_save_original_dataset(dataset_name, project_root / "data", out_filename=orig_fname, feature_type=effective_feature_type)

    dataset_obj = DatasetPT(orig_pt_path)
    ds_metadata = dataset_obj.metadata
    # If feature_type was None, we sync it with metadata just in case.
    effective_feature_type = feature_type if feature_type is not None else ds_metadata.get("feature_type", "constant")

    if is_per_graph_strategy(effective_feature_type) and distribution_sampler is not None:
        raise ValueError(f"'{effective_feature_type}' features are per-graph and incompatible with distribution_sampler.")

    # For per-graph strategies, features are transplanted during accumulation.
    is_per_graph = is_per_graph_strategy(effective_feature_type)
    if is_per_graph:
        assigner = None
        logger.info(f"Feature strategy for {dataset_name}: {effective_feature_type} (per-graph transplanting).")
    else:
        assigner = node_feature_assigner_from_metadata(ds_metadata)
        logger.info(f"Feature assigner for {dataset_name}: {assigner.__class__.__name__} (in_dim={assigner.in_dim})")

    # Pre-generate one seed per (graph, variant) pair for full reproducibility.
    all_seeds: list[list[int]] = [
        [int(rng.integers(0, 2**31)) for _ in range(num_variants)]
        for _ in range(len(dataset_obj))
    ]

    if distribution_sampler is not None:
        is_discrete = ds_metadata.get("is_discrete")
        if is_discrete is None:
            raise ValueError("Dataset metadata is missing 'is_discrete'. Re-run with --process_original.")
        encoder = KNOWN_SAMPLERS[distribution_sampler](num_classes=ds_metadata["num_classes"], is_discrete=is_discrete, rng=rng)
        tasks = _build_tasks_distributional(dataset_obj, ds_metadata, num_variants, all_seeds, encoder)
    else:
        tasks = _build_tasks_direct(dataset_obj, method, num_variants, all_seeds, include_source_features=is_per_graph, feature_type=effective_feature_type)

    # Accumulators indexed by variant index.
    variant_datasets: list[list[Data]] = [[] for _ in range(num_variants)]
    variant_seeds:    list[list[int]]  = [[] for _ in range(num_variants)]
    variant_infos:    list[list[dict]] = [[] for _ in range(num_variants)]

    worker_func = partial(_worker_generate_variants, method=method, num_variants=num_variants)

    if num_workers > 1:
        logger.info(f"Generating synthetic variants in parallel using {num_workers} workers.")
        chunksize = max(1, len(tasks) // (num_workers * 2))
        _spawn_ctx = mp.get_context("spawn")
        with _spawn_ctx.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(worker_func, tasks, chunksize=chunksize), total=len(tasks), desc=f"Phase A [{dataset_name}/{method}]"))
        results.sort(key=lambda r: r[0])
        for result in results:
            task_i = tasks[result[0]]
            _accumulate_result(result, method, num_variants, variant_datasets, variant_seeds, variant_infos, task_i["y"], assigner, source_x=task_i.get("source_x"), feature_type=effective_feature_type)
    else:
        with logging_redirect_tqdm():
            for task in tqdm(tasks, desc=f"Phase A [{dataset_name}/{method}]"):
                result = worker_func(task)
                _accumulate_result(result, method, num_variants, variant_datasets, variant_seeds, variant_infos, task["y"], assigner, source_x=task.get("source_x"), feature_type=effective_feature_type)

    # Extract target statistics for each variant before saving
    variant_target_stats: list[list[dict]] = [[] for _ in range(num_variants)]
    for task in tasks:
        for v in range(num_variants):
            variant_target_stats[v].append(task["target_stats"][v])

    # Persist each variant to disk
    sampler_tag = distribution_sampler if distribution_sampler is not None else "nosampler"
    for v, (graphs, seeds, infos) in enumerate(zip(variant_datasets, variant_seeds, variant_infos)):
        filename = f"{dataset_name}_{method}_{sampler_tag}_{effective_feature_type}_v{v}.pt"

        # For log-binned features, recompute bin edges from the generated graphs so the encoding is consistent with the variant's own degree distribution.
        from src.node_features import LogBinDegFeatureAssigner
        if is_per_graph:
            # Features already transplanted per-graph; just record the metadata.
            variant_feature_meta = {"feature_type": effective_feature_type, "in_dim": ds_metadata.get("in_dim", 1)}
        elif isinstance(assigner, LogBinDegFeatureAssigner):
            variant_assigner = LogBinDegFeatureAssigner.from_dataset(graphs)
            logger.info(f"Variant {v}: recomputed {variant_assigner.in_dim} bins from {len(graphs)} synthetic graphs.")
            graphs = [variant_assigner.assign_pyg(data) for data in graphs]
            variant_feature_meta = variant_assigner.to_metadata()
        else:
            variant_feature_meta = assigner.to_metadata()

        # Map generator-specific keys to standardised topology metric keys.
        precomputed_stats = [
            {**({"annd": info["best_annd"]} if "best_annd" in info else {}),**({"eccentricity": info["best_eccentricity"]} if "best_eccentricity" in info else {}),}
            for info in infos]

        synth_stats = per_graph_statistics(graphs, precomputed_stats=precomputed_stats, show_progress=False)
        synth_agg = aggregate_statistics(synth_stats)

        variant_metadata = {
            "source": method,
            "dataset_name": dataset_name,
            "variant_idx": v,
            "num_variants": num_variants,
            "seeds": seeds,
            **variant_feature_meta,
            "distribution_sampler": distribution_sampler,
            "per_graph_target_statistics": variant_target_stats[v],
            "per_graph_statistics": synth_stats,
            "aggregate_statistics": synth_agg,
        }
        
        save_synthetic_dataset(graphs, output_dir, filename, extra_metadata=variant_metadata)
        logger.info(f"Saved variant {v + 1}/{num_variants} for {dataset_name}/{method} → {output_dir / filename}")

def _worker_generate_variants(task: dict, method: str, num_variants: int) -> tuple[int, list, list, list, list, list]:
    """Generates all synthetic variants for a single graph (worker entry point).

    Args:
        task: Dict with keys ``'i'``, ``'target_stats'``, ``'y'``, ``'obs_nx'``, ``'seeds'``.
        method: Generation method name.
        num_variants: Number of variants to generate
    Returns:
        Tuple ``(graph_idx, graphs, seeds, errors, infos, captured_logs)``.
    """
    import io
    import logging
    
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    
    try:
        torch.set_num_threads(1)
    
        i: int = task["i"]
        target_stats_list: list[dict] = task["target_stats"]
        y = task["y"]
        obs_nx = task["obs_nx"]
        seeds_list: list[int] = task["seeds"]
    
        graphs, seeds, infos, errors = [], [], [], []
    
        for v in range(num_variants):
            current_seed = seeds_list[v]
            target_stats = dict(target_stats_list[v])
            if obs_nx is not None:
                target_stats["observed_nx"] = obs_nx
            try:
                synth_nx, info = generate_graph(target_stats, method, np.random.default_rng(current_seed))
                synth_ig = networkx_to_igraph(synth_nx)
                graphs.append(synth_ig)
                seeds.append(current_seed)
                infos.append(info)
            except Exception as exc:
                errors.append((v, str(exc)))
                graphs.append(None)
                seeds.append(-1)
                infos.append({})
    finally:
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)
        
    captured_logs = [line for line in log_stream.getvalue().splitlines() if line.strip()]
    return i, graphs, seeds, errors, infos, captured_logs