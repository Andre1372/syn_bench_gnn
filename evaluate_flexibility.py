"""Complete pipeline for evaluating the flexibility of the generative model."""

import argparse
import csv
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.log_utils import setup_console_logging
from src.generate_datasets import generate_synthetic_variants, KNOWN_METHODS
from src.data_utils import preprocess_and_save_original_dataset, DatasetPT
from src.enc_dec_dataset import KNOWN_SAMPLERS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the flexibility of the generative model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and generation
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["all"],
        help="One or more TUDataset names to use as baseline (e.g. PROTEINS MUTAG). Use 'all' (default) to process all datasets in the data/ folder.",
    )
    parser.add_argument(
        "--cut_datasets",
        type=int,
        default=500,
        metavar="N",
        help=(
            "If set, each dataset is down-sampled to at most N graphs before "
            "measuring statistics.  Sampling is stratified by label and "
            "preserves the node/edge-count distribution within each class."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="anndgE",
        metavar="METHOD",
        help=f"Generation methods to run.  Supported: {', '.join(KNOWN_METHODS)}.",
    )
    parser.add_argument(
        "--num_synth_datasets",
        "-S",
        type=int,
        default=200,
        help="Number of independent synthetic datasets to generate.",
    )
    parser.add_argument(
        "--process_original",
        action="store_true",
        help="Pre-process (and sample) the original dataset even if it exists.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="gmcm",
        choices=["moments", "gmcm", "percentile", "percentile_corr"],
        help="The encoder-decoder sampler to use to generate the dataset embedding.",
    )

    # Infrastructure
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help=(
            "RNG seed for synthetic data generation.  "
            "GNN training is always randomised independently.  "
            "Pass -1 for a fully stochastic run."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, int(mp.cpu_count() * 0.9)),
        help="Number of worker processes for parallel generation (Phase A).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    """Writes a list of dicts to a CSV file, creating parent dirs as needed."""
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_dataset_stats_emb(dataset_name: str, num_graphs: int, orig_metadata: dict, sampler_name: str, rng: np.random.Generator,) -> dict:
    """Computes the embedding for the complete dataset (ignoring classes) and returns the row with embedding and non-targeted metrics."""
    orig_stats = orig_metadata.get("per_graph_statistics", [])
    if not orig_stats:
        raise ValueError(f"No per_graph_statistics found in metadata for {dataset_name}.")
        
    if isinstance(orig_stats[0], dict):
        from src.data_utils import flatten_stats
        stat_list = []
        for item in orig_stats:
            flat_arr, _ = flatten_stats(item)
            stat_list.append(flat_arr)
        stat_matrix = np.vstack(stat_list)
    else:
        stat_matrix = np.array(orig_stats)

    is_discrete = np.array(orig_metadata.get("is_discrete", [True, True] + [False] * 12))

    # Get embedding
    encoder = KNOWN_SAMPLERS[sampler_name](num_classes=1, is_discrete=is_discrete, rng=rng)
    encoder.encode_features(stat_matrix, class_id=0)
    dataset_embedding = encoder.get_embedding(class_id=0)
    
    # Construct CSV row
    agg_stats = orig_metadata.get("aggregate_statistics", {})
    row = {
        "Dataset": dataset_name,
        "Num_Graphs": num_graphs,
        "n_nodes_mean": agg_stats.get("n_nodes", 0.0),
    }
    
    # Add embedding elements
    for idx, val in enumerate(dataset_embedding):
        row[f"emb_{idx}"] = val
        
    # Add non-targeted metrics
    row["modularity_mean"] = agg_stats.get("modularity", 0.0)
    row["clustering_mean"] = agg_stats.get("clustering", 0.0)
    row["assortativity_mean"] = agg_stats.get("assortativity", 0.0)
    row["efficiency_mean"] = agg_stats.get("efficiency", 0.0)
    row["diameter_mean"] = agg_stats.get("diameter", 0.0)
    
    return row


def generate_target_statistics(stats_rows: list[dict[str, str | int | float | None]], num_synth_datasets: int, sampler_name: str, rng: np.random.Generator,) -> list[dict[str, str | int | float | None]]:
    """Generates target statistics vectors using asymmetric parent selection, directional
    interpolation/extrapolation, per-dimension Gaussian noise, and semantic+topological validation.

    The four phases are:
    1. Asymmetric Parent Selection: Parent A is sampled uniformly; Parent B is sampled
       proportionally to average node count to pull targets towards larger, structured graphs.
    2. Directional Interpolation / Extrapolation along the A→B vector:
       - 50 % of the time: uniform interpolation in [0, 1] (midpoint exploration).
       - 50 % of the time: exponential-decay extrapolation beyond the A–B segment,
         with equal probability of going below A (t<0) or above B (t>1).
    3. Dimensional Perturbation: Independent zero-mean Gaussian noise added to every
       dimension, with std = 10 % of the absolute coordinate distance |val_b - val_a|.
    4. Dual Validation:
       - Semantic: ``encoder.load_embedding`` verifies that sampler-level invariants
         (non-negative variance, monotone percentiles, etc.) are satisfied.
       - Topological (O(1)): inspects ``encoder._encodings[0]`` directly to estimate
         the maximum reachable n_nodes and avg_degree. Embeddings that could produce
         graphs with more than 500 nodes or more than 1000 edges are discarded.

    Args:
        stats_rows: A list of dictionaries containing mean statistics of original datasets.
        num_synth_datasets: The number of target statistics vectors to generate.
        sampler_name: The name of the sampler being evaluated to enforce its semantic bounds.
        rng: The numpy random generator used to ensure reproducible generation.
    Returns:
        A list of dictionaries representing the generated target dataset statistics/embeddings.
    """
    logger = logging.getLogger(__name__)
    targeted_features = [k for k in stats_rows[0].keys() if k.startswith("emb_")]
    non_targeted_features = ["modularity_mean", "clustering_mean", "assortativity_mean", "efficiency_mean", "diameter_mean"]

    # Convert original statistics to a fast numpy matrix for vectorized math
    features_matrix = np.array([[row[feat] for feat in targeted_features] for row in stats_rows], dtype=np.float64)
    num_graphs = np.array([row["Num_Graphs"] for row in stats_rows], dtype=np.int64)
    node_counts = np.array([row["n_nodes_mean"] for row in stats_rows], dtype=np.float64)

    # Asymmetric Sampling probabilities for Parent B
    prob_b = node_counts / np.sum(node_counts)

    # Initialize the encoder to validate the generated target embeddings
    is_discrete = np.array([True, True] + [False] * 12)
    encoder = KNOWN_SAMPLERS[sampler_name](num_classes=1, is_discrete=is_discrete, rng=rng)

    mean_num_graphs = int(np.mean(num_graphs))
    target_rows = []
    
    target_idx = 0
    attempts = 0
    max_attempts = num_synth_datasets * 100
    
    while len(target_rows) < num_synth_datasets and attempts < max_attempts:
        attempts += 1
        
        # Phase 1: Asymmetric Parent Selection
        # A is chosen uniformly; B is size-weighted to pull toward larger graphs.
        idx_a = rng.choice(len(stats_rows))
        idx_b = rng.choice(len(stats_rows), p=prob_b)

        val_a = features_matrix[idx_a]
        val_b = features_matrix[idx_b]
        direction = val_b - val_a  # directional vector A -> B

        # Phase 2: Directional Interpolation / Extrapolation
        if rng.random() < 0.5:
            # Interpolation: r uniformly in [0, 1] (stay on the A-B segment)
            r_val = rng.uniform(0.0, 1.0)
        else:
            # Extrapolation: exponential decay beyond the A-B boundaries.
            # rate=1 keeps typical |t| ≈ 1; large |t| is exponentially rarer.
            t_exp = rng.exponential(scale=1.0)
            sign  = rng.choice([-1.0, 1.0])
            r_val = sign * t_exp  # can be negative (beyond A) or > 1 (beyond B)

        interpolated = val_a + r_val * direction

        # Clamped r for the n_nodes_mean estimate (node count only makes sense in [0,1])
        r_val_clamped = float(np.clip(r_val, 0.0, 1.0))

        # Phase 3: Dimensional Perturbation
        # Gaussian noise with std = 10 % of the absolute coordinate distance per dimension.
        noise_std = 0.1 * np.abs(direction)
        perturbed = interpolated + rng.normal(0.0, noise_std)

        # Phase 4a: Semantic Validation
        if not encoder.load_embedding(perturbed, class_id=0):
            continue

        # Phase 4b: Topological Validation (O(1) — no graph generated)
        # Inspect encoder._encodings[0] to bound max reachable n_nodes and avg_degree.
        enc = encoder._encodings[0]
        if enc is None:
            continue

        if sampler_name == "moments":
            # enc shape (k, num_features): row 0 = mean, row 1 = variance.
            # 3-sigma rule covers 99.7 % of generated graphs.
            mu_nodes   = float(enc[0, 0])
            mu_degree  = float(enc[0, 1])
            if enc.shape[0] >= 2:
                sigma_nodes  = float(np.sqrt(max(enc[1, 0], 0.0)))
                sigma_degree = float(np.sqrt(max(enc[1, 1], 0.0)))
            else:
                sigma_nodes = sigma_degree = 0.0
            max_nodes  = mu_nodes  + 3.0 * sigma_nodes
            max_degree = mu_degree + 3.0 * sigma_degree
        else:
            # Percentile / GMCM: enc shape (num_percentiles, num_features).
            # Last row = 100th percentile = absolute maximum.
            max_nodes  = float(enc[-1, 0])
            max_degree = float(enc[-1, 1])

        max_edges = (max_nodes * max_degree) / 2.0

        if max_nodes > 500 or max_edges > 1000:
            logger.debug(f"Topological check failed: max_nodes={max_nodes:.1f}, max_edges={max_edges:.1f} — discarding.")
            continue

        # All checks passed — record the target.
        target_row = {
            "Dataset": f"target_{target_idx}",
            "Num_Graphs": mean_num_graphs,
            "n_nodes_mean": float(np.interp(r_val_clamped, [0.0, 1.0], [node_counts[idx_a], node_counts[idx_b]])),
        }
        for j, feat in enumerate(targeted_features):
            target_row[feat] = float(perturbed[j])

        for feat in non_targeted_features:
            target_row[feat] = None

        target_rows.append(target_row)
        target_idx += 1
            
    if len(target_rows) < num_synth_datasets:
        logger.warning(f"Only generated {len(target_rows)} valid target embeddings out of {num_synth_datasets} after {attempts} attempts.")
        
    return target_rows



def worker_generate_graph(task: dict) -> tuple:
    """Generates a single graph for a synthetic dataset (Phase C)."""
    import numpy as np
    import torch
    from src.generate_datasets import generate_graph, networkx_to_igraph
    from src.data_utils import igraph_to_pytorch
    from torch_geometric.data import Data
    
    target_stats = task["target_stats"]
    method = task["method"]
    seed = task["seed"]
    
    rng = np.random.default_rng(seed)
    try:
        synth_nx, info = generate_graph(target_stats, method, rng)
        synth_ig = networkx_to_igraph(synth_nx)
        pyg_data = igraph_to_pytorch(synth_ig, y=torch.tensor([0]))
        return pyg_data, info, True
    except Exception as exc:
        # Fallback for graph generation failure
        import logging
        logging.getLogger(__name__).warning(f"Graph generation failed for method '{method}': {exc}")
        pyg_data = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([0]), num_nodes=1)
        return pyg_data, {}, False




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    project_root = Path(".")

    if args.dataset == ["all"]:
        data_dir = project_root / "data"
        if data_dir.exists():
            args.dataset = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    n_datasets = len(args.dataset)
    session_tag = (
        f"flexibility_{n_datasets}D_{args.method}_{args.num_synth_datasets}S"
        if n_datasets > 1
        else f"flexibility_{args.dataset[0]}_{args.method}_{args.num_synth_datasets}S"
    )
    setup_console_logging(project_root, session_tag)
    logger = logging.getLogger(__name__)
    
    # Reproducibility
    rng = (
        np.random.default_rng()
        if args.seed == -1
        else np.random.default_rng(args.seed)
    )

    # ── PHASE A: Original Dataset Processing ────────────────────────────────────────────
    if not args.process_original:
        for dataset_name in args.dataset:
            orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
            if not orig_pt_path.exists():
                args.process_original = True
                logger.info(f"Dataset {dataset_name} not found, enabling --process_original.")
                break

    stats_rows = []

    if args.process_original:
        logger.info("=" * 60)
        logger.info(f"PHASE A: Processing {n_datasets} dataset(s)")
        logger.info("=" * 60)

        for dataset_name in args.dataset:
            # Preprocess and save (with optional down-sampling).
            # Only re-process if explicitly requested or if the file doesn't exist.
            orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
            original_data_list, orig_metadata = preprocess_and_save_original_dataset(
                dataset_name,
                project_root / "data",
                max_size=args.cut_datasets,
                rng=rng,
            )
            
            # Extract statistics for CSV
            row = extract_dataset_stats_emb(dataset_name, len(original_data_list), orig_metadata, args.sampler, rng)
            stats_rows.append(row)
    else:
        logger.info("Skipping Phase A (dataset processing). Using existing preprocessed datasets.")
        for dataset_name in args.dataset:
            orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
            logger.info(f"Loading metadata for existing dataset {dataset_name}...")
            dataset_obj = DatasetPT(orig_pt_path)
            orig_metadata = dataset_obj.metadata
            num_graphs = len(dataset_obj)
            
            # Extract statistics for CSV
            row = extract_dataset_stats_emb(dataset_name, num_graphs, orig_metadata, args.sampler, rng)
            stats_rows.append(row)

    # Save the aggregated statistics to a CSV in results
    if stats_rows:
        results_dir = project_root / "results"
        csv_path = results_dir / "original_datasets_mean_stats.csv"
        _write_csv(stats_rows, csv_path)
        logger.info(f"Phase A completed. Mean statistics saved to {csv_path}")

    # ── PHASE B: Target Datasets Generation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE B: Generating {args.num_synth_datasets} target statistics vectors")
    logger.info("=" * 60)

    if not stats_rows:
        logger.error("No original dataset statistics found. Cannot compute feature bounds for Phase B.")
        return

    # Choose between the uniform range, SMOTE-inspired, or asymmetric interpolation method
    target_rows = generate_target_statistics(stats_rows, args.num_synth_datasets, args.sampler, rng)

    # Save target statistics to CSV
    results_dir = project_root / "results"
    target_csv_path = results_dir / "target_datasets_mean_stats.csv"
    _write_csv(target_rows, target_csv_path)
    logger.info(f"Phase B completed. Target mean statistics saved to {target_csv_path}")

    # ── PHASE C: Synthetic Datasets Generation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE C: Generating and evaluating synthetic datasets using {args.method}")
    logger.info("=" * 60)

    num_emb_features = sum(1 for k in target_rows[0].keys() if k.startswith("emb_"))
    
    dataset_name = args.dataset[0]
    orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
    dataset_obj = DatasetPT(orig_pt_path)
    orig_metadata = dataset_obj.metadata
    stat_structure = orig_metadata.get("stat_structure")
        
    if stat_structure is None:
        from src.data_utils import flatten_stats
        _, stat_structure = flatten_stats(orig_metadata.get("per_graph_statistics", [])[0])

    synth_rows = []
    
    # Initialize a Pool once for generating the graphs in parallel
    pool = mp.Pool(processes=args.num_workers) if args.num_workers > 1 else None
    
    try:
        from src.data_utils import unflatten_stats, flatten_stats
        from src.graph_analysis import per_graph_statistics, aggregate_statistics
        
        for target_idx, target_row in enumerate(tqdm(target_rows, desc="Phase C (datasets)")):
            # 1. Extract target embedding from row
            target_emb = np.array([target_row[f"emb_{idx}"] for idx in range(num_emb_features)])
            
            # 2. Load target embedding into sampler
            encoder = KNOWN_SAMPLERS[args.sampler](
                num_classes=1, 
                is_discrete=np.array([True, True] + [False] * 12), 
                rng=rng
            )
            encoder.load_embedding(target_emb, class_id=0)
            
            # 3. Sample feature vectors
            sampled_matrix = encoder.sample_features(num_samples=args.cut_datasets, class_id=0)
            
            # 4. Prepare graph generation tasks
            graph_tasks = []
            for i in range(args.cut_datasets):
                row_stats = sampled_matrix[i]
                target_stats = unflatten_stats(row_stats, stat_structure)
                graph_tasks.append({
                    "target_stats": target_stats,
                    "method": args.method,
                    "seed": int(rng.integers(0, 2**31)) if rng is not None else int(np.random.default_rng().integers(0, 2**31)),
                })
                
            # 5. Generate graphs in parallel or sequentially
            graphs = []
            infos = []
            if pool is not None:
                results = pool.map(worker_generate_graph, graph_tasks)
                for pyg_data, info, success in results:
                    graphs.append(pyg_data)
                    infos.append(info)
            else:
                for task in graph_tasks:
                    pyg_data, info, success = worker_generate_graph(task)
                    graphs.append(pyg_data)
                    infos.append(info)
                    
            # 6. Evaluate per-graph statistics
            precomputed_stats = [
                {**({"annd": info["best_annd"]} if "best_annd" in info else {}),
                 **({"eccentricity": info["best_eccentricity"]} if "best_eccentricity" in info else {}),}
                for info in infos
            ]
            synth_stats = per_graph_statistics(graphs, precomputed_stats=precomputed_stats, show_progress=False)
            synth_agg = aggregate_statistics(synth_stats)
            
            # 7. Compute embedding for generated dataset
            synth_flat_list = []
            for item in synth_stats:
                flat_arr, _ = flatten_stats(item)
                synth_flat_list.append(flat_arr)
            synth_matrix = np.vstack(synth_flat_list)
            
            synth_encoder = KNOWN_SAMPLERS[args.sampler](
                num_classes=1, 
                is_discrete=np.array([True, True] + [False] * 12), 
                rng=rng
            )
            synth_encoder.encode_features(synth_matrix, class_id=0)
            synth_emb = synth_encoder.get_embedding(class_id=0)
            
            # 8. Construct CSV row
            row = {
                "Dataset": f"synth_{target_idx}",
                "Num_Graphs": len(graphs),
                "n_nodes_mean": synth_agg.get("n_nodes", 0.0),
            }
            
            # Add embedding elements
            for idx, val in enumerate(synth_emb):
                row[f"emb_{idx}"] = val
                
            # Add non-targeted metrics
            row["modularity_mean"] = synth_agg.get("modularity", 0.0)
            row["clustering_mean"] = synth_agg.get("clustering", 0.0)
            row["assortativity_mean"] = synth_agg.get("assortativity", 0.0)
            row["efficiency_mean"] = synth_agg.get("efficiency", 0.0)
            row["diameter_mean"] = synth_agg.get("diameter", 0.0)
            
            synth_rows.append(row)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Save the synthetic dataset statistics to a CSV in results
    synth_csv_path = results_dir / "synthetic_datasets_mean_stats.csv"
    _write_csv(synth_rows, synth_csv_path)
    logger.info(f"Phase C completed. Synthetic datasets statistics saved to {synth_csv_path}")
    
if __name__ == "__main__":
    main()
