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
        default=max(1, int(mp.cpu_count() * 0.5)),
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


def _append_csv(rows: list[dict], path: Path) -> None:
    """Appends rows to an existing CSV, or creates it if absent."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_csv(rows, path)
        return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Ordered names that mirror TARGETED_FEATURES used in the analysis notebooks.
_VECTOR_STAT_KEYS: list[tuple[str, str, int]] = [
    # (agg_stats key,  csv column prefix,  expected length)
    ("degree_moments", "degree_moments", 4),
    ("annd",           "annd",           4),
    ("eccentricity",   "eccentricity",   4),
]


def _add_targeted_stats_to_row(row: dict, agg_stats: dict) -> None:
    """Adds the 14 targeted graph statistics as flat ``<name>_mean`` columns.

    Handles both the case where ``agg_stats`` is a fully-populated dict (Phase A
    and Phase C) and the case where it is an empty dict / ``None`` (Phase B,
    where no real graphs exist).  Missing or ``None`` values are written as
    ``None`` so that the CSV column is present but empty.

    Vector statistics (``degree_moments``, ``annd``, ``eccentricity``) are
    expanded element-wise into columns ``<prefix>_0`` … ``<prefix>_{n-1}``.

    Args:
        row: The CSV row dict to mutate in place.
        agg_stats: The aggregate_statistics dict returned by
            :func:`src.graph_analysis.aggregate_statistics`, or an empty dict.
    """
    # Scalar targeted stat
    row["n_edges_mean"] = agg_stats.get("n_edges", None) if agg_stats else None

    # Vector targeted stats
    for agg_key, col_prefix, length in _VECTOR_STAT_KEYS:
        vec = agg_stats.get(agg_key, None) if agg_stats else None
        for i in range(length):
            col = f"{col_prefix}_{i}_mean"
            if vec is not None:
                try:
                    row[col] = float(vec[i])
                except (IndexError, TypeError):
                    row[col] = None
            else:
                row[col] = None

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
        
    # Add targeted stats (n_edges + vector stats)
    _add_targeted_stats_to_row(row, agg_stats)

    # Add non-targeted metrics
    row["modularity_mean"] = agg_stats.get("modularity", 0.0)
    row["clustering_mean"] = agg_stats.get("clustering", 0.0)
    row["assortativity_mean"] = agg_stats.get("assortativity", 0.0)
    row["efficiency_mean"] = agg_stats.get("efficiency", 0.0)
    row["diameter_mean"] = agg_stats.get("diameter", 0.0)

    return row


def _get_enc_block_size(encoder) -> int:
    """Returns the number of scalar values in the ``_encodings`` block of *encoder*.

    For percentile / GMCM encoders this is ``num_percentiles × num_features``;
    for the moments encoder it is ``k × num_features``.
    """
    num_features = encoder._is_discrete.size
    if hasattr(encoder, "percentile_size"):
        q = np.arange(0, 1 + encoder.percentile_size / 2, encoder.percentile_size)
        q[q > 1.0] = 1.0
        if q[-1] < 1.0:
            q = np.append(q, 1.0)
        return len(q) * num_features
    elif hasattr(encoder, "k"):
        return encoder.k * num_features
    raise ValueError(f"Cannot determine encoding block size for {type(encoder).__name__}.")


def intelligent_perturbing(
    embedding: np.ndarray,
    enc_size: int,
    enc_rows: int,
    sampler_name: str,
    rng: np.random.Generator,
    noise_std_frac: float = 0.05,
    shift_frac: float = 0.05,
    var_frac: float = 0.10,
) -> np.ndarray:
    """Perturbs only the ``_encodings`` block of a flat embedding vector.

    The ``_encodings`` block occupies the first ``enc_size`` elements of
    *embedding*.  The remainder (GMM weights/means/covariances for GMCM,
    correlation matrix for ``percentile_corr``) is returned **unchanged**.

    Perturbation strategy:

    * **moments** sampler — each scalar in the block is independently
      shifted by zero-mean Gaussian noise whose std equals
      ``noise_std_frac × |value|`` (with a small absolute floor).
    * **percentile / percentile_corr / gmcm** samplers — the block is
      a ``(num_percentiles, num_features)`` matrix of monotone columns.
      For each feature column two scalars are drawn:
      - ``shift`` ~ Uniform(−shift_frac·span, +shift_frac·span) translates the whole column.
      - ``scale`` ~ Uniform(1−var_frac, 1+var_frac) expands/compact the column around its median value.

    Args:
        embedding:       Flat 1-D embedding vector as stored in the CSV.
        enc_size:        Number of scalar values belonging to ``_encodings``.
        enc_rows:        Number of rows in the encoding matrix (``k`` for moments, ``num_percentiles`` for percentile/GMCM).
        sampler_name:    One of ``"moments"``, ``"percentile"``, ``"percentile_corr"``, ``"gmcm"``.
        rng:             NumPy random generator.
        noise_std_frac:  Fraction of |value| used as noise std (moments).
        shift_frac:      Fraction of column span used as shift range (percentile-based samplers).
        var_frac:        Maximum fractional change in column spread (percentile-based samplers).
    Returns:
        New flat embedding with the ``_encodings`` block perturbed and the rest of the vector unchanged.
    """
    result = embedding.copy()
    enc_flat = embedding[:enc_size].copy()
    num_features = enc_size // enc_rows
    enc_mat = enc_flat.reshape(enc_rows, num_features)

    if sampler_name == "moments":
        # Independent Gaussian noise on every scalar.
        noise = rng.normal(0.0, np.maximum(noise_std_frac * np.abs(enc_mat), 1e-6))
        enc_mat = enc_mat + noise
    else:
        # Percentile-based samplers: shift + scale per feature column.
        for col in range(num_features):
            column = enc_mat[:, col]
            span = column[-1] - column[0]
            median_val = np.median(column)
            shift = rng.uniform(-shift_frac * max(abs(span), 1e-6), shift_frac * max(abs(span), 1e-6))
            scale = rng.uniform(1.0 - var_frac, 1.0 + var_frac)
            enc_mat[:, col] = median_val + scale * (column - median_val) + shift

    result[:enc_size] = enc_mat.flatten()
    return result

def _check_topo(encoder, sampler_name: str) -> bool:
    """Verifies that the encoder's learned feature distributions do not exceed
    topological thresholds (max nodes <= 500 and max edges <= 1000).
    """
    enc = encoder._encodings[0]
    if enc is None:
        return False

    if sampler_name == "moments":
        mu_nodes  = float(enc[0, 0])
        mu_degree = float(enc[0, 1])
        s_nodes   = float(np.sqrt(max(enc[1, 0], 0.0))) if enc.shape[0] >= 2 else 0.0
        s_degree  = float(np.sqrt(max(enc[1, 1], 0.0))) if enc.shape[0] >= 2 else 0.0
        max_nodes  = mu_nodes  + 3.0 * s_nodes
        max_degree = mu_degree + 3.0 * s_degree
    else:
        max_nodes  = float(enc[-1, 0])
        max_degree = float(enc[-1, 1])

    max_edges = (max_nodes * max_degree) / 2.0
    return max_nodes <= 500 and max_edges <= 1000

def generate_target_statistics(stats_rows: list[dict[str, str | int | float | None]], num_synth_datasets: int, sampler_name: str, rng: np.random.Generator,) -> list[dict[str, str | int | float | None]]:
    """Generates target statistics in two phases.

    **Phase 1 — Per-original perturbation (5 variants each):**
    For each row in *stats_rows*, ``intelligent_perturbing`` is applied 5 times
    to produce variants that stay close to the original distributions.  Only the
    ``_encodings`` block is perturbed; the GMM or correlation-matrix tail of the
    embedding is kept **unchanged**.

    **Phase 2 — Two-parent interpolation/extrapolation:**
    After Phase 1, the remaining slots up to *num_synth_datasets* are filled by
    the original asymmetric parent-selection strategy:

    1. Parent A is sampled uniformly; Parent B is sampled proportionally to
       node count (larger graphs are preferred as the B anchor).
    2. Directional interpolation (50 %) in [0, 1] or extrapolation (50 %) with
       exponential decay beyond the A–B segment.
    3. The resulting embedding is perturbed using ``intelligent_perturbing`` to
       only explore feature variations while keeping copula correlations/GMM parameters intact.
    4. Dual validation: semantic (``load_embedding``) + topological (O(1)).

    Args:
        stats_rows: List of dicts with original dataset mean statistics (must contain ``emb_*`` columns).
        num_synth_datasets: Total number of target rows to generate.
        sampler_name: Name of the sampler (``"gmcm"``, ``"moments"``, etc.).
        rng: NumPy random generator.
    Returns:
        List of dicts representing the generated target statistics/embeddings.
    """
    _PERTURB_PER_ORIG = 0   # fixed number of per-original variants in Phase 1

    logger = logging.getLogger(__name__)
    targeted_features = [k for k in stats_rows[0].keys() if k.startswith("emb_")]
    non_targeted_features = ["modularity_mean", "clustering_mean", "assortativity_mean", "efficiency_mean", "diameter_mean"]

    features_matrix = np.array([[row[feat] for feat in targeted_features] for row in stats_rows], dtype=np.float64)
    num_graphs      = np.array([row["Num_Graphs"] for row in stats_rows], dtype=np.int64)
    node_counts     = np.array([row["n_nodes_mean"] for row in stats_rows], dtype=np.float64)
    mean_num_graphs = int(np.mean(num_graphs))

    # Shared encoder infrastructure.
    is_discrete = np.array([True, True] + [False] * 12)
    _probe   = KNOWN_SAMPLERS[sampler_name](num_classes=1, is_discrete=is_discrete, rng=rng)
    enc_size = _get_enc_block_size(_probe)
    enc_rows = enc_size // _probe._is_discrete.size  # num_percentiles or k
    encoder  = KNOWN_SAMPLERS[sampler_name](num_classes=1, is_discrete=is_discrete, rng=rng)

    target_rows: list[dict] = []
    target_idx = 0

    # ── PHASE 1: Per-original intelligent perturbation ─────────────────────────
    for orig_idx, orig_row in enumerate(stats_rows):
        base_emb     = np.array([orig_row[feat] for feat in targeted_features], dtype=np.float64)
        orig_n_nodes = float(node_counts[orig_idx])

        accepted  = 0
        attempts  = 0
        max_att_p1 = _PERTURB_PER_ORIG * 50

        while accepted < _PERTURB_PER_ORIG and attempts < max_att_p1:
            attempts += 1

            candidate = intelligent_perturbing(base_emb, enc_size, enc_rows, sampler_name, rng)

            if not encoder.load_embedding(candidate, class_id=0):
                continue

            if not _check_topo(encoder, sampler_name):
                continue

            target_row = {
                "Dataset": f"target_{target_idx}",
                "Num_Graphs": mean_num_graphs,
                "n_nodes_mean": orig_n_nodes,
            }
            for j, feat in enumerate(targeted_features):
                target_row[feat] = float(candidate[j])
            _add_targeted_stats_to_row(target_row, {})
            for feat in non_targeted_features:
                target_row[feat] = None

            target_rows.append(target_row)
            target_idx += 1
            accepted += 1

        if accepted < _PERTURB_PER_ORIG:
            logger.warning(f"Phase 1: only {accepted}/{_PERTURB_PER_ORIG} variants for '{orig_row.get('Dataset')}' after {attempts} attempts.")

    # ── PHASE 2: Two-parent interpolation/extrapolation ────────────────────────
    prob_b       = node_counts / np.sum(node_counts)
    remaining    = num_synth_datasets - len(target_rows)
    max_attempts = remaining * 100
    attempts     = 0

    while len(target_rows) < num_synth_datasets and attempts < max_attempts:
        attempts += 1

        idx_a = rng.choice(len(stats_rows))
        idx_b = rng.choice(len(stats_rows), p=prob_b)
        val_a = features_matrix[idx_a]
        val_b = features_matrix[idx_b]
        direction = val_b - val_a

        if rng.random() < 0.5:
            r_val = rng.uniform(0.0, 1.0)
        else:
            t_exp = rng.exponential(scale=0.5)
            sign  = rng.choice([-1.0, 1.0])
            r_val = sign * t_exp

        interpolated    = val_a + r_val * direction
        r_val_clamped   = float(np.clip(r_val, 0.0, 1.0))
        perturbed       = intelligent_perturbing(interpolated, enc_size, enc_rows, sampler_name, rng)

        if not encoder.load_embedding(perturbed, class_id=0):
            continue

        if not _check_topo(encoder, sampler_name):
            logger.debug(f"Phase 2 topological check failed: max_nodes={encoder._encodings[0][-1, 0] if encoder._encodings[0] is not None and sampler_name != 'moments' else 'unknown'} — discarding.")
            continue

        target_row = {
            "Dataset": f"target_{target_idx}",
            "Num_Graphs": mean_num_graphs,
            "n_nodes_mean": float(np.interp(r_val_clamped, [0.0, 1.0], [node_counts[idx_a], node_counts[idx_b]])),
        }
        for j, feat in enumerate(targeted_features):
            target_row[feat] = float(perturbed[j])
        _add_targeted_stats_to_row(target_row, {})
        for feat in non_targeted_features:
            target_row[feat] = None

        target_rows.append(target_row)
        target_idx += 1

    if len(target_rows) < num_synth_datasets:
        logger.warning(f"generate_target_statistics: generated only {len(target_rows)}/{num_synth_datasets} targets after Phase 1 + Phase 2 ({attempts} Phase-2 attempts).")

    return target_rows


def worker_generate_graph(task: dict) -> tuple:
    """Generates a single graph for a synthetic dataset (Phase C).

    Returns:
        A 4-tuple ``(pyg_data, info, success, captured_logs)`` where *captured_logs* is
        a list of log messages.
    """
    import io
    import logging
    import numpy as np
    import torch
    from src.generate_datasets import generate_graph, networkx_to_igraph
    from src.data_utils import igraph_to_pytorch
    from torch_geometric.data import Data

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    
    captured_logs = []

    try:
        target_stats = task["target_stats"]
        method = task["method"]
        seed = task["seed"]
    
        rng = np.random.default_rng(seed)
        try:
            synth_nx, info = generate_graph(target_stats, method, rng)
            synth_ig = networkx_to_igraph(synth_nx)
            pyg_data = igraph_to_pytorch(synth_ig, y=torch.tensor([0]))
            success = True
        except Exception as exc:
            captured_logs.append(f"Graph generation failed for method '{method}' (seed={seed}): {exc}")
            pyg_data = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([0]), num_nodes=1)
            info = {}
            success = False
    finally:
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)
        
    captured_logs.extend([line for line in log_stream.getvalue().splitlines() if line.strip()])
    return pyg_data, info, success, captured_logs




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
    results_dir = project_root / "results"
    if stats_rows:
        csv_path = results_dir / "original_datasets_mean_stats.csv"
        _write_csv(stats_rows, csv_path)
        logger.info(f"Phase A completed. Mean statistics saved to {csv_path}")

    # Determine whether Phase B/C should append to existing CSVs or overwrite them.
    target_csv_path = results_dir / "target_datasets_mean_stats.csv"
    synth_csv_path  = results_dir / "synthetic_datasets_mean_stats.csv"
    append_mode = (
        not args.process_original
        and target_csv_path.exists()
        and synth_csv_path.exists()
    )
    if append_mode:
        logger.info("Append mode: new target/synthetic rows will be added to existing CSV files.")

    # ── PHASE B: Target Datasets Generation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE B: Generating {args.num_synth_datasets} target statistics vectors")
    logger.info("=" * 60)

    if not stats_rows:
        logger.error("No original dataset statistics found. Cannot compute feature bounds for Phase B.")
        return

    # Choose between the uniform range, SMOTE-inspired, or asymmetric interpolation method
    target_rows = generate_target_statistics(stats_rows, args.num_synth_datasets, args.sampler, rng)

    # Renumber Dataset indices to continue from the last existing row when appending.
    if append_mode and target_rows:
        import pandas as _pd
        existing_count = len(_pd.read_csv(target_csv_path))
        for i, row in enumerate(target_rows):
            row["Dataset"] = f"target_{existing_count + i}"

    # Save / append target statistics to CSV.
    if append_mode:
        _append_csv(target_rows, target_csv_path)
        logger.info(f"Phase B completed. Appended {len(target_rows)} target rows to {target_csv_path}")
    else:
        _write_csv(target_rows, target_csv_path)
        logger.info(f"Phase B completed. Target mean statistics saved to {target_csv_path}")

    # ── PHASE C: Synthetic Datasets Generation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PHASE C: Generating and evaluating synthetic datasets using {args.method}")
    logger.info("=" * 60)

    if not target_rows:
        logger.error("Phase B produced 0 valid target embeddings — aborting Phase C.")
        return

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

    # Use the 'spawn' start method to avoid deadlocks from inherited locks
    # (logging, tqdm, etc.) that are a known hazard of the default 'fork' on Linux.
    _spawn_ctx = mp.get_context("spawn")
    pool = _spawn_ctx.Pool(processes=args.num_workers) if args.num_workers > 1 else None
    
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
                for pyg_data, info, success, captured_logs in results:
                    for log_msg in captured_logs:
                        logger.warning(log_msg)
                    graphs.append(pyg_data)
                    infos.append(info)
            else:
                for task in graph_tasks:
                    pyg_data, info, success, captured_logs = worker_generate_graph(task)
                    for log_msg in captured_logs:
                        logger.warning(log_msg)
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
                
            # Add targeted stats (n_edges + vector stats)
            _add_targeted_stats_to_row(row, synth_agg)

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

    # Save / append synthetic dataset statistics to a CSV.
    if append_mode and synth_rows:
        import pandas as _pd
        existing_synth_count = len(_pd.read_csv(synth_csv_path))
        for i, row in enumerate(synth_rows):
            row["Dataset"] = f"synth_{existing_synth_count + i}"
        _append_csv(synth_rows, synth_csv_path)
        logger.info(f"Phase C completed. Appended {len(synth_rows)} synthetic rows to {synth_csv_path}")
    else:
        _write_csv(synth_rows, synth_csv_path)
        logger.info(f"Phase C completed. Synthetic datasets statistics saved to {synth_csv_path}")
    
if __name__ == "__main__":
    main()
