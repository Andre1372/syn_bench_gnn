"""Multi-dataset, multi-method synthetic graph generation and GNN evaluation."""

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
from src.data_utils import load_all_synthetic_variants, get_split_indices, preprocess_and_save_original_dataset, DatasetPT
from src.generate_datasets import generate_synthetic_variants, KNOWN_METHODS
from src.train_gnn import evaluate_dataset

ALL_DATASETS = [
        "BZR", "DHFR", "Mutagenicity", "MUTAG"
    ]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset, multi-method synthetic graph generation and GNN evaluation.",
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
        default=None,
        metavar="N",
        help=(
            "If set, each dataset is down-sampled to at most N graphs before "
            "generation and evaluation.  Sampling is stratified by label and "
            "preserves the node/edge-count distribution within each class."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["padma"],
        metavar="METHOD",
        help=f"Generation methods to run.  Supported: {', '.join(KNOWN_METHODS)}.",
    )
    parser.add_argument(
        "--num_synth_datasets",
        "-V",
        type=int,
        default=20,
        help="Number of independent synthetic variants V to generate per (dataset, method) pair.",
    )

    # GNN training
    parser.add_argument(
        "--process_original",
        action="store_true",
        help="Pre-process (and sample) the original dataset even if it exists. Also enables evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of GNN training epochs.",
    )
    parser.add_argument(
        "--gnn_runs",
        "-R",
        type=int,
        default=10,
        help="Number of independent GNN training runs per dataset for variance estimation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for GNN training and evaluation.",
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
        "--skip_generation",
        action="store_true",
        help="Skip Phase A (generation) and use existing synthetic data on disk.",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip Phase B (GNN evaluation).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, int(mp.cpu_count() * 0.9)),
        help="Number of worker processes for parallel generation (Phase A).",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run a fast functional test (1 epoch, 1 run, small models).",
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    project_root = Path(".")
    
    # Handle "all" dataset selection
    if "all" in args.dataset:
        args.dataset = ALL_DATASETS

    n_datasets = len(args.dataset)
    n_methods = len(args.methods)
    session_tag = (
        f"multi_exp_{n_datasets}D_{n_methods}M_{args.num_synth_datasets}V"
        if n_datasets > 1
        else f"exp_{args.dataset[0]}_{n_methods}M_{args.num_synth_datasets}V"
    )
    setup_console_logging(project_root, session_tag)
    logger = logging.getLogger(__name__)
    
    # Reproducibility
    rng = (
        np.random.default_rng()
        if args.seed == -1
        else np.random.default_rng(args.seed)
    )

    # ── PHASE A: Data Generation ────────────────────────────────────────────
    if not args.skip_generation:
        logger.info("=" * 60)
        logger.info(f"PHASE A: Generating {args.num_synth_datasets} synthetic variants for {n_datasets} dataset(s) x {n_methods} method(s)")
        logger.info("=" * 60)

        for dataset_name in args.dataset:
            # Preprocess and save (with optional down-sampling) before generation.
            # Only re-process if explicitly requested or if the file doesn't exist.
            orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
            if args.process_original or not orig_pt_path.exists():
                preprocess_and_save_original_dataset(dataset_name, project_root / "data", max_size=args.cut_datasets, rng=rng)
            else:
                logger.info(f"Using existing original dataset for {dataset_name} (skip re-processing).")

            for method in args.methods:
                logger.info(f"--- Generating: dataset={dataset_name}  method={method} ---")
                output_dir = project_root / "synthetic_data" / dataset_name / method
                generate_synthetic_variants(
                    dataset_name=dataset_name,
                    method=method,
                    num_variants=args.num_synth_datasets,
                    rng=rng,
                    project_root=project_root,
                    output_dir=output_dir,
                    num_workers=args.num_workers,
                )
                logger.info(f"Done: {args.num_synth_datasets} variants saved to {output_dir}")
    else:
        logger.info("Skipping Phase A (--skip_generation).")

    # ── PHASE B: GNN Evaluation ─────────────────────────────────────────────
    if args.skip_evaluation:
        logger.info("Skipping Phase B (--skip_evaluation).")
        return

    logger.info("=" * 60)
    logger.info(f"PHASE B: GNN Evaluation for {n_datasets} dataset(s) x {n_methods} method(s)")
    logger.info("=" * 60)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_epochs = 1 if args.quick_test else args.epochs
    gnn_config_base = {
        "num_runs"  : 1 if args.quick_test else args.gnn_runs,
        "lr"        : 5e-4,
        "in_dim"    : 1,
        "hidden_dim": 16 if args.quick_test else 256,
        "num_layers": 1 if args.quick_test else 3,
        "dropout"   : 0.1,
    }

    for dataset_name in args.dataset:
        logger.info("─" * 60)
        logger.info(f"DATASET: {dataset_name}")
        logger.info("─" * 60)

        orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
        if not orig_pt_path.exists():
            preprocess_and_save_original_dataset(dataset_name, project_root / "data", max_size=args.cut_datasets, rng=rng)

        orig_dataset_obj = DatasetPT(orig_pt_path)
        # The .pt already contains the (possibly cut) dataset — no extra sampling needed.
        original_data_list = [orig_dataset_obj[i] for i in range(len(orig_dataset_obj))]


        gnn_config = {
            **gnn_config_base,
            "num_classes": orig_dataset_obj.metadata.get("num_classes"),
        }

        # Shared train/val/test split (fixed across all methods for comparability)
        split_indices = get_split_indices(original_data_list, seed=42)

        # Optionally evaluate on original dataset and save the baseline CSV
        if args.process_original:
            logger.info(f"Evaluating original dataset ({dataset_name})...")
            
            # Match the total number of runs used across all synthetic variants (R * V)
            orig_gnn_config = gnn_config.copy()
            orig_gnn_config["num_runs"] = gnn_config["num_runs"] * args.num_synth_datasets

            with tqdm(total=orig_gnn_config["num_runs"], desc=f"GNN [Original/{dataset_name}]") as pbar:
                with logging_redirect_tqdm():
                    glob_res, pg_res = evaluate_dataset(
                        pt_path=orig_pt_path,
                        gnn_config=orig_gnn_config,
                        device=device,
                        split_indices=split_indices,
                        dataset_name=dataset_name,
                        epochs=test_epochs,
                        batch_size=args.batch_size,
                        pbar=pbar,
                    )
            _write_csv(glob_res, results_dir / f"gnn_eval_{dataset_name}_original.csv")
            _write_csv(pg_res,   results_dir / f"per_graph_{dataset_name}_original.csv")
            logger.info("Original results saved.")

        # Evaluate each synthetic method
        for method in args.methods:
            logger.info(f"Evaluating method: {method}")
            base_synth_dir = project_root / "synthetic_data" / dataset_name / method
            eval_tasks = load_all_synthetic_variants(base_synth_dir, dataset_name)

            if not eval_tasks:
                logger.warning(f"No synthetic data found for dataset={dataset_name} method={method} - skipping.")
                continue

            all_results: list[dict] = []
            all_pg_results: list[dict] = []
            
            with tqdm(total=len(eval_tasks)*gnn_config["num_runs"], desc=f"GNN [{method.upper()}/{dataset_name}]") as pbar:
                with logging_redirect_tqdm():
                    for pt_path in eval_tasks:
                        logger.debug(f"Evaluating path: {pt_path}.")
                        glob_res, pg_res = evaluate_dataset(
                            pt_path=pt_path,
                            gnn_config=gnn_config,
                            device=device,
                            split_indices=split_indices,
                            dataset_name=dataset_name,
                            epochs=test_epochs,
                            batch_size=args.batch_size,
                            pbar=pbar,
                        )
                        all_results.extend(glob_res)
                        all_pg_results.extend(pg_res)

            _write_csv(all_results,   results_dir / f"gnn_eval_{dataset_name}_{method}.csv")
            _write_csv(all_pg_results, results_dir / f"per_graph_{dataset_name}_{method}.csv")
            logger.info(f"Results saved → gnn_eval_{dataset_name}_{method}.csv / per_graph_{dataset_name}_{method}.csv")

    logger.info("=" * 60)
    logger.info("ALL DATASETS PROCESSED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
