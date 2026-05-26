"""Script to download TUDatasets and filter them based on specific structural criteria.

This module provides functionality to programmatically download datasets from the
TUDataset collection, compute structural statistics (node/edge counts, etc.),
filter them based on predefined constraints, run a quick GNN benchmark,
and keep only datasets where both GCN and GIN reach a mean test F1 >= 0.6.
Incompatible datasets are automatically removed to save storage.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from benchmark_gnn import benchmark_dataset, BenchmarkConfig

MIN_F1_THRESHOLD = 0.65


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_dataset_stats(dataset_name: str, data_dir: Path) -> dict[str, Any] | None:
    """Downloads a TUDataset and extracts its structural statistics."""
    try:
        dataset = TUDataset(root=str(data_dir), name=dataset_name)

        num_graphs = len(dataset)
        num_classes = dataset.num_classes
        num_features = dataset.num_node_features

        # Compute node and edge statistics
        nodes = [data.num_nodes for data in dataset]
        edges = [data.num_edges for data in dataset]

        avg_nodes = float(np.mean(nodes))
        std_nodes = float(np.std(nodes))
        avg_edges = float(np.mean(edges))
        std_edges = float(np.std(edges))

        return {
            "dataset": dataset_name,
            "graphs": num_graphs,
            "classes": num_classes,
            "avg_nodes": avg_nodes,
            "std_nodes": std_nodes,
            "avg_edges": avg_edges,
            "std_edges": std_edges,
            "node_features": num_features,
            "compatible": (num_graphs >= 50 and num_graphs <= 5000 and avg_nodes <= 300)
        }
    except Exception as e:
        logger.error(f"Failed to process dataset '{dataset_name}': {e}")
        return None


def cleanup_incompatible_dataset(dataset_name: str, data_dir: Path) -> None:
    """Removes the dataset directory from the local filesystem."""
    dataset_path = data_dir / dataset_name
    if dataset_path.exists() and dataset_path.is_dir():
        try:
            shutil.rmtree(dataset_path)
            logger.info(f"Successfully deleted incompatible dataset folder: {dataset_name}")
        except OSError as e:
            logger.error(f"Error while deleting {dataset_name}: {e}")


def display_results(stats_list: list[dict[str, Any]], discarded_structural: list[str], discarded_f1: list[str], ran_benchmark: bool = True, min_f1: float = MIN_F1_THRESHOLD, feature_type: str = "log_bin_deg") -> None:
    """Formats and prints the compatible and discarded dataset statistics to the console."""
    if discarded_structural:
        print("\n" + "="*80)
        print(f"Discarded by structural criteria ({len(discarded_structural)})")
        print("="*80)
        print(", ".join(discarded_structural))
        print("="*80)

    if ran_benchmark and discarded_f1:
        print("\n" + "="*80)
        print(f"Discarded by GNN benchmark (did not reach F1 >= {min_f1} for both GCN & GIN on {feature_type}) ({len(discarded_f1)})")
        print("="*80)
        print(", ".join(discarded_f1))
        print("="*80)

    if not stats_list:
        logger.warning("No datasets matched all criteria.")
        return

    df = pd.DataFrame(stats_list)

    if ran_benchmark:
        def _fmt(val: float) -> str:
            return f"{val:.3f}"

        def _pass(val: float) -> str:
            mark = "\u2713" if val >= min_f1 else "\u2717"
            return f"{val:.3f} {mark}"

        # Format output columns for readability
        display_df = pd.DataFrame({
            "dataset": df["dataset"],
            "graphs": df["graphs"],
            "classes": df["classes"],
            "avg nodes +- std": df.apply(lambda r: f"{r['avg_nodes']:.1f} +- {r['std_nodes']:.1f}", axis=1),
            "avg edges +- std": df.apply(lambda r: f"{r['avg_edges']:.1f} +- {r['std_edges']:.1f}", axis=1),
            "node feat": df["node_features"],
            f"GCN {feature_type} F1": df[f"gcn_{feature_type}_mean_f1"].map(_pass),
            f"GIN {feature_type} F1": df[f"gin_{feature_type}_mean_f1"].map(_pass),
            "kept config": df["kept_config"],
        })

        print("\n" + "="*130)
        print(f"Final Compatible Datasets (structural + GCN & GIN F1 >= {min_f1} on {feature_type})")
        print("="*130)
    else:
        # Format simpler output columns if benchmark was skipped
        display_df = pd.DataFrame({
            "dataset": df["dataset"],
            "graphs": df["graphs"],
            "classes": df["classes"],
            "avg nodes +- std": df.apply(lambda r: f"{r['avg_nodes']:.1f} +- {r['std_nodes']:.1f}", axis=1),
            "avg edges +- std": df.apply(lambda r: f"{r['avg_edges']:.1f} +- {r['std_edges']:.1f}", axis=1),
            "node feat": df["node_features"],
        })

        print("\n" + "="*100)
        print("Final Compatible Datasets (structural criteria only)")
        print("="*100)

    print(display_df.to_string(index=False))
    print("="*130 + "\n")


def main() -> None:
    """Main execution flow for downloading and filtering TUDatasets."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Download TUDatasets and filter them based on specific structural criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_benchmark",
        action="store_true",
        help="Run GNN benchmark for filtering (otherwise, only use structural criteria).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=[
        "AIDS", "BZR", "COX2", "DHFR", "FRANKENSTEIN", "Mutagenicity", "MUTAG", "NCI1", "NCI109",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "DD", "KKI", "OHSU", "Peking_1", "PROTEINS", 
        "FIRSTMM_DB", "Letter_high", "IMDB-BINARY", "REDDIT-BINARY",

        # Originally with more than 2 classes, but can be binarized
        "ENZYMES", "COIL-DEL", "COIL-RAG", "Cuneiform", "Fingerprint", "Letter-low", "Letter-med", 
        "MSRC_9", "MSRC_21", "MSRC_21C", "IMDB-MULTI"

        # "SYNTHETIC", "Synthie", --> Discarded because synthetic
        # "BZR_MD", "COX2_MD", "DHFR_MD", "ER_MD"  --> Discarded because fully connected
        # "dblp_ct1", "facebook_ct1", "highschool_ct1", "infectious_ct1", "mit_ct1", "tumblr_ct1" --> Discarded because problematic
        ],
        help="One or more TUDataset names to process.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs for GNN benchmark.",
    )
    parser.add_argument(
        "--min_f1",
        type=float,
        default=0.65,
        help="Minimum mean F1 threshold to keep the dataset.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="log_bin_deg",
        choices=["constant", "log_bin_deg", "random_sample", "degree_ordered", "neighbor_degree_ordered"],
        help="The node feature assignment strategy (assigner) to benchmark.",
    )
    args = parser.parse_args()

    dataset_names = args.dataset
    min_f1 = args.min_f1

    project_root = Path(__file__).parent.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    structurally_compatible: list[dict[str, Any]] = []
    discarded_structural: list[str] = []
    logger.info(f"Processing {len(dataset_names)} TUDatasets: {dataset_names}...")

    # --- Phase 1: Structural filtering ---
    for name in tqdm(dataset_names, desc="Phase 1 - structural filter"):
        stats = fetch_dataset_stats(name, data_dir)

        if stats and stats["compatible"]:
            structurally_compatible.append(stats)
        else:
            discarded_structural.append(name)
            logger.info(f"Dataset '{name}' does not meet structural criteria. Cleaning up...")
            cleanup_incompatible_dataset(name, data_dir)

    logger.info(
        f"Phase 1 done: {len(structurally_compatible)} datasets passed structural criteria, "
        f"{len(discarded_structural)} discarded."
    )

    if not args.run_benchmark:
        logger.info("GNN benchmark skipped as requested. Keeping all structurally compatible datasets.")
        display_results(structurally_compatible, discarded_structural, [], ran_benchmark=False, min_f1=min_f1, feature_type=args.feature)
        return

    # --- Phase 2: GNN benchmark filtering ---
    final_results: list[dict[str, Any]] = []
    discarded_f1: list[str] = []

    for stats in tqdm(structurally_compatible, desc="Phase 2 - GNN benchmark"):
        name = stats["dataset"]
        bench_config = BenchmarkConfig(runs=args.runs, epochs=50, feature_type=args.feature)
        logger.info(f"Benchmarking '{name}' ({bench_config.runs} runs, 50 epochs) – {args.feature}...")
        bench = benchmark_dataset(name, data_dir, bench_config)

        if bench is None:
            discarded_f1.append(name)
            logger.warning(f"'{name}' failed benchmark - discarding.")
            cleanup_incompatible_dataset(name, data_dir)
            continue

        gcn_f1 = bench[f"gcn_{args.feature}_mean_f1"]
        gin_f1 = bench[f"gin_{args.feature}_mean_f1"]

        f1_ok = gcn_f1 >= min_f1 and gin_f1 >= min_f1

        logger.info(
            f"'{name}': {args.feature} GCN={gcn_f1:.3f} GIN={gin_f1:.3f} ({'OK' if f1_ok else 'FAIL'})"
        )

        if f1_ok:
            final_results.append({
                **stats,
                f"gcn_{args.feature}_mean_f1": gcn_f1,
                f"gin_{args.feature}_mean_f1": gin_f1,
                "kept_config": args.feature,
            })
        else:
            discarded_f1.append(name)
            logger.info(
                f"'{name}' discarded: GCN & GIN did not both reach F1 >= {min_f1} for {args.feature}. Cleaning up..."
            )
            cleanup_incompatible_dataset(name, data_dir)

    display_results(final_results, discarded_structural, discarded_f1, ran_benchmark=True, min_f1=min_f1, feature_type=args.feature)


if __name__ == "__main__":
    main()
