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

MIN_F1_THRESHOLD = 0.6


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
            "compatible": (num_graphs >= 50 and num_graphs <= 4500 and avg_nodes <= 300)
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


def display_results(stats_list: list[dict[str, Any]], discarded_structural: list[str], discarded_f1: list[str]) -> None:
    """Formats and prints the compatible and discarded dataset statistics to the console."""
    if discarded_structural:
        print("\n" + "="*80)
        print(f"Discarded by structural criteria ({len(discarded_structural)})")
        print("="*80)
        print(", ".join(discarded_structural))
        print("="*80)

    if discarded_f1:
        print("\n" + "="*80)
        print(f"Discarded by GNN benchmark F1 < {MIN_F1_THRESHOLD} ({len(discarded_f1)})")
        print("="*80)
        print(", ".join(discarded_f1))
        print("="*80)

    if not stats_list:
        logger.warning("No datasets matched all criteria.")
        return

    df = pd.DataFrame(stats_list)

    # Format output columns for readability
    display_df = pd.DataFrame({
        "dataset": df["dataset"],
        "graphs": df["graphs"],
        "classes": df["classes"],
        "average nodes +- std": df.apply(lambda r: f"{r['avg_nodes']:.2f} +- {r['std_nodes']:.2f}", axis=1),
        "average edges +- std": df.apply(lambda r: f"{r['avg_edges']:.2f} +- {r['std_edges']:.2f}", axis=1),
        "node features": df["node_features"],
        "GCN F1": df["gcn_mean_f1"].map(lambda x: f"{x:.3f}"),
        "GIN F1": df["gin_mean_f1"].map(lambda x: f"{x:.3f}"),
    })

    print("\n" + "="*100)
    print(f"Final Compatible Datasets (structural + GNN F1 >= {MIN_F1_THRESHOLD})")
    print("="*100)
    print(display_df.to_string(index=False))
    print("="*100 + "\n")


def main() -> None:
    """Main execution flow for downloading and filtering TUDatasets."""
    dataset_names = [
        "AIDS", "BZR", "COX2", "DHFR", "FRANKENSTEIN", "Mutagenicity", "MUTAG", "NCI1", "NCI109",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "DD", "KKI", "OHSU", "Peking_1", "PROTEINS", 
        "FIRSTMM_DB", "Letter_high", "IMDB-BINARY", "REDDIT-BINARY",

        # Originally with more than 2 classes, but can be binarized
        "ENZYMES", "COIL-DEL", "COIL-RAG", "Cuneiform", "Fingerprint", "Letter-low", "Letter-med", 
        "MSRC_9", "MSRC_21", "MSRC_21C", "IMDB-MULTI"

        # "SYNTHETIC", "Synthie", --> Discarded because synthetic
        # "BZR_MD", "COX2_MD", "DHFR_MD", "ER_MD"  --> Discarded because fully connected
        # "dblp_ct1", "facebook_ct1", "highschool_ct1", "infectious_ct1", "mit_ct1", "tumblr_ct1" --> Discarded because problematic
    ]

    project_root = Path(__file__).parent.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    structurally_compatible: list[dict[str, Any]] = []
    discarded_structural: list[str] = []
    logger.info(f"Processing {len(dataset_names)} TUDatasets...")

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

    # --- Phase 2: GNN benchmark filtering ---
    final_results: list[dict[str, Any]] = []
    discarded_f1: list[str] = []

    for stats in tqdm(structurally_compatible, desc="Phase 2 - GNN benchmark"):
        name = stats["dataset"]
        bench_config = BenchmarkConfig(runs=10, epochs=50)
        logger.info(f"Benchmarking '{name}' ({bench_config.runs} runs, {bench_config.epochs} epochs)...")
        bench = benchmark_dataset(name, data_dir, bench_config)

        if bench is None:
            discarded_f1.append(name)
            logger.warning(f"'{name}' failed benchmark - discarding.")
            cleanup_incompatible_dataset(name, data_dir)
            continue

        gcn_f1 = bench["gcn_mean_f1"]
        gin_f1 = bench["gin_mean_f1"]
        logger.info(f"'{name}': GCN F1={gcn_f1:.3f}, GIN F1={gin_f1:.3f}")

        if gcn_f1 >= MIN_F1_THRESHOLD and gin_f1 >= MIN_F1_THRESHOLD:
            final_results.append({**stats, "gcn_mean_f1": gcn_f1, "gin_mean_f1": gin_f1})
        else:
            discarded_f1.append(name)
            logger.info(
                f"'{name}' discarded: GCN F1={gcn_f1:.3f}, GIN F1={gin_f1:.3f} "
                f"(threshold: {MIN_F1_THRESHOLD}). Cleaning up..."
            )
            cleanup_incompatible_dataset(name, data_dir)

    display_results(final_results, discarded_structural, discarded_f1)


if __name__ == "__main__":
    main()
