"""Script to download TUDatasets and filter them based on specific structural criteria.

This module provides functionality to programmatically download datasets from the
TUDataset collection, compute structural statistics (node/edge counts, etc.),
and filter them based on predefined constraints. Incompatible datasets are
automatically removed to save storage.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
from tqdm import tqdm


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
            "compatible": (num_graphs >= 160 and num_graphs <= 2000 and avg_nodes <= 1000 and num_classes == 2)
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


def display_results(stats_list: list[dict[str, Any]]) -> None:
    """Formats and prints the compatible dataset statistics to the console."""
    if not stats_list:
        logger.warning("No datasets matched the criteria.")
        return

    df = pd.DataFrame(stats_list)

    # Format output columns for readability
    display_df = pd.DataFrame({
        "dataset": df["dataset"],
        "graphs": df["graphs"],
        "average nodes +- std": df.apply(
            lambda r: f"{r['avg_nodes']:.2f} +- {r['std_nodes']:.2f}", axis=1
        ),
        "average edges +- std": df.apply(
            lambda r: f"{r['avg_edges']:.2f} +- {r['std_edges']:.2f}", axis=1
        ),
        "node features": df["node_features"]
    })

    print("\n" + "="*80)
    print("TUDatasets Matching Criteria (Graphs <= 2000, Avg Nodes <= 1000, Classes == 2)")
    print("="*80)
    print(display_df.to_string(index=False))
    print("="*80 + "\n")


def main() -> None:
    """Main execution flow for downloading and filtering TUDatasets."""
    dataset_names = [
        "AIDS", "BZR", "BZR_MD", "COX2", "COX2_MD", "DHFR", "DHFR_MD", "ER_MD", "MUTAG",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "DD", "ENZYMES", "KKI", "OHSU", "Peking_1",
        "PROTEINS", "Cuneiform", "FIRSTMM_DB", "Letter_high", "Letter-low", "Letter-med", "MSRC_9",
        "MSRC_21", "MSRC_21C", "dblp_ct1", "dblp_ct2", "facebook_ct1", "highschool_ct1",
        "IMDB-BINARY", "IMDB-MULTI", "infectious_ct1", "mit_ct1", "REDDIT-BINARY", "tumblr_ct1",
        "SYNTHETIC", "Synthie"
    ]

    project_root = Path(__file__).parent.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    compatible_results: list[dict[str, Any]] = []
    logger.info(f"Processing {len(dataset_names)} TUDatasets...")

    for name in tqdm(dataset_names, desc="Downloading and analyzing"):
        stats = fetch_dataset_stats(name, data_dir)

        if stats and stats["compatible"]:
            compatible_results.append(stats)
        else:
            logger.info(f"Dataset '{name}' does not meet criteria. Cleaning up...")
            cleanup_incompatible_dataset(name, data_dir)

    display_results(compatible_results)


if __name__ == "__main__":
    main()
