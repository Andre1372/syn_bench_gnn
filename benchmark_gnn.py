"""Benchmark script for GNN models (GCN and GIN) on TUDatasets.

This module can be used in two ways:

1. **As a standalone script** (CLI):
       python benchmark_gnn.py --dataset PROTEINS MUTAG --runs 10 --epochs 50

2. **As an importable module**:
       from benchmark_gnn import benchmark_dataset, BenchmarkConfig

Public API
----------
BenchmarkConfig
    Dataclass holding all hyperparameters for training and evaluation.
benchmark_dataset(dataset_name, data_dir, config, verbose, project_root) -> dict | None
    Core benchmark function. If verbose=True, it logs progress, saves plots,
    and prints a summary table. Returns mean F1 scores.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.data_utils import (
    get_split_indices,
    sample_dataset,
    preprocess_and_save_original_dataset,
)
from src.log_utils import setup_console_logging
from src.train_gnn import GCNGraphClassifier, GINGraphClassifier, run_single_experiment

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Hyperparameters and settings for a GNN benchmark run.

    Attributes:
        runs: Number of independent training runs per dataset.
        epochs: Training epochs per run.
        batch_size: Mini-batch size for the DataLoaders.
        hidden_dim: Width of the hidden GNN layers.
        num_layers: Number of message-passing layers.
        dropout: Dropout probability applied inside the GNN.
        lr: Learning rate for the Adam optimizer.
        sample_size: Maximum number of graphs to subsample per run.
            Set to None to use the full dataset.
    """
    runs: int = 10
    epochs: int = 50
    batch_size: int = 16
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 5e-4
    sample_size: int | None = 400


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def build_models(num_classes: int, config: BenchmarkConfig, in_dim: int = 1) -> dict[str, Any]:
    """Instantiates GCN and GIN classifiers with the given configuration.

    Args:
        num_classes: Number of output classes.
        config: Benchmark configuration.
        in_dim: Input feature dimensionality (1 for dummy, N for log-binned).
    Returns:
        Dict mapping model name → model instance.
    """
    return {
        "GCN": GCNGraphClassifier(
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=num_classes,
            dropout=config.dropout,
        ),
        "GIN": GINGraphClassifier(
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=num_classes,
            dropout=config.dropout,
        ),
    }


# ------------------------------------------------------------------
# Core benchmark routine (importable)
# ------------------------------------------------------------------

def _run_benchmark_with_features(
    dataset_name: str,
    full_data_list: list[Data],
    num_classes: int,
    in_dim: int,
    feature_tag: str,
    config: BenchmarkConfig,
    verbose: bool,
    project_root: Path | None,
) -> dict[str, float] | None:
    """Internal helper: runs GCN+GIN benchmark for a given feature configuration.

    Args:
        dataset_name: TUDataset name (used for logging/plotting).
        full_data_list: Pre-processed dataset with features already applied.
        num_classes: Number of output classes.
        in_dim: Node feature dimensionality.
        feature_tag: Short string label (e.g. ``'dummy'`` or ``'log_bin'``).
        config: Benchmark hyperparameters.
        verbose: If True saves plots and logs tables.
        project_root: Root path for saving plots (used when verbose=True).
    Returns:
        Dict keyed as ``'{model}_{feature_tag}_mean_f1'`` or *None* on failure.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []

        run_iterator = range(config.runs)
        if verbose:
            run_iterator = tqdm(run_iterator, desc=f"{dataset_name} [{feature_tag}]")

        for run_id in run_iterator:
            rng = np.random.default_rng(run_id)
            data_list = (
                sample_dataset(full_data_list, config.sample_size, rng)
                if config.sample_size is not None
                else full_data_list
            )
            split_indices = get_split_indices(data_list, seed=run_id)

            for model_name, model in build_models(num_classes, config, in_dim=in_dim).items():
                run_metrics = run_single_experiment(
                    model=model,
                    dataset=data_list,
                    run_id=run_id,
                    device=device,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    split_indices=split_indices,
                )
                results.append({"Run": run_id, "Model": model_name, "Test F1": run_metrics["test_f1"]})
                if verbose:
                    history.extend(
                        {"Run": run_id, "Model": model_name, "Epoch": ep + 1, "Val F1": val_f1}
                        for ep, val_f1 in enumerate(run_metrics["val_f1_history"])
                    )

        results_df = pd.DataFrame(results)
        summary = results_df.groupby("Model")["Test F1"].agg(["mean", "std"]).reset_index()

        if verbose and project_root is not None:
            history_df = pd.DataFrame(history)
            results_dir = project_root / "results"
            tag = feature_tag
            _plot_f1_distribution(results_df, f"{dataset_name}_{tag}", config.runs,
                                  results_dir / f"{dataset_name}_{tag}_comparison.png")
            _plot_training_curves(history_df, f"{dataset_name}_{tag}",
                                  results_dir / f"{dataset_name}_{tag}_training_curves.png")
            logger.info(f"Summary for {dataset_name} [{tag}]:\n{summary.to_string(index=False)}")

        gcn_f1 = summary[summary["Model"] == "GCN"]["mean"].iloc[0]
        gin_f1 = summary[summary["Model"] == "GIN"]["mean"].iloc[0]
        return {
            f"gcn_{feature_tag}_mean_f1": gcn_f1,
            f"gin_{feature_tag}_mean_f1": gin_f1,
        }

    except Exception as e:
        logger.error(f"Benchmark [{feature_tag}] failed for '{dataset_name}': {e}")
        return None


def benchmark_dataset(
    dataset_name: str,
    data_dir: Path,
    config: BenchmarkConfig | None = None,
    verbose: bool = False,
    project_root: Path | None = None,
) -> dict[str, float] | None:
    """Runs a GNN benchmark (GCN and GIN) on a dataset using both dummy and log-binned features.

    Args:
        dataset_name: Name of the TUDataset.
        data_dir: Path to the dataset cache directory.
        config: Benchmark hyperparameters. Uses defaults if None.
        verbose: If True, uses tqdm, saves plots, and logs a summary table.
        project_root: Required only if verbose=True to save results.
    Returns:
        Dict with keys ``'gcn_dummy_mean_f1'``, ``'gin_dummy_mean_f1'``,
        ``'gcn_log_bin_mean_f1'``, ``'gin_log_bin_mean_f1'``
        or *None* if any step fails.
    """
    if config is None:
        config = BenchmarkConfig()
    if verbose:
        logger.info(f"--- Starting Benchmark for Dataset: {dataset_name} ---")
        if project_root is None:
            project_root = Path(__file__).parent.resolve()

    # --- Dummy features ---
    try:
        dummy_data, dummy_meta = preprocess_and_save_original_dataset(
            dataset_name, data_dir, use_log_bin_deg=False,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}' (dummy): {e}")
        return None

    dummy_res = _run_benchmark_with_features(
        dataset_name, dummy_data,
        num_classes=dummy_meta["num_classes"],
        in_dim=dummy_meta["in_dim"],
        feature_tag="dummy",
        config=config, verbose=verbose, project_root=project_root,
    )
    if dummy_res is None:
        return None

    # --- Log-binned degree features ---
    try:
        logbin_data, logbin_meta = preprocess_and_save_original_dataset(
            dataset_name, data_dir, use_log_bin_deg=True,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}' (log_bin): {e}")
        return None

    logbin_res = _run_benchmark_with_features(
        dataset_name, logbin_data,
        num_classes=logbin_meta["num_classes"],
        in_dim=logbin_meta["in_dim"],
        feature_tag="log_bin",
        config=config, verbose=verbose, project_root=project_root,
    )
    if logbin_res is None:
        return None

    return {**dummy_res, **logbin_res}


# ------------------------------------------------------------------
# Plotting helpers (private)
# ------------------------------------------------------------------

def _plot_f1_distribution(results_df: pd.DataFrame, dataset_name: str, num_runs: int, output_path: Path) -> None:
    """Saves a boxplot + strip-plot of test F1 distributions per model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.boxplot(
        data=results_df, x="Model", y="Test F1", hue="Model",
        palette="muted", showfliers=False, legend=False,
    )
    sns.stripplot(data=results_df, x="Model", y="Test F1", color="0.3", alpha=0.6, ax=ax)
    plt.title(f"Test F1 Scores on {dataset_name} ({num_runs} Runs)", fontsize=14)
    plt.ylabel("Test F1 Score", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to '{output_path}'.")
    plt.close()


def _plot_training_curves(history_df: pd.DataFrame, dataset_name: str, output_path: Path) -> None:
    """Saves a subplot of per-model validation F1 training curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    sns.set_theme(style="whitegrid")

    palette = {"GCN": ("skyblue", "blue"), "GIN": ("salmon", "red")}
    for i, model_name in enumerate(["GCN", "GIN"]):
        ax = axes[i]
        model_data = history_df[history_df["Model"] == model_name]
        light, dark = palette[model_name]

        sns.lineplot(data=model_data, x="Epoch", y="Val F1", units="Run",
                     estimator=None, alpha=0.4, ax=ax, color=light)
        sns.lineplot(data=model_data, x="Epoch", y="Val F1", ax=ax,
                     color=dark, linewidth=2, label="Mean")
        ax.set_title(f"{model_name} Training Curves", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Val F1 Score" if i == 0 else "", fontsize=11)
        ax.legend()

    num_runs = history_df["Run"].nunique()
    plt.suptitle(
        f"Validation F1 Progression across {dataset_name} ({num_runs} Runs)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Training curves saved to '{output_path}'.")
    plt.close()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GNN models on TUDatasets.")
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        help="One or more TUDataset names (e.g. PROTEINS MUTAG).")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of independent runs per dataset (default: 10).")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per run (default: 50).")
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size,
                        help=f"Mini-batch size (default: {BenchmarkConfig.batch_size}).")
    return parser.parse_args()


def _run_cli() -> None:
    """Entry point when the script is executed directly."""
    args = _parse_args()
    project_root = Path(__file__).parent.resolve()

    exp_suffix = "_".join(args.dataset).lower()
    if len(exp_suffix) > 50:
        exp_suffix = "multiple_datasets"
    setup_console_logging(project_root, f"benchmark_{exp_suffix}")

    config = BenchmarkConfig(runs=args.runs, epochs=args.epochs, batch_size=args.batch_size)
    data_dir = project_root / "data"

    for dataset_name in args.dataset:
        benchmark_dataset(dataset_name, data_dir, config, verbose=True, project_root=project_root)

    logger.info("All benchmarks completed.")


if __name__ == "__main__":
    _run_cli()
