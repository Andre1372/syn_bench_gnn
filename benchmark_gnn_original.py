"""GNN benchmark on original datasets only.

For each dataset the script:
  1. Preprocesses the original dataset with BOTH feature strategies
     (dummy all-ones  AND  log-binned degree one-hot).
  2. Trains GCN + GIN on each version for ``--gnn_runs`` independent runs.
  3. Prints a clean summary table of test F1 (mean ± std) per dataset,
     feature strategy and model.

Usage examples
--------------
  # Quick sanity check (1 epoch, 1 run):
  venv_sbg/bin/python benchmark_gnn_original.py --quick_test

  # Full run on selected datasets:
  venv_sbg/bin/python benchmark_gnn_original.py --datasets MUTAG BZR --gnn_runs 20
"""

import argparse
import csv
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.log_utils import setup_console_logging
from src.data_utils import (
    DatasetPT,
    get_split_indices,
    preprocess_and_save_original_dataset,
)
from src.train_gnn import evaluate_dataset

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ALL_DATASETS = ["MUTAG", "BZR", "DHFR", "Mutagenicity", "AIDS", "COX2", "DD", "IMDB-BINARY", "PROTEINS", "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR"]

FEATURE_STRATEGIES: dict[str, bool] = {
    "dummy":      False,
    "log_bin_deg": True,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN benchmark on original datasets — dummy vs log-binned features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        help="TUDataset names to benchmark. Use 'all' for all defaults.",
    )
    parser.add_argument(
        "--cut_datasets", type=int, default=None, metavar="N",
        help="Down-sample each dataset to at most N graphs (stratified).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per GNN run.",
    )
    parser.add_argument(
        "--gnn_runs", "-R", type=int, default=10,
        help="Independent GNN runs for variance estimation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed. Pass -1 for a fully stochastic run.",
    )
    parser.add_argument(
        "--force_reprocess", action="store_true",
        help="Re-preprocess even if the .pt file already exists.",
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="1 epoch, 1 run, tiny model — for fast smoke-testing.",
    )
    parser.add_argument(
        "--out_csv", type=str, default="results/benchmark_original.csv",
        help="Path to write the detailed CSV results.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pt_path(data_dir: Path, dataset_name: str, strategy: str) -> Path:
    """Returns the .pt path for a given dataset + feature strategy combination."""
    suffix = "_logbindeg" if strategy == "log_bin_deg" else ""
    return data_dir / dataset_name / f"{dataset_name}_original{suffix}.pt"


def _pt_filename(dataset_name: str, strategy: str) -> str:
    """Returns just the filename component for the given strategy."""
    suffix = "_logbindeg" if strategy == "log_bin_deg" else ""
    return f"{dataset_name}_original{suffix}.pt"


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[dict]) -> None:
    """Prints a formatted F1 summary table grouped by dataset and strategy."""

    # Collect unique keys
    datasets  = list(dict.fromkeys(r["dataset"]  for r in all_results))
    strategies = list(dict.fromkeys(r["strategy"] for r in all_results))
    models    = list(dict.fromkeys(r["model"]    for r in all_results))

    # Column widths
    col_ds  = max(len("Dataset"), max(len(d) for d in datasets))
    col_str = max(len("Features"), max(len(s) for s in strategies))
    col_mod = max(len("Model"), max(len(m) for m in models))
    col_f1  = 16  # "0.8234 ± 0.0123"

    sep = (
        f"+-{'-'*col_ds}-+-{'-'*col_str}-+-{'-'*col_mod}-"
        f"+-{'-'*col_f1}-+-{'-'*col_f1}-+"
    )
    header = (
        f"| {'Dataset':<{col_ds}} | {'Features':<{col_str}} | {'Model':<{col_mod}} "
        f"| {'Test F1 (mean±std)':<{col_f1}} | {'Val F1 (best,mean±std)':<{col_f1}} |"
    )

    print("\n" + "=" * len(sep))
    print("  GNN BENCHMARK — Original Datasets")
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)

    for ds in datasets:
        for strat in strategies:
            for model in models:
                rows = [
                    r for r in all_results
                    if r["dataset"] == ds and r["strategy"] == strat and r["model"] == model
                ]
                if not rows:
                    continue
                test_f1s = [r["test_f1"] for r in rows]
                val_f1s  = [r["val_best_f1"] for r in rows]
                tf1 = f"{np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}"
                vf1 = f"{np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}"
                print(
                    f"| {ds:<{col_ds}} | {strat:<{col_str}} | {model:<{col_mod}} "
                    f"| {tf1:<{col_f1}} | {vf1:<{col_f1}} |"
                )
            print(sep)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    project_root = Path(".")
    data_dir = project_root / "data"

    if "all" in args.datasets:
        args.datasets = ALL_DATASETS

    setup_console_logging(project_root, "benchmark_original")
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng() if args.seed == -1 else np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    epochs    = 1  if args.quick_test else args.epochs
    num_runs  = 1  if args.quick_test else args.gnn_runs
    hid_dim   = 16 if args.quick_test else 256
    num_layers = 1 if args.quick_test else 3

    all_run_results: list[dict] = []

    for dataset_name in args.datasets:
        logger.info("=" * 60)
        logger.info(f"DATASET: {dataset_name}")
        logger.info("=" * 60)

        # We need the same train/val/test split across both feature strategies
        # so that the comparison is perfectly fair. We build it once from
        # whichever .pt file exists first (both share the same graph ordering).
        split_indices = None

        for strategy_name, use_log_bin_deg in FEATURE_STRATEGIES.items():
            pt_path = _pt_path(data_dir, dataset_name, strategy_name)

            # ── Step 1: Preprocess ──────────────────────────────────────────
            if not pt_path.exists() or args.force_reprocess:
                logger.info(f"Preprocessing {dataset_name} with strategy='{strategy_name}'...")
                preprocess_and_save_original_dataset(
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    max_size=args.cut_datasets,
                    rng=rng,
                    use_log_bin_deg=use_log_bin_deg,
                    out_filename=_pt_filename(dataset_name, strategy_name),
                )
            else:
                logger.info(f"Found existing .pt for {dataset_name}/{strategy_name}, skipping preprocessing.")

            # ── Step 2: Load dataset ────────────────────────────────────────
            dataset_obj = DatasetPT(pt_path)
            data_list = [dataset_obj[i] for i in range(len(dataset_obj))]
            metadata  = dataset_obj.metadata

            in_dim      = metadata.get("in_dim", 1)
            num_classes = metadata.get("num_classes")

            # Build the shared split once (both strategies have same graphs/labels)
            if split_indices is None:
                split_indices = get_split_indices(data_list, seed=42)

            logger.info(
                f"  {strategy_name}: {len(data_list)} graphs, "
                f"in_dim={in_dim}, num_classes={num_classes}"
            )

            # ── Step 3: GNN training ────────────────────────────────────────
            gnn_config = {
                "num_runs"  : num_runs,
                "lr"        : 5e-4,
                "in_dim"    : in_dim,
                "hidden_dim": hid_dim,
                "num_layers": num_layers,
                "dropout"   : 0.1,
                "num_classes": num_classes,
            }

            total_steps = num_runs  # evaluate_dataset increments pbar once per run
            with tqdm(
                total=total_steps,
                desc=f"GNN [{dataset_name}/{strategy_name}]",
            ) as pbar:
                with logging_redirect_tqdm():
                    glob_res, _ = evaluate_dataset(
                        pt_path=pt_path,
                        gnn_config=gnn_config,
                        device=device,
                        split_indices=split_indices,
                        dataset_name=dataset_name,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        pbar=pbar,
                    )

            # Tag each row with the feature strategy for the summary table
            for row in glob_res:
                row["strategy"] = strategy_name

            all_run_results.extend(glob_res)

    # ── Summary ─────────────────────────────────────────────────────────────
    _print_summary(all_run_results)

    out_csv = Path(args.out_csv)
    _write_csv(all_run_results, out_csv)
    logger.info(f"Detailed results saved → {out_csv}")


if __name__ == "__main__":
    main()
