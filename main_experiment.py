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
from src.enc_dec_dataset import KNOWN_SAMPLERS
from src.train_gnn import evaluate_dataset
from src.node_features import is_per_graph_strategy

ALL_DATASETS = [
        # "BZR", "DHFR", "Mutagenicity", "MUTAG", "Cuneiform", "MSRC_9", "MSRC_21", "MSRC_21C"
        "AIDS", "BZR", "COX2", "DHFR", "Mutagenicity", "MUTAG", "NCI1", "PROTEINS", "OHSU", "ENZYMES"
        # "AIDS", "BZR", "COX2", "DHFR", "Mutagenicity", "MUTAG", "NCI1", "NCI109", "DD", "PROTEINS", "Cuneiform", "MSRC_9", "MSRC_21", "MSRC_21C", "KKI", "OHSU", "ENZYMES"
    ]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset, multi-method synthetic graph generation and GNN evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=ALL_DATASETS,
        help="One or more TUDataset names to use as baseline (e.g. PROTEINS MUTAG). Defaults to all datasets defined in ALL_DATASETS.",
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

    # ── Generation ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["dummyNodes", "dummyEdges", "padma", "anndg", "anndgE", "ergm"],
        metavar="METHOD",
        help=f"Generation methods to run.  Supported: {', '.join(KNOWN_METHODS)}.  Defaults to all known methods. Note: 'ergm' is only compatible with no_sampler (requires observed graph).",
    )
    parser.add_argument(
        "--num_synth_datasets",
        "-V",
        type=int,
        default=20,
        help="Number of independent synthetic variants V to generate per (dataset, method) pair.",
    )
    parser.add_argument(
        "--distribution_sampler",
        type=str,
        nargs="+",
        default=list(KNOWN_SAMPLERS),
        metavar="SAMPLER",
        choices=list(KNOWN_SAMPLERS) + ["none"],
        help=(
            f"One or more encoder-decoders for distributional stat sampling. "
            f"Supported: {', '.join(sorted(KNOWN_SAMPLERS))}. "
            f"Pass 'none' to run without any sampler. "
            f"Defaults to all known samplers. "
            f"Per-graph feature strategies ('random_sample', 'degree_ordered', 'neighbor_degree_ordered') "
            f"are always run without a sampler; incompatible combinations are skipped automatically."
        ),
    )

    # ── Node features ───────────────────────────────────────────────────────
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["random_sample", "degree_ordered", "neighbor_degree_ordered"],
        choices=["constant", "log_bin_deg", "random_sample", "degree_ordered", "neighbor_degree_ordered"],
        help=(
            "One or more node feature strategies to run. "
            "'constant': all-ones dummy (in_dim=1). "
            "'log_bin_deg': one-hot log-binned degree. "
            "'random_sample', 'degree_ordered', 'neighbor_degree_ordered': per-graph "
            "transplanting strategies (always run without a sampler). "
            "Defaults to all three per-graph strategies."
        ),
    )

    # ── GNN training & evaluation ────────────────────────────────────────────
    parser.add_argument(
        "--process_original",
        action="store_true",
        help="Pre-process (and sample) the original dataset even if it exists. Also enables evaluation.",
    )
    parser.add_argument(
        "--gnn_runs",
        "-R",
        type=int,
        default=10,
        help="Number of independent GNN training runs per dataset for variance estimation.",
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
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
        "--quick_test",
        action="store_true",
        help="Run a fast functional test (1 epoch, 1 run, small models).",
    )
    return parser.parse_args()


PER_GRAPH_FEATURES = frozenset({"random_sample", "degree_ordered", "neighbor_degree_ordered"})


def _build_run_combinations(features_list: list[str], samplers_list: list[str] | None) -> list[tuple[str, str | None]]:
    """Returns valid (feature, sampler) combinations.

    Per-graph feature strategies ('random_sample', 'degree_ordered',
    'neighbor_degree_ordered') are always run without a distribution sampler
    (sampler=None).  If samplers are also requested, a warning is logged and
    the incompatible combinations are silently skipped.

    Args:
        features_list: List of requested feature strategies.
        samplers_list: List of requested samplers, or None for direct (no-sampler) mode.

    Returns:
        A list of (feature, sampler_or_None) tuples representing valid combinations.
    """
    _log = logging.getLogger(__name__)
    global_samplers: list[str | None] = samplers_list if samplers_list is not None else [None]

    per_graph_feats = [f for f in features_list if f in PER_GRAPH_FEATURES]
    if per_graph_feats and samplers_list is not None:
        _log.warning(
            "Per-graph feature strategies (%s) are incompatible with distribution samplers "
            "and will always run without a sampler. The requested sampler(s) are ignored for them.",
            ", ".join(per_graph_feats),
        )

    combos: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for feat in features_list:
        if feat in PER_GRAPH_FEATURES:
            if (feat, None) not in seen:
                combos.append((feat, None))
                seen.add((feat, None))
        else:
            for samp in global_samplers:
                if (feat, samp) not in seen:
                    combos.append((feat, samp))
                    seen.add((feat, samp))
    return combos


def _combo_tag(feature: str, sampler: str | None) -> str:
    """Returns a compact string tag for a (feature, sampler) combination."""
    samp_part = sampler if sampler is not None else "nosampler"
    return f"{samp_part}_{feature}"


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

    # Convert the sentinel value "none" to actual None (no-sampler mode).
    if args.distribution_sampler == ["none"]:
        args.distribution_sampler = None

    # Build all valid (feature, sampler) combinations upfront.
    run_combos = _build_run_combinations(args.features, args.distribution_sampler)

    # Validate ergm compatibility
    if "ergm" in args.methods:
        incompatible_samplers = sorted({s for _, s in run_combos if s is not None})
        if incompatible_samplers:
            print(
                f"[ERROR] Method 'ergm' is incompatible with distribution samplers "
                f"({', '.join(incompatible_samplers)}). "
                f"ergm requires '--distribution_sampler' to be omitted or all features "
                f"to be per-graph strategies (random_sample, degree_ordered, neighbor_degree_ordered). "
                f"Aborting."
            )
            raise SystemExit(1)

    n_datasets = len(args.dataset)
    n_methods = len(args.methods)
    n_combos = len(run_combos)
    features_tag = "-".join(sorted(args.features))
    samplers_tag = "-".join(sorted(args.distribution_sampler)) if args.distribution_sampler else "nosampler"
    session_tag = (
        f"multi_exp_{n_datasets}D_{n_methods}M_{n_combos}C_{features_tag}_s{samplers_tag}"
        if n_datasets > 1
        else f"exp_{args.dataset[0]}_{n_methods}M_{n_combos}C_{features_tag}_s{samplers_tag}"
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
        logger.info(f"PHASE A: Generating {args.num_synth_datasets} synthetic variants for {n_datasets} dataset(s) x {n_methods} method(s) x {n_combos} combination(s)")
        logger.info("=" * 60)

        for dataset_name in args.dataset:
            generated_origs = set()
            for feature, sampler in run_combos:
                if is_per_graph_strategy(feature):
                    orig_fname = f"{dataset_name}_original_native.pt"
                else:
                    orig_fname = f"{dataset_name}_original_{feature}.pt"

                orig_pt_path = project_root / "data" / dataset_name / orig_fname
                
                process_this = args.process_original
                if not orig_pt_path.exists():
                    process_this = True
                    logger.info(f"Dataset {dataset_name} ({orig_fname}) not found, forcing processing.")
                
                if process_this and orig_fname not in generated_origs:
                    preprocess_and_save_original_dataset(
                        dataset_name,
                        project_root / "data",
                        max_size=args.cut_datasets,
                        rng=rng,
                        feature_type=feature,
                        out_filename=orig_fname,
                    )
                    generated_origs.add(orig_fname)
                elif orig_fname in generated_origs:
                    logger.info(f"Using just-generated original dataset for {dataset_name} ({orig_fname}).")
                else:
                    logger.info(f"Using existing original dataset for {dataset_name} ({orig_fname}) (skip re-processing).")

                for method in args.methods:
                    tag = _combo_tag(feature, sampler)
                    logger.info(f"--- Generating: dataset={dataset_name}  method={method}  features={feature}  sampler={sampler} ---")
                    output_dir = project_root / "synthetic_data"
                    generate_synthetic_variants(
                        dataset_name=dataset_name,
                        method=method,
                        num_variants=args.num_synth_datasets,
                        rng=rng,
                        project_root=project_root,
                        output_dir=output_dir,
                        num_workers=args.num_workers,
                        distribution_sampler=sampler,
                        feature_type=feature,
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

    gnn_config_base = {
        "num_runs"  : 1 if args.quick_test else args.gnn_runs,
        "lr"        : 5e-4,
        "hidden_dim": 16 if args.quick_test else 256,
        "num_layers": 1 if args.quick_test else 3,
        "dropout"   : 0.1,
        "epochs"    : 1 if args.quick_test else 50,
        "batch_size": 16,
    }

    for dataset_name in args.dataset:
        logger.info("─" * 60)
        logger.info(f"DATASET: {dataset_name}")
        logger.info("─" * 60)

        evaluated_origs = set()

        for feature, sampler in run_combos:
            tag = _combo_tag(feature, sampler)
            logger.info(f"  Combination: features={feature}  sampler={sampler}")

            if is_per_graph_strategy(feature):
                orig_fname = f"{dataset_name}_original_native.pt"
                orig_eval_tag = "native"
            else:
                orig_fname = f"{dataset_name}_original_{feature}.pt"
                orig_eval_tag = feature

            orig_pt_path = project_root / "data" / dataset_name / orig_fname
            
            if not orig_pt_path.exists():
                preprocess_and_save_original_dataset(
                    dataset_name,
                    project_root / "data",
                    max_size=args.cut_datasets,
                    rng=rng,
                    feature_type=feature,
                    out_filename=orig_fname,
                )

            orig_dataset_obj = DatasetPT(orig_pt_path)
            # The .pt already contains the (possibly cut) dataset — no extra sampling needed.
            original_data_list = [orig_dataset_obj[i] for i in range(len(orig_dataset_obj))]

            gnn_config = {
                **gnn_config_base,
                "in_dim"     : orig_dataset_obj.metadata.get("in_dim", 1),
                "num_classes": orig_dataset_obj.metadata.get("num_classes"),
            }

            # Shared train/val/test split (fixed across all methods for comparability)
            split_indices = get_split_indices(original_data_list, seed=42)

            # Optionally evaluate on original dataset and save the baseline CSV
            if args.process_original:
                if orig_eval_tag not in evaluated_origs:
                    logger.info(f"Evaluating original dataset ({dataset_name}) [feature={orig_eval_tag}]...")

                    # Match the total number of runs used across all synthetic variants (R * V)
                    orig_gnn_config = gnn_config.copy()
                    orig_gnn_config["num_runs"] = gnn_config["num_runs"] * args.num_synth_datasets

                    with tqdm(total=orig_gnn_config["num_runs"], desc=f"GNN [Original/{dataset_name}/{orig_eval_tag}]") as pbar:
                        with logging_redirect_tqdm():
                            glob_res, pg_res = evaluate_dataset(
                                pt_path=orig_pt_path,
                                gnn_config=orig_gnn_config,
                                device=device,
                                split_indices=split_indices,
                                dataset_name=dataset_name,
                                pbar=pbar,
                            )
                    _write_csv(glob_res, results_dir / f"gnn_global_{dataset_name}_original_{orig_eval_tag}.csv")
                    _write_csv(pg_res,   results_dir / f"gnn_per_graph_{dataset_name}_original_{orig_eval_tag}.csv")
                    logger.info(f"Original results saved → gnn_global_{dataset_name}_original_{orig_eval_tag}.csv")
                    evaluated_origs.add(orig_eval_tag)
                else:
                    logger.info(f"Original dataset ({dataset_name}) [feature={orig_eval_tag}] already evaluated. Skipping.")

            # Evaluate each synthetic method
            for method in args.methods:
                logger.info(f"Evaluating method: {method} [combo={tag}]")
                synth_dir = project_root / "synthetic_data"
                eval_tasks = load_all_synthetic_variants(
                    synth_dir,
                    dataset_name,
                    method=method,
                    sampler=sampler,
                    feature=feature,
                )

                if not eval_tasks:
                    logger.warning(f"No synthetic data found for dataset={dataset_name} method={method} combo={tag} - skipping.")
                    continue

                all_results: list[dict] = []
                all_pg_results: list[dict] = []

                with tqdm(total=len(eval_tasks)*gnn_config["num_runs"], desc=f"GNN [{method.upper()}/{dataset_name}/{tag}]") as pbar:
                    with logging_redirect_tqdm():
                        for pt_path in eval_tasks:
                            logger.debug(f"Evaluating path: {pt_path}.")
                            variant_meta = DatasetPT(pt_path).metadata
                            variant_in_dim = variant_meta.get("in_dim", gnn_config["in_dim"])
                            variant_gnn_config = {**gnn_config, "in_dim": variant_in_dim}
                            glob_res, pg_res = evaluate_dataset(
                                pt_path=pt_path,
                                gnn_config=variant_gnn_config,
                                device=device,
                                split_indices=split_indices,
                                dataset_name=dataset_name,
                                pbar=pbar,
                            )
                            all_results.extend(glob_res)
                            all_pg_results.extend(pg_res)

                csv_stem = f"{dataset_name}_{method}_{tag}"
                _write_csv(all_results,    results_dir / f"gnn_global_{csv_stem}.csv")
                _write_csv(all_pg_results, results_dir / f"gnn_per_graph_{csv_stem}.csv")
                logger.info(f"Results saved → gnn_global_{csv_stem}.csv / gnn_per_graph_{csv_stem}.csv")

    logger.info("=" * 60)
    logger.info("ALL DATASETS PROCESSED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
