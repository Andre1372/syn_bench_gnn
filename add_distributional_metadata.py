"""Script to patch distributional metadata into existing original dataset files.

For each ``*_original.pt`` file found under ``data/``, this script adds the three
keys required by the distributional-sampling path of
:func:`src.generate_datasets.generate_synthetic_variants`:

* ``is_discrete``   – boolean array of shape ``(num_features,)``
* ``per_class_stats`` – per-class stat matrices built from ``per_graph_statistics``
* ``stat_structure``  – feature-structure dict from :func:`src.data_utils.flatten_stats`

All pre-existing metadata keys are left untouched.  Datasets that already carry
all three keys are skipped automatically.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.log_utils import setup_console_logging
from src.data_utils import DatasetPT, flatten_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_distributional_metadata(metadata: dict) -> bool:
    """Returns True if all three distributional keys are already present."""
    return all(k in metadata for k in ("is_discrete", "per_class_stats", "stat_structure"))


def _build_distributional_metadata(
    orig_stats: list[dict],
    data_labels: list[int],
) -> tuple[np.ndarray, dict, dict]:
    """Derives distributional metadata from per-graph statistics.

    Args:
        orig_stats: List of per-graph statistics dicts (same order as the dataset).
        data_labels: Integer class label for each graph (same order).

    Returns:
        A tuple ``(is_discrete, per_class_stats, stat_structure)`` ready to be
        written into the dataset metadata.
    """
    if not orig_stats:
        raise ValueError("per_graph_statistics is empty – cannot build distributional metadata.")

    # --- Derive stat_structure from the first graph (structure is dataset-wide) ---
    _, stat_structure = flatten_stats(orig_stats[0])

    # --- Build is_discrete boolean array ---
    is_discrete_list: list[bool] = []
    for k, size in stat_structure.items():
        discrete = k in {"n_nodes", "n_edges"}
        is_discrete_list.extend([discrete] * size)
    is_discrete = np.array(is_discrete_list, dtype=bool)

    # --- Aggregate flat stat vectors per class ---
    # Use int keys to match the convention in data_utils.preprocess_and_save_original_dataset.
    per_class_accum: dict[int, list[np.ndarray]] = {}
    expected_size: int | None = None
    for per_graph_stats, y_val in zip(orig_stats, data_labels):
        flat_arr, _ = flatten_stats(per_graph_stats)
        # Guard: all graphs must produce the same flat-array length for vstack to work.
        if expected_size is None:
            expected_size = flat_arr.size
        elif flat_arr.size != expected_size:
            raise ValueError(
                f"Inconsistent stat array size: expected {expected_size}, "
                f"got {flat_arr.size} for a graph in class {y_val}. "
                "This usually means 'diameter' or another stat is None for some graphs. "
                "Re-run preprocess_and_save_original_dataset to regenerate the .pt file."
            )
        per_class_accum.setdefault(y_val, []).append(flat_arr)

    per_class_stats: dict[int, dict] = {}
    for y_val, stat_list in per_class_accum.items():
        stat_matrix = np.vstack(stat_list)
        per_class_stats[y_val] = {
            "num_samples": len(stat_list),
            "stat_matrix": stat_matrix,
        }

    return is_discrete, per_class_stats, stat_structure


def _patch_pt_file(pt_path: Path, logger: logging.Logger, force: bool = False) -> bool:
    """Patches a single ``*_original.pt`` file with distributional metadata.

    Args:
        pt_path: Path to the ``.pt`` file to patch.
        logger: Logger instance for progress/error messages.

    Returns:
        ``True`` if the file was patched, ``False`` if it was skipped.
    """
    logger.info(f"Loading: {pt_path}")
    payload: dict = torch.load(pt_path, weights_only=False)
    metadata: dict = payload.get("metadata", {})

    dataset_name = metadata.get("dataset_name", pt_path.parent.name)

    if _has_distributional_metadata(metadata) and not force:
        logger.info(f"[{dataset_name}] Already has distributional metadata – skipping (use --force to overwrite).")
        return False
    if _has_distributional_metadata(metadata) and force:
        logger.info(f"[{dataset_name}] Already has distributional metadata – re-patching (--force).") 

    orig_stats: list[dict] = metadata.get("per_graph_statistics", [])
    if not orig_stats:
        logger.error(
            f"[{dataset_name}] 'per_graph_statistics' not found in metadata. "
            f"Re-run with --process_original to regenerate the .pt file. Skipping."
        )
        return False

    # Recover labels from the collated data tensors
    dataset_obj = DatasetPT(pt_path)
    data_labels: list[int] = [int(dataset_obj[i].y.item()) for i in range(len(dataset_obj))]

    if len(data_labels) != len(orig_stats):
        logger.error(
            f"[{dataset_name}] Mismatch: {len(data_labels)} graphs vs "
            f"{len(orig_stats)} stat entries. Skipping."
        )
        return False

    logger.info(f"[{dataset_name}] Building distributional metadata from {len(orig_stats)} graphs ...")
    is_discrete, per_class_stats, stat_structure = _build_distributional_metadata(orig_stats, data_labels)

    logger.info(
        f"[{dataset_name}] stat_structure={list(stat_structure.keys())}  "
        f"num_features={is_discrete.size}  "
        f"classes={sorted(int(k) for k in per_class_stats)}"
    )

    # --- Patch metadata in-place (do NOT overwrite any existing keys) ---
    metadata["is_discrete"] = is_discrete
    metadata["per_class_stats"] = per_class_stats
    metadata["stat_structure"] = stat_structure
    payload["metadata"] = metadata

    torch.save(payload, pt_path)
    logger.info(f"[{dataset_name}] Patched and saved → {pt_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Scans data/ for original .pt files and patches distributional metadata."""
    parser = argparse.ArgumentParser(
        description="Add distributional metadata to existing original dataset .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-patch files that already carry distributional metadata (overwrites existing keys).",
    )
    args = parser.parse_args()

    project_root = Path(".")
    data_dir = project_root / "data"

    session_tag = "add_distributional_metadata"
    setup_console_logging(project_root, session_tag)
    logger = logging.getLogger(__name__)

    # Discover all *_original.pt files
    pt_files = sorted(data_dir.glob("**/*_original.pt"))

    if not pt_files:
        logger.warning(f"No *_original.pt files found under '{data_dir}'. Nothing to do.")
        return

    logger.info("=" * 60)
    logger.info(f"Found {len(pt_files)} original dataset file(s) to inspect.")
    logger.info("=" * 60)

    patched, skipped, failed = 0, 0, 0

    for pt_path in pt_files:
        logger.info("─" * 60)
        try:
            was_patched = _patch_pt_file(pt_path, logger, force=args.force)
            if was_patched:
                patched += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.exception(f"Unexpected error while processing '{pt_path}': {exc}")
            failed += 1

    logger.info("=" * 60)
    logger.info(
        f"Done. Patched: {patched}  |  Already up-to-date (skipped): {skipped}  |  Failed: {failed}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
