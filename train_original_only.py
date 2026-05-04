import logging
import csv
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.log_utils import setup_console_logging
from src.data_utils import get_split_indices, DatasetPT
from src.train_gnn import evaluate_dataset

ALL_DATASETS = [
    "BZR", "DHFR", "Mutagenicity", "MUTAG"
]

def _write_csv(rows: list[dict], path: Path) -> None:
    """Writes a list of dicts to a CSV file, creating parent dirs as needed."""
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    project_root = Path(".")
    
    # Setup logging
    session_tag = "train_original_only"
    setup_console_logging(project_root, session_tag)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in ALL_DATASETS:
        logger.info("─" * 60)
        logger.info(f"DATASET: {dataset_name}")
        logger.info("─" * 60)

        orig_pt_path = project_root / "data" / dataset_name / f"{dataset_name}_original.pt"
        if not orig_pt_path.exists():
            logger.error(f"Original dataset file not found: {orig_pt_path}. Skipping.")
            continue

        orig_dataset_obj = DatasetPT(orig_pt_path)
        original_data_list = [orig_dataset_obj[i] for i in range(len(orig_dataset_obj))]
        
        # Match the total number of runs used across all synthetic variants (R * V = 10 * 20 = 200)
        # We set num_runs to 200 to have the same statistical power as the full experiment
        orig_gnn_config = {
            "num_runs"  : 200,
            "lr"        : 5e-4,
            "in_dim"    : 1,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout"   : 0.1,
            "num_classes": orig_dataset_obj.metadata.get("num_classes"),
        }
        
        # Shared train/val/test split (fixed for comparability)
        split_indices = get_split_indices(original_data_list, seed=42)

        logger.info(f"Evaluating original dataset ({dataset_name})...")
        
        with tqdm(total=orig_gnn_config["num_runs"], desc=f"GNN [Original/{dataset_name}]") as pbar:
            with logging_redirect_tqdm():
                glob_res, pg_res = evaluate_dataset(
                    pt_path=orig_pt_path,
                    gnn_config=orig_gnn_config,
                    device=device,
                    split_indices=split_indices,
                    dataset_name=dataset_name,
                    epochs=50,
                    batch_size=16,
                    pbar=pbar,
                )
        
        _write_csv(glob_res, results_dir / f"gnn_eval_{dataset_name}_original.csv")
        _write_csv(pg_res,   results_dir / f"per_graph_{dataset_name}_original.csv")
        logger.info(f"Original results for {dataset_name} saved.")

    logger.info("=" * 60)
    logger.info("ALL ORIGINAL DATASETS PROCESSED SUCCESSFULLY")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
