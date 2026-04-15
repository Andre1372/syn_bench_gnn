"""GNN model definitions and training/evaluation routines."""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, Sequential, global_mean_pool, global_max_pool

from src.data_utils import make_loaders, DatasetPT, remove_features
from src.graph_analysis import per_graph_statistics, aggregate_statistics_per_class

logger = logging.getLogger(__name__)


class GCNGraphClassifier(nn.Module):
    """GCN model for graph classification."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float) -> None:
        """Initializes the GCN classifier.

        Args:
            in_dim: Dimension of the input node features.
            hidden_dim: Dimension of the hidden layers.
            num_layers: Total number of GCN layers.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        # Define GCN layers in a PyG Sequential container
        layers = []
        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            layers.append((GCNConv(dim_in, hidden_dim), 'x, edge_index -> x'))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.gnn = Sequential('x, edge_index', layers)

        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GCN classifier."""
        x = self.gnn(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(x)


class GINGraphClassifier(nn.Module):
    """GIN model for graph classification."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float) -> None:
        """Initializes the GIN classifier.

        Args:
            in_dim: Dimension of the input node features.
            hidden_dim: Dimension of the hidden layers.
            num_layers: Total number of GIN layers.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            # GIN using an internal MLP
            mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append((GINConv(mlp), 'x, edge_index -> x'))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.gnn = Sequential('x, edge_index', layers)

        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GIN classifier."""
        x = self.gnn(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(x)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Runs a single training epoch.

    Args:
        model: The GNN model to train.
        loader: DataLoader providing minibatches.
        optimizer: Optimizer used for gradient updates.
        device: Device on which to run the computation.
    Returns:
        The average cross-entropy loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass: compute logits for the batch
        logits = model(batch.x, batch.edge_index, batch.batch)
        
        # Compute loss using the pre-instantiated criterion class
        loss = criterion(logits, batch.y)
        
        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics weighted by graph count in batch
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs


def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float, float]:
    """Evaluates the model on an entire loader.

    Args:
        model: The GNN model to evaluate.
        loader: DataLoader providing minibatches.
        device: Device on which to run the computation.
    Returns:
        A tuple of (avg_loss, accuracy, macro_f1, roc_auc).
        roc_auc is NaN if it cannot be computed (e.g. single class present).
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass: compute logits and probabilities
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            # Compute loss
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

            all_labels.extend(batch.y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total_graphs
    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    try:
        probs_tensor = torch.tensor(all_probs)
        num_model_classes = probs_tensor.shape[-1]
        
        if num_model_classes == 2:
            roc_auc = float(roc_auc_score(all_labels, probs_tensor[:, 1].numpy()))
        else:
            all_classes = list(range(num_model_classes))
            roc_auc = float(roc_auc_score(
                all_labels, 
                probs_tensor.numpy(), 
                multi_class="ovr", 
                average="macro", 
                labels=all_classes
            ))
    except (ValueError, TypeError) as e:
        logger.warning("ROC AUC could not be computed: %s", e)
        roc_auc = float("nan")

    return avg_loss, accuracy, macro_f1, roc_auc


def get_per_graph_predictions(model: nn.Module, dataset: list[Data], device: torch.device, batch_size: int) -> np.ndarray:
    """Evaluates the model on the entire dataset sequentially to track per-graph correctness.

    Args:
        model: The trained GNN model.
        dataset: The complete list of PyG Data objects (train + val + test).
        device: The device to run inference on.
        batch_size: The batch size for the DataLoader.
    Returns:
        A NumPy array of integers (0 or 1) indicating correct classification.
    Raises:
        RuntimeError: If the model inference fails due to memory or dimension mismatch.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_graph_correct: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                logits = model(batch.x, batch.edge_index, batch.batch)
                preds = logits.argmax(dim=-1)
                correct = (preds == batch.y).long()
                per_graph_correct.extend(correct.cpu().tolist())
            except Exception as e:
                raise RuntimeError(f"Model inference failed: {e}") from e

    return np.array(per_graph_correct, dtype=np.int64)


def run_single_experiment(model: nn.Module, dataset: list[Data], run_id: int, device: torch.device, epochs: int, batch_size: int, lr: float, split_indices: tuple[list[int], list[int], list[int]]) -> dict[str, Any]:
    """Trains and evaluates a GNN model.
    
    Args:
        model: The GNN model to train.
        dataset: List of PyG Data objects (the full dataset to split).
        run_id: Integer identifier for the run.
        device: Device on which to run training.
        epochs: Number of training epochs.
        batch_size: Batch size for data loaders.
        lr: Learning rate for the optimizer.
        split_indices: Tuple of lists containing train/val/test split indices.
    Returns:
        A dictionary with keys: 'run_id', 'val_best_f1', 'test_f1',
        'test_acc', 'test_roc_auc'.
    """
    model = model.to(device)
    
    train_loader, val_loader, test_loader = make_loaders(dataset=dataset, split_indices=split_indices, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        _, _, val_f1, _ = eval_epoch(model, val_loader, criterion, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 10 == 0:
            logger.debug(
                "Run %d | Epoch %d/%d | train_loss=%.4f | val_f1=%.4f",
                run_id, epoch + 1, epochs, train_loss, val_f1,
            )

    # Restore best checkpoint and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_acc, test_f1, test_roc_auc = eval_epoch(model, test_loader, criterion, device)
    logger.info(
        "Run %d finished | best_val_f1=%.4f | test_acc=%.4f | test_f1=%.4f | test_roc_auc=%.4f",
        run_id, best_val_f1, test_acc, test_f1, test_roc_auc,
    )
    
    return {
        "run_id": run_id,
        "val_best_f1": best_val_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "test_roc_auc": test_roc_auc,
    }


def evaluate_dataset(
    pt_path: str | Path,
    gnn_config: dict[str, Any],
    device: torch.device,
    split_indices: tuple[list[int], list[int], list[int]],
    dataset_name: str,
    epochs: int,
    batch_size: int,
    pbar: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs a full suite of GNN evaluations loading data from a .pt file.

    Args:
        pt_path: Path to the .pt file containing the dataset and metadata.
        gnn_config: Dictionary of model parameters and hyperparameters.
        device: Torch device to run training on.
        split_indices: Pre-computed ``(train_idx, val_idx, test_idx)`` index lists.
        dataset_name: Dataset name written verbatim to each result row.
        epochs: Number of training epochs per GNN run.
        batch_size: Mini-batch size for DataLoaders.
        pbar: Optional tqdm progress bar to update (one step per run).
    Returns:
        A tuple containing:
            - A list of global result dictionaries (one per model × run).
            - A list of per-graph evaluation dictionaries.
    """
    train_idx, val_idx, test_idx = split_indices

    split_map: dict[int, str] = {idx: "Train" for idx in train_idx}
    split_map.update({idx: "Val" for idx in val_idx})
    split_map.update({idx: "Test" for idx in test_idx})

    model_registry = {"GCN": GCNGraphClassifier, "GIN": GINGraphClassifier}
    global_results: list[dict[str, Any]] = []
    per_graph_results: list[dict[str, Any]] = []

    dataset_obj = DatasetPT(pt_path)
    data_list = [dataset_obj[i] for i in range(len(dataset_obj))]
    metadata = dataset_obj.metadata

    source = metadata.get("source", "original")
    if "variant_idx" in metadata:
        source = f"{source}_{metadata['variant_idx']}"
        
    seeds = metadata.get("seeds", None)

    global_stats_list: list[dict[str, Any]] = []
    per_graph_stats: list[dict[str, float]] = []

    if "per_graph_statistics" in metadata:
         per_graph_stats = metadata["per_graph_statistics"]
    if "aggregate_statistics_per_class" in metadata:
        for cls_label, cls_stats in metadata["aggregate_statistics_per_class"].items():
            class_dict = {"graph_class": cls_label}
            for stat_name, stat_val in cls_stats.items():
                class_dict[stat_name] = stat_val
            global_stats_list.append(class_dict)
                
    if not per_graph_stats:
        logger.debug("Computing network statistics locally for %s...", source)
        local_pg_stats = per_graph_statistics(data_list)
        per_graph_stats = local_pg_stats
        agg_class = aggregate_statistics_per_class(data_list, local_pg_stats)
        for cls_label, cls_stats in agg_class.items():
            class_dict = {"graph_class": cls_label}
            for stat_name, stat_val in cls_stats.items():
                class_dict[stat_name] = stat_val
            global_stats_list.append(class_dict)

    # Accumulate correctness counts across all runs for each model
    num_graphs = len(data_list)
    n_correct: dict[str, np.ndarray] = {
        model_name: np.zeros(num_graphs, dtype=int)
        for model_name in model_registry.keys()
    }

    for run_id in range(gnn_config["num_runs"]):
        for model_name, model_cls in model_registry.items():
            # Each run needs a fresh model instance
            model = model_cls(
                in_dim=gnn_config["in_dim"],
                hidden_dim=gnn_config["hidden_dim"],
                num_layers=gnn_config["num_layers"],
                num_classes=gnn_config["num_classes"],
                dropout=gnn_config["dropout"],
            )

            run_result = run_single_experiment(
                model=model,
                dataset=data_list,
                run_id=run_id,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=gnn_config["lr"],
                split_indices=split_indices,
            )

            n_correct[model_name] += get_per_graph_predictions(model, data_list, device, batch_size)

            for cls_stats in global_stats_list:
                global_results.append({
                    "dataset": dataset_name,
                    "source": source,
                    "model": model_name,
                    **run_result,
                    **cls_stats,
                })
        
        if pbar: pbar.update(1)

    # Assemble one record per graph with the accumulated counts
    for graph_idx in range(num_graphs):
        graph_res: dict[str, Any] = {
            "dataset": dataset_name,
            "source": source,
            "graph_idx": graph_idx,
            "label": int(data_list[graph_idx].y.item()) if data_list[graph_idx].y is not None else None,
            "split": split_map.get(graph_idx, "Unknown"),
            "seed": seeds[graph_idx] if seeds else None,
        }

        for model_name in model_registry.keys():
            mean_accuracy = float(n_correct[model_name][graph_idx]) / gnn_config["num_runs"]
            graph_res[f"mean_acc_{model_name}"] = round(mean_accuracy, 4)

        if per_graph_stats:
            graph_res.update(per_graph_stats[graph_idx])

        per_graph_results.append(graph_res)

    return global_results, per_graph_results
