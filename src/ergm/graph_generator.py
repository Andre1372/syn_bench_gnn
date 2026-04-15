from __future__ import annotations
from typing import Any
from pathlib import Path
import logging
from contextlib import nullcontext
import csv
import numpy as np
import networkx as nx
import igraph as ig
import torch

from src.padma.graph_generator import generate_graph as padma_generate_graph
from src.graph_analysis import count_deg_moments

from src.data_utils import (
    networkx_to_igraph, 
    igraph_to_pytorch, 
    save_synthetic_dataset
)
from src.graph_analysis import analyze_single_graph
from src.ergm.graph_state import GraphState
from src.ergm.statistics import StatisticsManager, evaluate_significance_profile
from src.ergm.transformations import TransformationStrategy
from src.ergm.optimization import (
    ActivationStrategy, 
    EarlyStoppingTracker, 
    RobbinsMonroEstimator,
)
from src.ergm.mcmc import MetropolisHastingsSampler

import src.ergm.transformations
import src.ergm.optimization
import src.ergm.mcmc


# Try to import EstimationLogger from log_utils, fallback to a null context if not found
class EstimationLogger:
    """Telemetry logger managing file descriptors for ongoing stochastics."""

    def __init__(self, project_root: Path, experiment_name: str, num_stats: int) -> None:
        """Initializes the logger enforcing the project structural boundaries.

        Args:
            project_root: The root directory path of the project.
            experiment_name: The name used as a prefix for log and result files.
            num_stats: The total number of graph statistics being tracked.
        """
        self.experiment_name = experiment_name
        self.num_stats = num_stats
        
        # Enforce architecture semantics
        self._logs_dir = project_root / "logs"
        self._results_dir = project_root / "results"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # File paths (CSV telemetry goes to results, textual logs to logs)
        self._optim_csv_path = self._results_dir / f"{experiment_name}_optimization.csv"

    def __enter__(self) -> EstimationLogger:
        """Context manager entry point for safe file handling.

        Returns:
            The initialized EstimationLogger instance.
        """
        self._optim_file = self._optim_csv_path.open("w", encoding="utf-8", newline='')
        self._optim_writer = csv.writer(self._optim_file)

        # Headers updated with statistics errors, thetas and model mask
        theta_headers = [f"theta{i}" for i in range(self.num_stats)]
        error_headers = [f"error{i}" for i in range(self.num_stats)]
        mask_headers = [f"mask{i}" for i in range(self.num_stats)]
        
        self._optim_writer.writerow(
            ["step", "learning_rate", "acceptance_rate", "gradient_norm", "clipped", "j_metric"] 
            + error_headers + theta_headers + mask_headers
        )
        
        logger.info("Experiment started. File streams opened.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures absolute closure of file descriptors upon exiting the context.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
            exc_val: The exception instance.
            exc_tb: The traceback associated with the exception.
        """
        self._optim_file.close()
        
        if exc_type:
            logger.error(f"Experiment interrupted by {exc_type.__name__}: {exc_val}")
        else:
            logger.info("Experiment completed successfully. File streams closed.")

    def log_optimization_step(
        self,
        step: int,
        learning_rate: float,
        thetas: np.ndarray,
        gradient_norm: float,
        errors: np.ndarray,
        clipped: bool,
        acceptance_rate: float,
        mask: np.ndarray,
        j_metric: float = np.nan,
    ) -> None:
        """Logs the Robbins-Monro parameter bounds and component-wise errors.

        Args:
            step: Step index of the stochastic Robbins-Monro process.
            learning_rate: Actual LR scaling the bounds in the current step.
            thetas: Model parameters updated directly by stochastic steps.
            gradient_norm: Magnitudes identifying structural gradient deviations.
            errors: Difference between current and target statistics.
            clipped: Boolean flag indicating if the gradient was clipped.
            acceptance_rate: The MCMC acceptance rate during the current step.
            mask: Logic mask indicating if each stat is currently parameterized.
            j_metric: Current value of the multi-objective stopping metric J.
        """
        row = (
            [step, learning_rate, acceptance_rate, gradient_norm, clipped, j_metric] 
            + errors.tolist() + thetas.tolist() + mask.astype(int).tolist()
        )
        
        self._optim_writer.writerow(row)
        self._optim_file.flush()

def instantiate_estimation_components(config_dict: dict[str, Any], rng: np.random.Generator) -> tuple[StatisticsManager, TransformationStrategy, dict[str, Any], ActivationStrategy | None, EarlyStoppingTracker | None]:
    """Instantiates all estimation components from the experiment configuration."""
    # ── Statistics ────────────────────────────────────────────────────────────
    k = int(config_dict["statistics"])
    manager = StatisticsManager(k)

    # ── Transformation Strategy ───────────────────────────────────────────────
    strategy_config = config_dict["strategy"]
    strategy_class = getattr(src.ergm.transformations, strategy_config["name"])
    strategy: TransformationStrategy = strategy_class(rng=rng, **strategy_config.get("args", {}))

    # ── Estimator Hyperparameters ─────────────────────────────────────────────
    estimation_config = config_dict["estimation"]

    # ── Activation Strategy (Optional) ────────────────────────────────────────
    activation_config = config_dict.get("activation_strategy")
    activation_strategy = None
    if activation_config:
        activation_class = getattr(src.ergm.optimization, activation_config["name"])
        activation_strategy = activation_class(**activation_config.get("args", {}))

    # ── Early Stopping (Optional) ─────────────────────────────────────────────
    early_stopping_config = config_dict.get("early_stopping")
    early_stopping_tracker = None
    if early_stopping_config:
        early_stopping_tracker = EarlyStoppingTracker(**early_stopping_config)

    return manager, strategy, estimation_config, activation_strategy, early_stopping_tracker


def ergm_fit_sample(
    observed_nx: nx.Graph,
    config_dict: dict,
    init_method: str,
    project_root: Path,
    experiment_name: str,
    seed: int | None = None,
    n_samples: int | None = None,
    seed_only_init: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    save_results: bool = False,
) -> tuple[ig.Graph, list[ig.Graph]]:
    """Executes the full ERGM estimation pipeline for a single graph sample.

    Args:
        observed_nx: The observed graph in NetworkX format.
        config_dict: Parsed YAML configuration dictionary.
        init_method: Name of the initialisation strategy.
        n_samples: Number of samples to draw in Phase 3.
        seed: RNG seed for reproducible execution.
        project_root: Root directory of the project.
        experiment_name: Final name string used for log and result files.
        seed_only_init: If active, the seed is only used for the initial graph,
            leaving estimation and sampling randomized.
        show_progress: Whether to show progress bars.
        verbose: Whether to show detailed logs.
        save_results: If True, saves the initial and final graphs to a .pt file
            and computes/saves shape errors to a .csv file.
    Returns:
        A tuple (initial_igraph, list_of_posterior_igraphs).
    """
    logger = logging.getLogger(__name__)

    _log = logger.info if verbose else logger.debug

    # Derive RNGs
    if seed is None:
        _log(f"[{experiment_name}] No seed provided. Execution will be fully stochastic.")
        rng_init = np.random.default_rng()
        rng = np.random.default_rng()
    else:
        rng_init = np.random.default_rng(seed)
        if seed_only_init:
            _log(f"[{experiment_name}] Using seed {seed} ONLY for initialization. Rest of the pipeline is randomized.")
            rng = np.random.default_rng()
        else:
            _log(f"[{experiment_name}] Using fixed seed {seed} for the entire pipeline.")
            rng = rng_init

    # 1. State initialization
    observed_ig = networkx_to_igraph(observed_nx)
    observed_state = GraphState(observed_ig)
    
    # 2. Generate initial state
    _log(f"[{experiment_name}] Generating initial state via {init_method}")
    target_stats = {
        "n_nodes": observed_ig.vcount(),
        "n_edges": observed_ig.ecount(),
        "normalized_degree_moments": count_deg_moments(observed_ig).tolist()
    }
    initial_nx, _ = padma_generate_graph(target_stats, rng_init)
    initial_ig = networkx_to_igraph(initial_nx)
    _log(f"[{experiment_name}] Initial state generated with {initial_ig.vcount()} nodes and {initial_ig.ecount()} edges.")

    # 3. Initialize MCMC objects
    _log(f"[{experiment_name}] Initializing MCMC objects")
    initial_state = GraphState(initial_ig)
    manager, transformation_strategy, est_cfg, activation_strategy, early_stop_tracker = instantiate_estimation_components(config_dict, rng)

    # Calculate initial thetas from the Significance Profile (SP) of the initial graph
    _log(f"[{experiment_name}] Calculating initial thetas from Significance Profile (motifs up to k={manager._k})")
    initial_thetas = evaluate_significance_profile(initial_ig, manager._k, num_samples=25, rng=rng)

    sampler = MetropolisHastingsSampler(
        graph_state=initial_state,
        strategy=transformation_strategy,
        rng=rng,
        manager=manager,
        # initial_thetas=rng.normal(0, 1, size=manager.num_statistics),
        initial_thetas=initial_thetas*2,
    )
    estimator = RobbinsMonroEstimator(sampler=sampler, manager=manager)

    # 4. Fit Process
    _log(f"[{experiment_name}] Executing estimation pipeline")
    try:
        # Calculate target statistics once from the observed graph
        target_stats = manager.calculate_initial_values(observed_state)

        # File-based logging is only active if verbose is True
        logger_context = EstimationLogger(
            project_root,
            experiment_name,
            num_stats=manager.num_statistics,
        ) if verbose else nullcontext()

        with logger_context as estimation_logger:
            # Adaptive thinning based on the expected number of changes
            adaptive_thinning = int(10 * initial_state._num_edges * 4 / transformation_strategy.expected_changes(initial_state))

            _log(f"[{experiment_name}] Early stopping {f'enabled (patience={early_stop_tracker._patience}, alpha={early_stop_tracker._alpha:.2f})' if early_stop_tracker else 'disabled'}")

            final_theta = estimator.fit(
                target_stats=target_stats,
                thinning=int(adaptive_thinning / 1.5),
                updates=est_cfg["updates"],
                learning_rate=est_cfg["learning_rate"],
                lr_decay=est_cfg["lr_decay"],
                clip_gradient_norm=est_cfg["clip_gradient_norm"],
                covariance_update_interval=int(est_cfg["covariance_update_interval"] * 1.5),
                covariance_update_alpha=est_cfg["covariance_update_alpha"],
                activation_strategy=activation_strategy,
                estimation_logger=estimation_logger,
                early_stopping_tracker=early_stop_tracker,
                show_progress=show_progress,
            )
            _log(f"[{experiment_name}] Estimation phase concluded.")

            val_graphs = estimator.validate(
                final_samples=n_samples if n_samples is not None else est_cfg["final_samples"],
                thinning=int(adaptive_thinning * 1.5),
                show_progress=show_progress,
            )
            _log(f"[{experiment_name}] Validation phase concluded.")

            if save_results:
                _log(f"[{experiment_name}] Saving {len(val_graphs)} synthetic graphs...")
                
                # Persistence of graphs
                graph_list = [initial_ig] + val_graphs
                label = torch.zeros(1) # Default dummy label if not available
                
                pt_filename = f"{experiment_name}.pt"
                save_synthetic_dataset(
                    dataset_list=[igraph_to_pytorch(g, label) for g in graph_list],
                    output_dir=project_root / "synthetic_data" / "single",
                    filename=pt_filename,
                    extra_metadata={
                        "experiment": experiment_name,
                        "init": init_method,
                        "seed": seed
                    }
                )

                # Compute and save statistics for evolution analysis
                _log(f"[{experiment_name}] Computing absolute statistics...")
                stats_list = []
                observed_stats = analyze_single_graph(observed_ig)
                for i, fg in enumerate(val_graphs):
                    synth_stats = analyze_single_graph(fg)
                    row = {"synth_index": i}
                    row.update(synth_stats)
                    stats_list.append(row)

                csv_path = project_root / "results" / f"shape_errors_{experiment_name}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
                    writer.writeheader()
                    writer.writerows(stats_list)
                
                _log(f"[{experiment_name}] Shape errors saved to {csv_path}")

    except Exception as e:
        logger.error(f"[{experiment_name}] Estimation crashed due to: {e}", exc_info=True)
        raise
    
    _log(f"[{experiment_name}] Experiment successfully finalized.")

    return initial_ig, val_graphs