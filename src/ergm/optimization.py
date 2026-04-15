"""Module for parameter estimation using the Robbins-Monro stochastic approximation algorithm."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import igraph as ig
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.ergm.graph_state import GraphState
from src.ergm.mcmc import MetropolisHastingsSampler, welford_update
from src.ergm.statistics import StatisticsManager
if TYPE_CHECKING: # Because of cicling imports
    from src.log_utils import EstimationLogger

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Early Stopping Tracker
# -----------------------------------------------------------------------------

class EarlyStoppingTracker:
    """Tracks a multi-objective stopping metric J for the Robbins-Monro optimization loop.

    At each step t, the metric is computed as a direct convex combination of two
    relative error signals without variance normalization:

        J_t = alpha * e1_t + (1 - alpha) * e2_t

    Where:
        e1_t   -- Relative RMSE across all model and diagnostic statistics at step t.
        e2_t   -- Absolute deviation of the acceptance rate from its target.

    The tracker maintains the best-seen theta vector and graph state for optional
    restoration after early stopping is triggered.
    """

    def __init__(self, patience: int = 20, alpha: float = 0.75, acc_rate_target: float = 0.2) -> None:
        """Initializes the tracker for early stopping.

        Args:
            patience: Number of consecutive non-improving steps before stopping.
            alpha: Weight for the structural error term (e1). Must be in (0, 1).
            acc_rate_target: Target acceptance rate for the MCMC chain.
            
        Raises:
            ValueError: If alpha is not in the valid (0, 1) range or patience <= 0.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if patience <= 0:
            raise ValueError(f"patience must be strictly positive, got {patience}.")

        self._patience = patience
        self._alpha = alpha
        self._acc_rate_target = acc_rate_target

        self._step_count: int = 0
        self._best_j: float = float("inf")
        self._wait_count: int = 0
        self._inverse_normalizer: np.ndarray | None = None
        
        self.best_theta: np.ndarray | None = None
        self.best_graph: ig.Graph | None = None
        self.best_step: int | None = None

    def initialize(self, target_stats: np.ndarray) -> None:
        """Initializes constant normalization factors derived from target statistics."""
        self._inverse_normalizer = 1.0 / np.maximum(np.abs(target_stats), 1.0)

    def step(
        self,
        current_errors: np.ndarray,
        acc_rate: float,
        current_thetas: np.ndarray,
        current_graph: ig.Graph
    ) -> tuple[bool, float]:
        """Records a new optimization step and checks the stopping criterion.

        Computes e1 as the Relative RMSE using pre-calculated normalization factors,
        and e2 as the absolute deviation of the acceptance rate. J is their
        direct convex combination.

        Args:
            current_errors: The raw error vector (simulated - target).
            acc_rate: MCMC acceptance rate at this step.
            current_thetas: Current theta parameter vector.
            current_graph: Current graph state from the MCMC sampler.
            
        Returns:
            A tuple (should_stop, current_j) where should_stop is True if the 
            patience has been exhausted, and current_j is the current metric value.
        """
        self._step_count += 1
        
        # Compute e1 (Relative RMSE)
        relative_errors = current_errors * self._inverse_normalizer
        e1 = float(np.sqrt(np.mean(np.square(relative_errors))))

        # Compute e2 (Acceptance rate absolute deviation)
        e2 = abs(acc_rate - self._acc_rate_target)

        # Compute J (Direct combination)
        j = self._alpha * e1 + (1.0 - self._alpha) * e2

        # Update best state
        if j < self._best_j:
            self._best_j = j
            self._wait_count = 0
            self.best_theta = current_thetas.copy()
            self.best_graph = current_graph.copy()
            self.best_step = self._step_count
        else:
            self._wait_count += 1

        # Check stopping condition
        should_stop = self._wait_count >= self._patience
        return should_stop, j


# -----------------------------------------------------------------------------
# Activation Strategy hierarchy
# -----------------------------------------------------------------------------

class ActivationStrategy(ABC):
    """Abstract interface for statistic-activation strategies.

    Concrete implementations control which statistics (indices) participate in gradient updates and in acceptance probability at any given optimisation step.
    """

    @abstractmethod
    def initialize(self, initial_stats: np.ndarray, target_stats: np.ndarray) -> np.ndarray:
        """Computes and returns the initial boolean active mask.

        Args:
            initial_stats: Statistics measured on the starting (simulated) graph.
            target_stats: Statistics measured on the observed (target) graph.
        Returns:
            Boolean array of shape ``(num_stats,)`` with ``True`` for each statistic that should receive gradient updates from the first step.
        """

    @abstractmethod
    def step(self, step_num: int, current_stats: np.ndarray, target_stats: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, bool]:
        """Optionally updates the active mask based on the current optimisation state.

        Called once per optimisation step **before** the gradient update so that
        newly activated statistics can immediately influence the current step.

        Args:
            step_num: Current optimisation step counter.
            current_stats: Statistics sampled at this step.
            target_stats: Fixed observed (target) statistics.
            **kwargs: Additional information that concrete strategies may require.
        Returns:
            A tuple ``(active_mask, has_changed)`` where ``active_mask`` is the (possibly updated) boolean mask and ``has_changed`` is ``True`` if one or more indices were newly activated this step.
        """


class ErrorThresholdActivationStrategy(ActivationStrategy):
    """Activates statistics whose absolute error falls below an initial threshold."""

    def __init__(self, initial_h: int) -> None:
        if initial_h <= 0: raise ValueError(f"initial_h must be strictly positive, got {initial_h}.")

        self._initial_h = initial_h
        self.active_mask: np.ndarray = np.array([], dtype=bool)
        self.inclusion_threshold: float = 0.0

    def initialize(self, initial_stats: np.ndarray, target_stats: np.ndarray) -> np.ndarray:
        """Seeds the active mask with the ``initial_h`` lowest-error statistics."""
        num_stats = len(initial_stats)
        initial_errors = np.abs(initial_stats - target_stats)
        seed_indices = np.argsort(initial_errors)[: self._initial_h]

        self.active_mask = np.zeros(num_stats, dtype=bool)
        self.active_mask[seed_indices] = True
        self.inclusion_threshold = float(initial_errors[self.active_mask].max())

        logger.info(
            f"Curriculum learning initiated with active stats at {seed_indices.tolist()} "
            f"and inclusion threshold {self.inclusion_threshold:.4f}."
        )
        return self.active_mask.copy()

    def step(self, step_num: int, current_stats: np.ndarray, target_stats: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, bool]:
        """Promotes inactive statistics whose absolute error is within the threshold."""
        abs_errors = np.abs(current_stats - target_stats)
        newly_active = np.where((~self.active_mask) & (abs_errors <= self.inclusion_threshold))[0]

        if newly_active.size == 0:
            return self.active_mask, False

        self.active_mask[newly_active] = True
        logger.info(f"Step {step_num}: dynamically activated statistics at indices {newly_active.tolist()}.")
        return self.active_mask, True


class StaticActivationStrategy(ActivationStrategy):
    """Strategy that keeps a fixed set of statistics active from start to finish."""

    def __init__(self, active_mask: list[bool] | np.ndarray) -> None:
        self.active_mask = np.array(active_mask, dtype=bool)

    def initialize(self, initial_stats: np.ndarray, target_stats: np.ndarray) -> np.ndarray:
        """Sets the active mask, ensuring it matches the number of available statistics."""
        num_stats = len(initial_stats)
        if len(self.active_mask) != num_stats:
            logger.error(f"Static mask size ({len(self.active_mask)}) does not match number of statistics ({num_stats}).")
            raise ValueError("Static mask size does not match number of statistics.")

        active_indices = np.where(self.active_mask)[0].tolist()
        logger.info(f"Static activation strategy initialized. Active indices: {active_indices}.")
        return self.active_mask.copy()

    def step(self, step_num: int, current_stats: np.ndarray, target_stats: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, bool]:
        """Returns the static mask without modifications."""
        return self.active_mask, False


# -----------------------------------------------------------------------------
# Robbins-Monro Estimator
# -----------------------------------------------------------------------------

class RobbinsMonroEstimator:
    """Estimator using the Robbins-Monro algorithm for ERGMs.

    Guides the theta parameters towards the maximum likelihood solution using
    an adaptive strategy that periodically updates the estimated curvature
    (covariance matrix) of the parameter space.
    """

    def __init__(self, sampler: MetropolisHastingsSampler, manager: StatisticsManager) -> None:
        """Initializes the estimator.

        Args:
            sampler: The MCMC sampler initialized with a simulation graph.
            manager: The statistics manager currently used by the sampler.
        """
        self.sampler = sampler
        self.manager = manager

        self._global_covariance: np.ndarray = np.array([])
        self._inverse_covariance: np.ndarray = np.array([])
        self._invariant_mask: np.ndarray = np.array([], dtype=bool)
        self._modeled_mask: np.ndarray = np.array([], dtype=bool)
        self._active_indices: np.ndarray = np.array([], dtype=int)

    def fit(
        self,
        target_stats: np.ndarray,
        thinning: int,
        updates: int,
        learning_rate: float,
        lr_decay: float,
        clip_gradient_norm: float,
        covariance_update_interval: int,
        covariance_update_alpha: float,
        activation_strategy: ActivationStrategy | None = None,
        estimation_logger: EstimationLogger | None = None,
        early_stopping_tracker: EarlyStoppingTracker | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Executes the complete Robbins-Monro estimation algorithm.

        Args:
            target_stats: The observed statistics (sufficient + diagnostic) to fit against.
            thinning: Number of MCMC steps between collected samples.
            updates: Number of update steps for the optimization phase.
            learning_rate: Initial learning rate.
            lr_decay: Factor applied to the learning rate at each step (exponential decay).
            clip_gradient_norm: Maximum norm for gradient clipping.
            covariance_update_interval: Interval for covariance updates.
            covariance_update_alpha: EMA alpha for covariance updates.
            activation_strategy: Optional ActivationStrategy instance that controls which statistics receive gradient updates at each step.
            estimation_logger: Optional logger to record ongoing metrics.
            early_stopping_tracker: Optional EarlyStoppingTracker instance.
                If provided, the optimization loop will invoke it at every step
                and terminate early if the patience criterion is satisfied.
                Best-seen theta and graph state are restored upon stopping.
            show_progress: If True, displays progress bars during Phases 1 and 2.
        Returns:
            The final (or best-seen) theta parameters.
        """
        if updates <= 0: raise ValueError("Optimization updates must be strictly positive.")
        if learning_rate <= 0.0: raise ValueError("Learning rate must be strictly positive.")
        if clip_gradient_norm <= 0.0: raise ValueError("Clip gradient norm must be strictly positive.")
        if covariance_update_interval <= 1: raise ValueError("Covariance update interval must be > 1.")
        if not (0.0 < covariance_update_alpha <= 1.0): raise ValueError("Covariance update alpha must be in (0, 1].")
        if not (0.0 < lr_decay <= 1.0): raise ValueError("LR decay must be in (0, 1].")

        num_stats = self.manager.num_statistics
        initial_stats = self.manager.calculate_initial_values(self.sampler._graph_state)

        # Initialise modeled mask via strategy (or enable all statistics)
        if activation_strategy is not None:
            self._modeled_mask = activation_strategy.initialize(initial_stats, target_stats)
        else:
            self._modeled_mask = np.ones(num_stats, dtype=bool)
            logger.info("No activation strategy provided: all %d statistics are active.", num_stats)

        # Inject active mask into the sampler (Topological Blindness)
        self.sampler.modeled_mask = self._modeled_mask

        # Initialize early stopping tracker if present
        if early_stopping_tracker is not None:
            early_stopping_tracker.initialize(target_stats)

        # 1. Phase 1: Warm-up (Initial D0 curvature estimate)
        self._run_warmup_phase(covariance_update_interval, thinning, show_progress)

        # 2. Phase 2: Optimization Loop
        batch_mean = np.zeros(num_stats, dtype=np.float64)
        batch_m2 = np.zeros((num_stats, num_stats), dtype=np.float64)
        batch_count = 0

        context = logging_redirect_tqdm() if show_progress else nullcontext()
        current_lr = learning_rate
        with context:
            pbar = tqdm(range(1, updates + 1), desc="Optimization", unit="update", disable=not show_progress)
            for step in pbar:
                # A. Sampling Step
                res = self.sampler.sample(samples_count=1, burn_in=thinning - 1, thinning=1)
                current_stats = res.mean_statistics

                # B. Welford accumulation (global, over all statistics)
                batch_count += 1
                batch_mean, batch_m2 = welford_update(current_stats, batch_count, batch_mean, batch_m2)

                # C. Periodic Curvature Update (D0 Refresh)
                if step % covariance_update_interval == 0:
                    self._update_global_curvature(batch_m2, batch_count, covariance_update_alpha)
                    
                    # Reset batch accumulators
                    batch_mean = np.zeros(num_stats, dtype=np.float64)
                    batch_m2 = np.zeros((num_stats, num_stats), dtype=np.float64)
                    batch_count = 0

                # D. Dynamic activation via strategy
                current_errors = current_stats - target_stats
                if activation_strategy is not None:
                    self._modeled_mask, mask_changed = activation_strategy.step(step, current_stats, target_stats)
                    if mask_changed:
                        self.sampler.modeled_mask = self._modeled_mask
                        self._compute_inverse_and_mask(self._global_covariance)

                # E. Parameter Update (Theta Step)
                update, norm, clipped = self._compute_update_vector(current_errors, current_lr, clip_gradient_norm)
                self.sampler._thetas += update
                # Decay learning rate for the next step
                current_lr *= lr_decay
                
                # F. Early Stopping & Telemetry
                current_j = np.nan
                should_stop = False
                if early_stopping_tracker is not None:
                    should_stop, current_j = early_stopping_tracker.step(
                        current_errors=current_errors,
                        acc_rate=res.acceptance_rate,
                        current_thetas=self.sampler._thetas,
                        current_graph=self.sampler._graph_state.sync_igraph()
                    )
                    if not np.isnan(current_j):
                        pbar.set_postfix({"J": f"{current_j:.4f}"})
                    
                    if should_stop:
                        logger.info(f"Early stopping triggered at step %d (best J found at step %d).", step, early_stopping_tracker.best_step)
                        break

                if estimation_logger:
                    estimation_logger.log_optimization_step(
                        step, current_lr, self.sampler._thetas, norm, 
                        current_errors, clipped, res.acceptance_rate, self._modeled_mask.copy(), current_j
                    )

        # 3. Restoration
        self._restore_best_optimization_state(early_stopping_tracker)
        return self.sampler._thetas.copy()

    def _run_warmup_phase(self, interval, thinning, show_progress):
        """Runs the initial Phase 1 to estimate D0."""
        logger.info(f"Starting Phase 1 (Warm-up): {interval} samples.")
        res = self.sampler.sample(samples_count=interval, burn_in=thinning, thinning=thinning, show_progress=show_progress)
        self._global_covariance = res.covariance_matrix
        self._compute_inverse_and_mask(self._global_covariance)

    def _update_global_curvature(self, batch_m2, batch_count, alpha):
        """Updates the global D0 matrix using the latest batch and EMA."""
        d_batch = batch_m2 / (batch_count - 1)
        self._global_covariance = (1.0 - alpha) * self._global_covariance + alpha * d_batch
        self._compute_inverse_and_mask(self._global_covariance)

    def _restore_best_optimization_state(self, tracker: EarlyStoppingTracker | None):
        """Restores parameters and graph from the best seen state during optimization."""
        if tracker is not None and tracker.best_theta is not None:
            self.sampler._thetas = tracker.best_theta.copy()
            self.sampler._graph_state = GraphState(tracker.best_graph.copy())
            self.manager.calculate_initial_values(self.sampler._graph_state)

    def _compute_inverse_and_mask(self, covariance_matrix: np.ndarray) -> None:
        """Computes the inverse of the active sub-matrix and updates the invariant mask."""
        epsilon = 1e-8
        variances = np.diag(covariance_matrix)
        new_mask = variances < epsilon

        if not np.array_equal(self._invariant_mask, new_mask):
            invariant_indices = np.where(new_mask)[0]
            if len(invariant_indices) > 0:
                logger.warning(
                    f"Detected invariant statistics at indices {invariant_indices.tolist()}. "
                    "These parameters will be fixed during optimization."
                )
            self._invariant_mask = new_mask
        
        valid_mask = self._modeled_mask & ~self._invariant_mask
        self._active_indices = np.where(valid_mask)[0]

        # If all variables are invariant, inverse is empty
        if len(self._active_indices) == 0:
            self._inverse_covariance = np.array([[]])
            return

        active_cov = covariance_matrix[self._active_indices][:, self._active_indices]# + np.eye(len(self._active_indices)) * 0.05

        try:
            self._inverse_covariance = np.linalg.inv(active_cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix inversion failed. Falling back to pseudo-inverse.")
            self._inverse_covariance = np.linalg.pinv(active_cov)

    def _compute_update_vector(self, error_stats: np.ndarray, current_lr: float, clip_gradient_norm: float) -> tuple[np.ndarray, float, bool]:
        """Calculates the delta update for the parameters with gradient clipping.
        Formula: -a_n * D0^-1 * (u_sim - u_obs)
        """
        num_stats = len(error_stats)
        clipped = False
        
        if len(self._active_indices) == 0:
            return np.zeros(num_stats, dtype=np.float64), 0.0, clipped

        active_error = error_stats[self._active_indices]
        # active_error = np.sign(active_error) * (np.abs(active_error) ** 2)
        raw_update = -current_lr * (self._inverse_covariance @ active_error)

        norm = float(np.linalg.norm(raw_update))
        if norm > clip_gradient_norm:
            raw_update = raw_update * (clip_gradient_norm / norm)
            clipped = True
            norm = float(clip_gradient_norm)

        update = np.zeros(num_stats, dtype=np.float64)
        update[self._active_indices] = raw_update

        return update, norm, clipped

    def validate(self, final_samples: int, thinning: int, show_progress: bool = True) -> list[ig.Graph]:
        """Runs a final simulation phase to generate graph samples.

        Args:
            final_samples: Number of samples for the final evaluation phase.
            thinning: Thinning steps between samples.
            show_progress: If True, displays a progress bar during validation.
        Returns:
            A list of generated igraph.Graph objects.
        """
        if final_samples <= 0:
            raise ValueError("Validation samples must be strictly positive.")

        logger.info(f"Starting Validation Phase: {final_samples} samples.")

        res = self.sampler.sample(
            final_samples,
            burn_in=thinning,
            thinning=thinning,
            show_progress=show_progress,
            return_graphs=True
        )

        return res.graphs
