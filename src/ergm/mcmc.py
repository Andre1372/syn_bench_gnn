"""Module for orchestrating the MCMC simulation using Metropolis-Hastings."""

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass

import igraph as ig
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.ergm.graph_state import GraphChange, GraphState
from src.ergm.statistics import StatisticsManager
from src.ergm.transformations import TransformationStrategy


logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Encapsulates the results of the MCMC simulation."""
    mean_statistics: np.ndarray
    covariance_matrix: np.ndarray
    acceptance_rate: float
    graphs: list[ig.Graph] | None = None


class MetropolisHastingsSampler:
    """Core MCMC sampler using the Metropolis-Hastings algorithm.

    Orchestrates the interactions between the graph state, the statistics,
    and the transformation strategy, efficiently aggregating statistics
    using Welford's online algorithm.
    """

    def __init__(self, graph_state: GraphState, strategy: TransformationStrategy, rng: np.random.Generator, manager: StatisticsManager, initial_thetas: np.ndarray) -> None:
        """Initializes the sampler.

        Args:
            graph_state: The initial graph state.
            strategy: The proposal strategy for graph transitions.
            rng: The random number generator for reproducibility.
            manager: The StatisticsManager instance handling sufficient statistics.
            initial_thetas: An array or list of initial parameter values.
        Raises:
            ValueError: If the number of statistics does not match the length of initial_thetas.
        """
        if manager.num_statistics != len(initial_thetas):
            raise ValueError(
                f"Mismatch between number of statistics ({manager.num_statistics}) "
                f"and number of initial thetas ({len(initial_thetas)})."
            )
        
        self._graph_state: GraphState = graph_state
        self._strategy: TransformationStrategy = strategy
        self._rng: np.random.Generator = rng
        self._manager: StatisticsManager = manager
        self._num_statistics: int = self._manager.num_statistics
        self._last_deltas: np.ndarray = np.zeros(self._num_statistics, dtype=np.float64)
        self._thetas: np.ndarray = np.array(initial_thetas, dtype=np.float64)
        self.modeled_mask: np.ndarray | None = None

    def _log_acceptance_ratio(self, change: GraphChange) -> float:
        """Calculates the log-acceptance ratio for a proposed change.

        Args:
            change: The proposed graph change.
        Returns:
            The calculated log(alpha) = theta^T * (U_proposed - U_current).
        """
        self._last_deltas = self._manager.calculate_deltas(self._graph_state, change)
        
        if self.modeled_mask is not None:
            # Calculate dot product ONLY for active statistics (topological blindness)
            log_ratio = float(np.dot(self._last_deltas[self.modeled_mask], self._thetas[self.modeled_mask]))
        else:
            # Fallback to standard behavior (all statistics)
            log_ratio = float(np.dot(self._last_deltas, self._thetas))

        return log_ratio
    
    def _step(self, temperature: float) -> bool:
        """Executes a single step of the Metropolis-Hastings algorithm.

        Args:
            temperature: A scaling factor for the log-acceptance ratio.
        Returns:
            True if the proposed change was accepted, False otherwise.
        """
        # 1. Proposal
        change = self._strategy.propose_change(self._graph_state)

        # 2. Invalid proposal handling
        if change is None:
            return False

        # 3. Log acceptance ratio calculation
        log_alpha = self._log_acceptance_ratio(change) * temperature

        # 4. Acceptance decision
        if log_alpha >= 0.0:
            accepted = True
        else:
            u = self._rng.random()
            # Handle extremely rare edge case where u is exactly 0.0
            log_u = -float('inf') if u == 0.0 else math.log(u)
            accepted = log_u < log_alpha

        # 5. Action
        if accepted:
            self._graph_state.apply_change(change)
            self._manager.update_statistics(self._last_deltas)
            return True

        return False

    def sample(self, samples_count: int, burn_in: int, thinning: int = 1, show_progress: bool = False, return_graphs: bool = False) -> SamplingResult:
        """Executes the simulation and returns aggregated statistics.

        Args:
            samples_count: The number of valid samples to collect.
            burn_in: The number of iterations to discard before sampling.
            thinning: The interval between collected samples to reduce autocorrelation.
            show_progress: If True, displays a progress bar during sampling.
            return_graphs: If True, also returns the list of generated graphs.
        Returns:
            A SamplingResult object containing the mean, covariance, and metrics.
        """
        # 1. Welford initialization
        stats_mean = np.zeros(self._num_statistics, dtype=np.float64)
        stats_m2 = np.zeros((self._num_statistics, self._num_statistics), dtype=np.float64)
        
        graphs: list[ig.Graph] = []
        total_proposals = 0
        accepted_proposals = 0

        # 2. Burn-in phase
        total_proposals += burn_in
        for _ in range(burn_in):
            if self._step(temperature=0.75): accepted_proposals += 1

        # 3. Sampling phase
        if samples_count == 1 and not show_progress: # FAST PATH
            total_proposals += thinning
            for _ in range(thinning):
                if self._step(temperature=0.75):
                    accepted_proposals += 1
            
            stats_mean = self._manager.current_values.copy()
            stats_cov = np.zeros((self._num_statistics, self._num_statistics), dtype=np.float64)
            collected_samples = 1
            
            if return_graphs:
                graphs.append(self._graph_state.sync_igraph().copy())
        else:
            collected_samples = 0
            current_step = 0

            context = logging_redirect_tqdm() if show_progress else nullcontext()
            with context:
                pbar = tqdm(total=samples_count, desc="MCMC Sampling", unit="sample", dynamic_ncols=True, disable=not show_progress)
                with pbar:
                    while collected_samples < samples_count:
                        total_proposals += 1
                        current_step += 1
                        
                        # Do a step
                        if self._step(temperature=0.5): accepted_proposals += 1

                        # Collect sample every thinning steps
                        if current_step % thinning == 0:
                            collected_samples += 1

                            stats = self._manager.current_values
                            
                            # Welford's online algorithm for vectors
                            stats_mean, stats_m2 = welford_update(stats, collected_samples, stats_mean, stats_m2)
                            
                            pbar.update(1)

                            if return_graphs:
                                graphs.append(self._graph_state.sync_igraph().copy())

            stats_cov = stats_m2 / (collected_samples - 1)

        acceptance_rate = accepted_proposals / total_proposals if total_proposals > 0 else 0.0

        return SamplingResult(
            mean_statistics=stats_mean,
            covariance_matrix=stats_cov,
            acceptance_rate=acceptance_rate,
            graphs=graphs
        )


def welford_update(value: float | np.ndarray, count: int, mean: float | np.ndarray, m2: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Performs a single step of Welford's online algorithm for mean and variance.

    new_mean = old_mean + (value - old_mean) / count
    new_m2 = old_m2 + (value - old_mean) * (value - new_mean)

    Args:
        value: The new observation (scalar or vector).
        count: Total number of observations including this one.
        mean: Current running mean.
        m2: Current M2 accumulator (sum of squares of differences from mean or covariance).
    Returns:
        A tuple of (updated_mean, updated_m2).
    """
    delta = value - mean
    new_mean = mean + delta / count
    delta2 = value - new_mean

    if isinstance(value, np.ndarray) and value.ndim > 0:
        new_m2 = m2 + np.outer(delta, delta2)
    else:
        new_m2 = m2 + delta * delta2

    return new_mean, new_m2