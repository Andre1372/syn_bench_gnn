import sys
from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from src.anndg.graph_state import GraphState, GraphChange


def _get_binned_degree_distribution(degrees: np.ndarray, bins: int) -> np.ndarray:
    """
    Bins the degree distribution into percentile groups and returns the mean degree per bin.

    Args:
        degrees: Array of node degrees.
        bins: Number of percentile bins.
    Returns:
        np.ndarray: Array of length `bins` containing the mean degree per bin.
    """
    if len(degrees) == 0:
        return np.zeros(bins, dtype=float)
    
    # Sort degrees and partition into percentile bins
    sorted_degrees = np.sort(degrees)
    return np.array([
        group.mean() if group.size > 0 else 0.0 
        for group in np.array_split(sorted_degrees, bins)
    ], dtype=float)


def _compute_objective(actual_annd: np.ndarray, target_annd: np.ndarray, weights: np.ndarray) -> float:
    """
    Computes a weighted Log-Cosh objective function between actual and target ANND.
    """
    # [OBSOLETE] Weighted L2 implementation:
    # diff = actual_annd - target_annd
    # error = np.sqrt(np.dot(weights, diff * diff))
    # return float(error)

    # # Advanced Log-Cosh implementation: sum(w_i * log(cosh(10 * delta_i)))
    # # We use a numerically stable identity: log(cosh(x)) = |x| - log(2) + log(1 + exp(-2|x|))
    x = 10.0 * (actual_annd - target_annd)
    log_cosh = np.abs(x) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(x)))
    error = np.dot(weights, log_cosh)
    
    return float(error)


def _propose_change(graph_state: GraphState, rng: np.random.Generator) -> GraphChange | None:
    """Propose a random degree-preserving double edge swap."""
    if graph_state.num_edges < 2:
        return None
        
    e1 = graph_state.get_random_edge(rng)
    e2 = graph_state.get_random_edge(rng)
    
    u, v = e1
    x, y = e2

    # Validity check (1): no shared nodes among sampled edges (avoids creating self-loops)
    if len({u, v, x, y}) != 4:
        return None

    # Coin flip for symmetric proposals (Cross vs Parallel)
    if rng.integers(0, 2) == 0:
        proposed_edges = [(u, y), (x, v)]
    else:
        proposed_edges = [(u, x), (v, y)]

    # Validity check (2): proposed new edges do not already exist (preserves simple graph property)
    for a, b in proposed_edges:
        if graph_state.has_edge(a, b):
            return None

    return GraphChange(
        edges_to_add=proposed_edges,
        edges_to_remove=[e1, e2]
    )


def optimizer(
    initial_graph: ig.Graph, 
    target_annd: np.ndarray, 
    rng: np.random.Generator | None = None, 
    debug: bool = False
) -> GraphState:
    """
    Optimizes a graph's ANND (Average Nearest Neighbor Degree) to match a target vector.

    Args:
        initial_graph: The initial graph structure.
        target_annd: The target mean degree for each degree in the original graph.
        rng: The random number generator instance.
        debug: Whether to print optimization progress and plot errors.
    Returns:
        The optimized `GraphState` exhibiting the best discovered ANND configuration.
    """
    rng = rng or np.random.default_rng()
    target_annd = np.asarray(target_annd, dtype=float)
    bins = len(target_annd)

    # Initialize graph state
    graph_state = GraphState(initial_graph)
    initial_annd = graph_state.get_annd(bins=bins)
    
    # Compute binned degree distribution for weighting (matches the binning in get_annd)
    degree_distribution = _get_binned_degree_distribution(graph_state._degrees, bins=bins)

    if debug:
        print(f"{'Degree distribution:':<40} {degree_distribution}")
        print(f"{'Initial ANND:':<40} {initial_annd}")
        print(f"{'Target ANND:':<40} {target_annd}")

    # Track metrics
    current_error = _compute_objective(initial_annd, target_annd, weights=degree_distribution)
    best_error = current_error
    best_state = graph_state.copy()
    
    errors_history = [current_error] if debug else []

    if debug:
        print(f"{'Initial ANND error:':<40} {current_error:.6f}")

    # Optimization loop parameters
    max_steps = 10000
    patience = 500
    steps_without_improvement = 0
    temperature = 1.0
    cooling = 10**(-3/max_steps) #0.998
    
    # Main loop
    for step in range(max_steps):
        change = _propose_change(graph_state, rng)
        steps_without_improvement += 1
        
        # State transitions without proposals skip evaluations
        if change is None: 
            if debug: errors_history.append(current_error)
            if steps_without_improvement >= patience:
                if debug: print(f"Early stopping at step {step + 1} - Best error: {best_error:.6f}")
                break
            continue

        # Propose state change
        graph_state.apply_change(change)
        proposed_error = _compute_objective(graph_state.get_annd(bins=bins), target_annd, weights=degree_distribution)
        
        if rng.random() < np.exp((best_error - proposed_error) / temperature):
            # accept
            current_error = proposed_error
            if current_error < best_error:
                best_error = current_error
                best_state = graph_state.copy()
                steps_without_improvement = 0
        else:
            # reject
            graph_state.revert_change(change)
        
        # Temperature decay
        temperature *= cooling
        
        if debug:
            errors_history.append(current_error)
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{max_steps} - Best error: {best_error:.6f}")
                
        if steps_without_improvement >= patience:
            if debug: print(f"Early stopping at step {step + 1} - Best error: {best_error:.6f}")
            break
        
    if debug:
        print(f"{'Final ANND error:':<40} {best_error:.6f}")
    
        # Plot optimization trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(errors_history, color='#2563eb', linewidth=1.5)
        plt.title("ANNDG Optimization Progress", fontsize=14, fontweight='bold')
        plt.xlabel("MCMC Step", fontsize=12)
        plt.ylabel("ANND Error", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    return best_state
