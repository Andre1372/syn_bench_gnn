import sys
from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from src.anndg.graph_state import GraphState, GraphChange
from src.graph_analysis import calculate_degree_assortativity


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

    degree_distribution = np.array([
        group.mean() if group.size > 0 else 0.0 
        for group in np.array_split(sorted_degrees, bins)
    ], dtype=float)

    degree_distribution = degree_distribution / degree_distribution.sum() 

    return degree_distribution


def _compute_annd_objective(actual_annd: np.ndarray, target_annd: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the weighted Log-Cosh loss between actual and target ANND profiles.

    The Log-Cosh objective function serves as a smooth, robust alternative to 
    Mean Squared Error (MSE).

    Formula:
        loss = Σ w_i * log(cosh(10 * (actual_i - target_i)))

    Numerical stability is ensured via the identity: 
        log(cosh(x)) = |x| - log(2) + log(1 + exp(-2|x|)).

    Args:
        actual_annd: Array of current ANND values.
        target_annd: Array of target ANND values.
        weights: Weighting coefficients, usually representing the degree distribution bin sizes.
    Returns:
        float: The total weighted objective value.
    """
    x = 10.0 * (actual_annd - target_annd)
    log_cosh = np.abs(x) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(x)))
    error = np.dot(weights, log_cosh)
    
    return float(error)


def _compute_diameter_objective(actual_diameter: float, target_diameter: float) -> float:
    """Calculates the relative error between actual and target diameter."""
    return abs(actual_diameter - target_diameter) / target_diameter


def _compute_eccentricity_objective(actual_ecc_moments: np.ndarray, target_ecc_moments: np.ndarray) -> float:
    """Calculates the structural error between actual and target eccentricity moments.
    
    The error is computed by applying an arcsinh transformation to both actual and target
    values to stabilize variance across different scales, followed by a power-law 
    penalty (exponent 1.5). The individual moment losses (mean, variance, skewness, 
    kurtosis) are aggregated using an L2 norm.

    Args:
        actual_ecc_moments: NumPy array containing the moments [mean, var, skew, kurt] of 
            the current graph's eccentricity distribution.
        target_ecc_moments: NumPy array containing the moments of the target distribution.
    Returns:
        A scalar penalty value representing the discrepancy in eccentricity moments.
    """
    k = len(target_ecc_moments)
    moment_losses = np.zeros(k)

    def _compute_metric_loss(actual_value, target_value):
        a = np.arcsinh(actual_value)
        b = np.arcsinh(target_value)
        loss = np.abs(a - b)**1.5

        return loss

    # Mean loss
    mean_actual, mean_target = actual_ecc_moments[0], target_ecc_moments[0]
    moment_losses[0] = _compute_metric_loss(mean_actual, mean_target)
    
    # Variance loss
    if k > 1:
        var_actual, var_target = actual_ecc_moments[1], target_ecc_moments[1]
        moment_losses[1] = _compute_metric_loss(var_actual, var_target)
    
    # Skewness loss (evaluated only when variance is stable)
    if k > 2 and var_actual > 1e-12:
        skew_actual, skew_target = actual_ecc_moments[2], target_ecc_moments[2]
        moment_losses[2] = _compute_metric_loss(skew_actual, skew_target)
        
    # Kurtosis loss
    if k > 3:
        kurt_actual, kurt_target = actual_ecc_moments[3], target_ecc_moments[3]
        moment_losses[3] = _compute_metric_loss(kurt_actual, kurt_target)
            
    penalty = float(np.mean(moment_losses ** 2.0) ** 0.5)
    
    return penalty


def _propose_double_edge_swap(graph_state: GraphState, rng: np.random.Generator) -> GraphChange | None:
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
    target_diameter: float | None = None,
    target_ecc_moments: np.ndarray | None = None,
    rng: np.random.Generator | None = None, 
    debug: bool = False
) -> tuple[GraphState, float]:
    """
    Optimizes a graph's ANND (Average Nearest Neighbor Degree) to match a target vector.

    Args:
        initial_graph: The initial graph structure.
        target_annd: The target mean degree for each degree in the original graph.
        rng: The random number generator instance.
        debug: Whether to print optimization progress and plot errors.
    Returns:
        The optimized `GraphState` exhibiting the best discovered ANND configuration and the final error.
    """
    rng = rng or np.random.default_rng()
    target_annd = np.asarray(target_annd, dtype=float)
    if debug:
        target_assortativity = calculate_degree_assortativity(initial_graph)
    bins = len(target_annd)

    # Initialize graph state
    graph_state = GraphState(initial_graph)
    initial_annd = graph_state.get_annd(bins=bins)
    initial_diameter = graph_state.exact_diameter if target_diameter is not None else None
    initial_ecc_moments = graph_state.ecc_moments if target_ecc_moments is not None else None
    
    # Compute binned degree distribution for weighting (matches the binning in get_annd)
    degree_distribution = _get_binned_degree_distribution(graph_state._degrees, bins=bins)

    if debug:
        print(f"{'Degree distribution:':<40} {degree_distribution}")
        print(f"{'Initial ANND:':<40} {initial_annd}")
        print(f"{'Target ANND:':<40} {target_annd}")

    # Optimization loop parameters
    max_steps = 10000
    patience = 500
    steps_without_improvement = 0
    temperature = 1.0
    cooling = 10**(-3/max_steps) #0.998

    # Track metrics
    current_error = _compute_annd_objective(initial_annd, target_annd, weights=degree_distribution)
    if target_diameter is not None:
        current_error = 0.9 * current_error + 0.1 * _compute_diameter_objective(initial_diameter, target_diameter)
    if target_ecc_moments is not None:
        current_error = 0.7 * current_error + 0.3 * _compute_eccentricity_objective(initial_ecc_moments, target_ecc_moments)
    
    errors_history = [current_error] if debug else []
    assortativity_errors_history = []
    
    if debug and target_assortativity is not None:
        initial_assortativity = calculate_degree_assortativity(graph_state.get_graph())
        assortativity_errors_history.append(abs(initial_assortativity - target_assortativity))
    if debug:
        print(f"{'Initial ANND error:':<40} {current_error:.6f}")

    best_state = {
        "error": current_error,
        "graph_state": graph_state.copy(),
        "temperature": temperature,
        "step": 0,
        }
    reset_allowed = True
    
    # Main loop
    for step in range(max_steps):
        change = _propose_double_edge_swap(graph_state, rng)
        steps_without_improvement += 1
        
        # State transitions without proposals skip evaluations
        if change is None: 
            if debug: 
                errors_history.append(current_error)
                assortativity_errors_history.append(assortativity_errors_history[-1])
            if steps_without_improvement >= patience:
                if debug: print(f"Early stopping at step {step + 1} - Best error: {best_state['error']:.6f} at step {best_state['step']}")
                break
            continue

        # Propose state change
        graph_state.apply_change(change)
        proposed_error = _compute_annd_objective(graph_state.get_annd(bins=bins), target_annd, weights=degree_distribution)
        if target_diameter is not None:
            proposed_error = 0.9 * proposed_error + 0.1 * _compute_diameter_objective(graph_state.exact_diameter, target_diameter)
        if target_ecc_moments is not None:
            proposed_error = 0.7 * proposed_error + 0.3 * _compute_eccentricity_objective(graph_state.ecc_moments, target_ecc_moments)

        if rng.random() < np.exp((best_state['error'] - proposed_error) / temperature):
            # accept
            current_error = proposed_error
            if current_error < best_state['error']:
                best_state['error'] = current_error
                best_state['graph_state'] = graph_state.copy()
                best_state['temperature'] = temperature
                best_state['step'] = step
                steps_without_improvement = 0
                reset_allowed = True
        else:
            # reject
            graph_state.revert_change(change)
        
        # Temperature decay
        temperature *= cooling
        
        if debug:
            errors_history.append(current_error)
            current_assortativity = calculate_degree_assortativity(graph_state.get_graph())
            assortativity_errors_history.append(abs(current_assortativity - target_assortativity))
                
        if steps_without_improvement >= patience:
            if debug: print(f"Early stopping at step {step + 1} - Best error: {best_state['error']:.6f} at step {best_state['step']}")
            break

        if steps_without_improvement >= patience * 0.75 and reset_allowed:
            graph_state = best_state['graph_state'].copy()
            temperature = min(1.0, best_state['temperature'] * 1.15)
            current_error = best_state['error']
            reset_allowed = False
        
    if debug:
        print(f"{'Final ANND error:':<40} {best_state['error']:.6f}")
    
        # Plot optimization trajectory
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(errors_history, color='#2563eb', linewidth=1.5, label='Objective Function')
        ax1.set_xlabel("MCMC Step", fontsize=12)
        ax1.set_ylabel("Total Error", fontsize=12, color='#2563eb')
        ax1.tick_params(axis='y', labelcolor='#2563eb')
        ax1.grid(True, linestyle='--', alpha=0.7)

        if assortativity_errors_history:
            ax2 = ax1.twinx()
            ax2.plot(assortativity_errors_history, color='#dc2626', linewidth=1.5, label='Assortativity Error')
            ax2.set_ylabel("Assortativity Absolute Error", fontsize=12, color='#dc2626')
            ax2.tick_params(axis='y', labelcolor='#dc2626')
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        plt.title("ANNDG Optimization Progress", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return best_state['graph_state'], best_state['error']
