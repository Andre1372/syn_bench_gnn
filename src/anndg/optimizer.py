import sys
from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from src.anndg.graph_state import GraphState, GraphChange


def _adapt_target_dimension(actual_annd: np.ndarray, target_annd: np.ndarray) -> np.ndarray:
    """
    Adjusts the target Average Nearest Neighbor Degree (ANND) vector to align with the current 
    graph's degree range and fills missing observations.

    This function handles two main alignment issues: Missing Degrees, Dimension Mismatch.

    Phases:
    - PHASE 1 (Interpolation): Uses linear interpolation to fill internal gaps where target_annd < 1.
    - PHASE 2 (Extrapolation): Applies a quadratic decay both backward and forward to extend the target vector.

    Args:
        actual_annd: ANND vector of the graph being optimized. Used to determine the required range.
        target_annd: The reference ANND vector to match.
    Returns:
        np.ndarray: A target ANND vector of length max(len(actual), len(target)).
    """
    len_actual = len(actual_annd)
    len_target = len(target_annd)

    # PHASE 1: Filling
    valid_mask = (target_annd >= 1)
    valid_indices = np.where(valid_mask)[0]
    filled_target = np.interp(np.arange(len_target), valid_indices, target_annd[valid_indices])
    
    # PHASE 2: Extension
    # Backward
    first_valid_idx = valid_indices[0]
    if first_valid_idx > 0:
        first_val = target_annd[first_valid_idx]
        steps_back = first_valid_idx - np.arange(first_valid_idx)
        left_decay = first_val * (1 - 0.05 * (steps_back**2))
        filled_target[:first_valid_idx] = np.maximum(left_decay, 1)

    # Forward
    num_to_add = len_actual - len_target
    if num_to_add > 0:
        last_val = filled_target[-1]
        steps_fwd = np.arange(1, num_to_add + 1)
        right_decay = last_val * (1 - 0.05 * (steps_fwd**2))
        final_target = np.concatenate((filled_target, np.maximum(right_decay, 1)))
    else:
        final_target = filled_target[:len_actual]
    
    return final_target


def _get_degree_distribution(graph: ig.Graph) -> np.ndarray:
    """
    Calculates the normalized degree distribution of the graph using igraph.
    
    Args:
        graph: The igraph.Graph instance.
    Returns:
        np.ndarray: An array where entry i is the fraction of nodes with degree i.
    """
    n = graph.vcount()
    if n == 0: 
        return np.array([], dtype=float)
    
    degrees = graph.degree()
    counts = np.bincount(degrees)
    return counts / n


def _compute_objective(actual_annd: np.ndarray, target_annd: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted L2 distance objective function between the actual and target ANND."""
    diff = actual_annd - target_annd
    error = np.sqrt(np.dot(weights, diff * diff))
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

    # Initialize graph state
    graph_state = GraphState(initial_graph)
    initial_annd = graph_state.get_annd()
    
    # Adapt target vectors and precompute fixed factors
    target_annd_adapted = _adapt_target_dimension(initial_annd, target_annd)
    degree_distribution = _get_degree_distribution(initial_graph)[1:]  # exclude degree = 0

    if debug:
        print(f"{'Degree distribution:':<40} {degree_distribution}")
        print(f"{'Initial ANND:':<40} {initial_annd}")
        print(f"{'Target ANND:':<40} {target_annd}")
        print(f"{'Target ANND (adapted):':<40} {target_annd_adapted}")

    # Track metrics
    current_error = _compute_objective(initial_annd, target_annd_adapted, weights=degree_distribution)
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
        proposed_error = _compute_objective(graph_state.get_annd(), target_annd_adapted, weights=degree_distribution)
        
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
