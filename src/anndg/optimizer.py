import sys
from pathlib import Path

import igraph as ig
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.anndg.graph_state import GraphState, GraphChange
from src.graph_analysis import calculate_degree_assortativity
from src.data_utils import igraph_to_networkx


def _compute_annd_objective(actual_annd: np.ndarray, target_annd: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the weighted loss between actual and target ANND profiles.

    Args:
        actual_annd: Array of current ANND values.
        target_annd: Array of target ANND values.
        weights: Mantained for compatibility with other methods.
    Returns:
        float: The total weighted objective value.
    """
    loss = np.log1p(40 * np.abs(actual_annd - target_annd)**1.5)
    error = np.dot(weights, loss)
    
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
        moment_losses[2] = 0.5*_compute_metric_loss(skew_actual, skew_target)
        
    # Kurtosis loss
    if k > 3:
        kurt_actual, kurt_target = actual_ecc_moments[3], target_ecc_moments[3]
        moment_losses[3] = 0.5*_compute_metric_loss(kurt_actual, kurt_target)
            
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


def _propose_intelligent_double_edge_swap(
    graph_state: GraphState, 
    current_annd: np.ndarray,
    target_annd: np.ndarray,
    rng: np.random.Generator
) -> GraphChange | None:
    """Propose an intelligent degree-preserving double edge swap."""
    if graph_state.num_edges < 2:
        return None

    # --- PART 1: Sample the first edge (u, v) ---
    # Prioritize the bin with the highest error
    bins_errors = np.abs(current_annd - target_annd)
    sorted_bins = np.argsort(bins_errors)[::-1]
    for b in sorted_bins:
        # If it is the last take it for sure
        if b == sorted_bins[-1]:
            bin_idx = b
            break
        # Otherwise take it with probability 0.75
        if rng.random() < 0.75:
            bin_idx = b
            break
    
    u = graph_state.get_random_node_from_bin(bin_idx, rng, current_value=current_annd[bin_idx], target_value=target_annd[bin_idx])
    if u is not None:
        # Sample a neighbor v to form the edge (u, v)
        neighbors_u = list(graph_state.neighbors(u))
        v = int(rng.choice(neighbors_u)) if neighbors_u else None

    # Fallback: if bin sampling failed, pick a completely random edge
    if u is None or v is None:
        u, v = graph_state.get_random_edge(rng)
        
    # --- PART 2: Sample the second edge (x, y) ---
    x, y = graph_state.get_random_edge(rng)

    # --- PART 3: Validity Checks and Proposal Building ---
    # Check 1: All nodes must be distinct to prevent self-loops and multi-edges
    if len({u, v, x, y}) != 4:
        return None

    # Randomly choose between the two possible swap configurations
    if rng.integers(0, 2) == 0:
        proposed_edges = [(u, y), (x, v)]
    else:
        proposed_edges = [(u, x), (v, y)]

    # Check 2: The new edges must not already exist in the graph
    for a, b in proposed_edges:
        if graph_state.has_edge(a, b):
            return None

    return GraphChange(
        edges_to_add=proposed_edges,
        edges_to_remove=[(u, v), (x, y)]
    )


def optimizer(
    initial_graph: ig.Graph, 
    target_annd: np.ndarray, 
    target_diameter: float | None = None,
    target_ecc_moments: np.ndarray | None = None,
    rng: np.random.Generator | None = None, 
    debug: bool = False
) -> tuple[nx.Graph, dict]:
    """
    Optimizes a graph's ANND (Average Nearest Neighbor Degree) to match a target vector.

    Args:
        initial_graph: The initial graph structure.
        target_annd: The target mean degree for each degree in the original graph.
        rng: The random number generator instance.
        debug: Whether to print optimization progress and plot errors.
    Returns:
        A tuple (graph, info) where graph is the optimized `nx.Graph` and info 
        is a dictionary containing 'best_error', 'best_annd', and 'best_ecc_moments'.
    """
    rng = rng or np.random.default_rng()
    target_annd = np.asarray(target_annd, dtype=float)
    if debug:
        target_assortativity = -0.2179
    bins = len(target_annd)
    weights = np.ones(bins)/bins

    # Initialize graph state
    graph_state = GraphState(initial_graph, bins=bins)
    initial_annd = graph_state.get_annd()
    initial_diameter = graph_state.exact_diameter if target_diameter is not None else None
    initial_ecc_moments = graph_state.ecc_moments if target_ecc_moments is not None else None
    
    if debug:
        print(f"{'Initial ANND:':<40} {initial_annd}")
        print(f"{'Target ANND:':<40} {target_annd}")
        print(f"{'Initial assortativity:':<40} {calculate_degree_assortativity(initial_graph)}")
        print(f"{'Target assortativity:':<40} {target_assortativity}")

    # Optimization loop parameters
    max_steps = 100 * graph_state.num_edges
    patience = max(500, int(max_steps * 0.25))
    steps_without_improvement = 0
    temperature = 1.0
    cooling = 2**(-3/max_steps)
    good_enough_threshold = 1e-4

    # Track metrics
    current_error = _compute_annd_objective(initial_annd, target_annd, weights=weights)
    if target_diameter is not None:
        current_error = 0.9 * current_error + 0.1 * _compute_diameter_objective(initial_diameter, target_diameter)
    if target_ecc_moments is not None:
        current_error = 0.9 * current_error + 0.1 * _compute_eccentricity_objective(initial_ecc_moments, target_ecc_moments)
    
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
        "step": 0,
        }

    if best_state['error'] < good_enough_threshold:
        if debug: print(f"Early stopping at step 0 - Best error: {best_state['error']:.6f} at step 0")
        best_gs = best_state['graph_state']
        info = {
            "best_error": best_state['error'],
            "best_annd": best_gs.get_annd(),
        }
        if target_diameter is not None:
            info["best_diameter"] = best_gs.exact_diameter
            
        if target_ecc_moments is not None:
            info["best_ecc_moments"] = best_gs.ecc_moments
        
        return igraph_to_networkx(best_gs.get_graph()), info

    current_annd = initial_annd
    # Main loop
    for step in range(1, max_steps):

        # Local loop to find a valid proposal
        change = None
        failed_proposals = 0
        while change is None:
            change = _propose_intelligent_double_edge_swap(graph_state, current_annd, target_annd, rng)
            if change is None:
                failed_proposals += 1
                if failed_proposals >= patience:
                    if debug:
                        print(f"Aborting: Failed to find a valid proposal after {patience} attempts at step {step}.")
                    break
        
        if change is None:
            break

        # Calculate proposal error
        graph_state.apply_change(change)
        proposed_annd = graph_state.get_annd()
        proposed_error = _compute_annd_objective(proposed_annd, target_annd, weights=weights)
        if target_diameter is not None:
            proposed_error = 0.9 * proposed_error + 0.1 * _compute_diameter_objective(graph_state.exact_diameter, target_diameter)
        if target_ecc_moments is not None:
            proposed_error = 0.9 * proposed_error + 0.1 * _compute_eccentricity_objective(graph_state.ecc_moments, target_ecc_moments)

        if proposed_error < best_state['error']:
            # accept sure improvements
            current_error = proposed_error
            best_state['error'] = current_error
            best_state['graph_state'] = graph_state.copy()
            best_state['step'] = step
            steps_without_improvement = 0
            current_annd = proposed_annd
            if debug:
                print(f"Accepted improvement at step {step}: error = {current_error:.4f}, annd_vector = {current_annd}")
        elif rng.random() < np.exp((best_state['error'] - proposed_error) / temperature):
            # accept non improving steps with probability exp(-delta/T)
            current_error = proposed_error
            steps_without_improvement += 1
            current_annd = proposed_annd
        else:
            # reject
            graph_state.revert_change(change)
            steps_without_improvement += 1
                    
        # Temperature decay
        temperature *= cooling
        
        if debug:
            errors_history.append(current_error)
            current_assortativity = calculate_degree_assortativity(graph_state.get_graph())
            assortativity_errors_history.append(abs(current_assortativity - target_assortativity))
                
        if steps_without_improvement >= patience:
            if debug: print(f"Early stopping at step {step} - Best error: {best_state['error']:.6f} at step {best_state['step']}")
            break

        if best_state['error'] < good_enough_threshold:
            if debug: print(f"Early stopping at step {step} - Best error: {best_state['error']:.6f} at step {best_state['step']}")
            break
        
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

    best_gs = best_state['graph_state']
    info = {
        "best_error": best_state['error'],
        "best_annd": best_gs.get_annd(),
    }
    if target_diameter is not None:
        info["best_diameter"] = best_gs.exact_diameter
    
    if target_ecc_moments is not None:
        info["best_ecc_moments"] = best_gs.ecc_moments

    return igraph_to_networkx(best_gs.get_graph()), info
