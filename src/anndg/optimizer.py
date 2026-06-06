from __future__ import annotations

import igraph as ig
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.anndg.graph_state import GraphState, GraphChange
from src.graph_analysis import calculate_degree_assortativity
from src.data_utils import igraph_to_networkx


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

_ANND_LOSS_SCALE: float = 40.0
_ANND_LOSS_EXPONENT: float = 1.5

_ECC_LOSS_SCALE: float = 40.0
_ECC_LOSS_EXPONENT: float = 1.5

# Relative weight assigned to ANND vs eccentricity objective
_ANND_WEIGHT: float = 0.9
_ECC_WEIGHT: float = 0.1

# Probability of selecting the highest-error bin when proposing an intelligent swap
_BIN_SELECTION_PROB: float = 0.75

# SA schedule
_PATIENCE_FRACTION: float = 0.25            # patience = max_steps * _PATIENCE_FRACTION
_MIN_PATIENCE: int = 500
_STEPS_PER_EDGE: int = 100                  # max_steps = _STEPS_PER_EDGE * graph_state.num_edges
_INITIAL_TEMPERATURE: float = 1.0           # T_0
_COOLING_BITS: int = 3                      # cooling = 2^(-_COOLING_BITS / max_steps)

_GOOD_ENOUGH_THRESHOLD: float = 1e-4

# For single graph generation debugging in notebooks/single_generation_analysis.ipynb
_DEBUG_TARGET_ASSORTATIVITY: float = -0.2179


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _compute_annd_errors(actual_annd: np.ndarray, target_annd: np.ndarray) -> np.ndarray:
    """Compute the per-bin ANND loss between the actual and target profiles.

    Args:
        actual_annd: Current per-bin normalized ANND values.
        target_annd: Target per-bin normalized ANND values.
    Returns:
        Array of per-bin loss values, shape ``(bins,)``.
    """
    return np.log1p(_ANND_LOSS_SCALE * np.abs(actual_annd - target_annd) ** _ANND_LOSS_EXPONENT)


def _compute_eccentricity_errors(actual_ecc: np.ndarray, target_ecc: np.ndarray) -> np.ndarray:
    """Compute the per-bin eccentricity loss between the actual and target profiles.

    Args:
        actual_ecc: Current per-bin normalized eccentricity values.
        target_ecc: Target per-bin normalized eccentricity values.
    Returns:
        Array of per-bin loss values, shape ``(bins,)``.
    """
    return np.log1p(_ECC_LOSS_SCALE * np.abs(actual_ecc - target_ecc) ** _ECC_LOSS_EXPONENT)


def _combined_error(annd_errors: np.ndarray, ecc_errors: np.ndarray | None) -> float:
    """Aggregate per-bin ANND and (optional) eccentricity errors into a scalar."""
    annd_mean = np.mean(annd_errors)
    if ecc_errors is None:
        return float(annd_mean)
    
    ecc_mean = np.mean(ecc_errors)
    return float(_ANND_WEIGHT * annd_mean + _ECC_WEIGHT * ecc_mean)


# ---------------------------------------------------------------------------
# Proposal generators
# ---------------------------------------------------------------------------

def _propose_double_edge_swap(graph_state: GraphState, rng: np.random.Generator) -> GraphChange | None:
    """Propose a uniformly random, degree-preserving double-edge swap.

    Samples two edges independently and proposes replacing them with two new
    edges that reconnect the same four endpoints (randomly choosing between
    the two possible pairings).  Returns ``None`` when fewer than two edges
    exist or the chosen endpoints do not form a valid proposal (shared nodes
    or proposed edges already present).
    """
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
    proposed_edges = [(u, y), (x, v)] if rng.integers(0, 2) == 0 else [(u, x), (v, y)]

    # Validity check (2): proposed new edges do not already exist (preserves simple graph property)
    if any(graph_state.has_edge(a, b) for a, b in proposed_edges):
        return None

    return GraphChange(edges_to_add=proposed_edges, edges_to_remove=[e1, e2])


def _propose_intelligent_double_edge_swap(
    graph_state: GraphState,
    current_annd: np.ndarray,
    target_annd: np.ndarray,
    knn_normalized: np.ndarray,
    rng: np.random.Generator,
    current_ecc: np.ndarray | None = None,
    target_ecc: np.ndarray | None = None,
) -> GraphChange | None:
    """Propose a degree-preserving double-edge swap biased toward high-error bins.

    **Part 1 – first edge (u, v):** the bin with the highest combined error
    (ANND + optional eccentricity) is visited first.  With probability
    ``_BIN_SELECTION_PROB`` a node *u* is sampled from that bin using
    KNN-based filtering (see ``GraphState.get_random_node_from_bin``); if
    sampling fails the next-highest-error bin is tried.  The last bin is
    always tried unconditionally.  If every bin fails, a fully random edge is
    used as fallback.

    **Part 2 – second edge (x, y):** sampled uniformly at random.

    **Part 3 – validity:** all four endpoints must be distinct and the two
    proposed new edges must not already exist in the graph.

    Returns ``None`` when a valid proposal cannot be formed.
    """
    if graph_state.num_edges < 2:
        return None

    # --- PART 1: Sample the first edge (u, v) ---
    annd_bin_errors = _compute_annd_errors(current_annd, target_annd)
    if current_ecc is not None and target_ecc is not None:
        ecc_bin_errors = _compute_eccentricity_errors(current_ecc, target_ecc)
        bin_errors = _ANND_WEIGHT * annd_bin_errors + _ECC_WEIGHT * ecc_bin_errors
    else:
        bin_errors = annd_bin_errors

    sorted_bins = np.argsort(bin_errors)[::-1]

    u = v = None
    for b in sorted_bins:
        is_last = b == sorted_bins[-1]
        if is_last or rng.random() < _BIN_SELECTION_PROB:
            u = graph_state.get_random_node_from_bin(b, rng, current_value=current_annd[b], target_value=target_annd[b], knn_normalized=knn_normalized)
            if u is not None:
                neighbors = list(graph_state.neighbors(u))
                if neighbors:
                    v = int(rng.choice(neighbors))
            break

    # Fallback: pick a fully random edge
    if u is None or v is None:
        u, v = graph_state.get_random_edge(rng)

    # --- PART 2: Sample the second edge (x, y) ---
    x, y = graph_state.get_random_edge(rng)

    # --- PART 3: Validity Checks and Proposal Building ---
    # Check 1: All nodes must be distinct to prevent self-loops and multi-edges
    if len({u, v, x, y}) != 4:
        return None

    # Coin flip for symmetric proposals (Cross vs Parallel)
    proposed_edges = [(u, y), (x, v)] if rng.integers(0, 2) == 0 else [(u, x), (v, y)]

    # Check 2: The new edges must not already exist in the graph
    if any(graph_state.has_edge(a, b) for a, b in proposed_edges):
        return None

    return GraphChange(edges_to_add=proposed_edges, edges_to_remove=[(u, v), (x, y)])


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimizer(
    initial_graph: ig.Graph,
    target_annd: np.ndarray,
    target_eccentricity: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    debug: bool = False,
) -> tuple[nx.Graph, dict]:
    """Optimize a graph's ANND profile via simulated-annealing double-edge swaps.

    At each step an intelligent double-edge swap is proposed (biased toward
    high-error bins) and accepted or rejected with the Metropolis criterion:

    - Moves that strictly improve the *global* best error are always accepted
      and update the best-state snapshot.
    - Non-improving moves are accepted with probability
      ``exp((best_error - proposed_error) / temperature)``.
    - Temperature decays geometrically by ``cooling`` after every step.

    The loop terminates early when:
    - No valid proposal can be found after ``patience`` consecutive attempts.
    - ``steps_without_improvement`` reaches ``patience``.
    - The best error drops below ``_GOOD_ENOUGH_THRESHOLD``.

    Args:
        initial_graph: Starting graph in ``igraph`` format.
        target_annd: Target normalized ANND value for each degree bin.
        target_eccentricity: Optional target eccentricity profile; when
            provided, the eccentricity term is included in the objective.
        rng: Random-number generator; a fresh default RNG is used when
            ``None``.
        debug: When ``True``, prints per-step progress messages and calls
            ``_plot_optimization_trajectory`` after the loop.
    Returns:
        A ``(graph, info)`` tuple where *graph* is the optimised
        ``nx.Graph`` and *info* is a dict with keys ``"best_error"`` and
        ``"best_annd"``, plus ``"best_eccentricity"`` when
        *target_eccentricity* is supplied.
    """
    rng = rng or np.random.default_rng()
    target_annd = np.asarray(target_annd, dtype=float)

    bins = len(target_annd)

    # ---- Initialise state ---------------------------------------------------
    graph_state = GraphState(initial_graph, bins=bins)
    current_annd, _knn_norm = graph_state.get_annd()
    current_ecc = graph_state.get_eccentricity() if target_eccentricity is not None else None

    # Early exit if the graph does not have enough edges for double-edge swap
    if graph_state.num_edges < 2:
        if debug:
            print("Graph has fewer than 2 edges; skipping optimization.")
        annd_errors = _compute_annd_errors(current_annd, target_annd)
        ecc_errors = _compute_eccentricity_errors(current_ecc, target_eccentricity) if current_ecc is not None else None
        current_error = _combined_error(annd_errors, ecc_errors)
        info = {"best_error": current_error, "best_annd": current_annd}
        if target_eccentricity is not None:
            info["best_eccentricity"] = current_ecc
        return igraph_to_networkx(initial_graph), info

    if debug:
        print(f"{'Initial ANND:':<40} {current_annd}")
        print(f"{'Target ANND:':<40} {target_annd}")
        print(f"{'Initial assortativity:':<40} {calculate_degree_assortativity(initial_graph)}")
        print(f"{'Target assortativity:':<40} {_DEBUG_TARGET_ASSORTATIVITY}")

    # ---- SA schedule --------------------------------------------------------
    max_steps = _STEPS_PER_EDGE * graph_state.num_edges
    patience = max(_MIN_PATIENCE, int(max_steps * _PATIENCE_FRACTION))
    temperature = _INITIAL_TEMPERATURE
    cooling = 2 ** (-_COOLING_BITS / max_steps)

    # ---- Initial error ------------------------------------------------------
    annd_errors = _compute_annd_errors(current_annd, target_annd)
    ecc_errors = _compute_eccentricity_errors(current_ecc, target_eccentricity) if current_ecc is not None else None
    current_error = _combined_error(annd_errors, ecc_errors)

    if debug:
        print(f"{'Initial error:':<40} {current_error:.6f}")
        errors_history = [current_error]
        assortativity_errors_history: list[float] = [abs(calculate_degree_assortativity(graph_state.get_graph()) - _DEBUG_TARGET_ASSORTATIVITY)]
    else:
        errors_history = []
        assortativity_errors_history = []

    # ---- Best state snapshot ------------------------------------------------
    best_state = {
        "error": current_error,
        "graph_state": graph_state.copy(),
        "step": 0,
    }

    def _build_info(gs: GraphState) -> dict:
        info: dict = {"best_error": best_state["error"], "best_annd": gs.get_annd()[0]}
        if target_eccentricity is not None:
            info["best_eccentricity"] = gs.get_eccentricity()
        return info

    # Early exit if already good enough
    if best_state["error"] < _GOOD_ENOUGH_THRESHOLD:
        if debug:
            print(f"Early stopping at step 0 — error already below threshold: {best_state['error']:.6f}")
        return igraph_to_networkx(best_state["graph_state"].get_graph()), _build_info(best_state["graph_state"])

    # ---- Main optimisation loop ---------------------------------------------
    steps_without_improvement = 0

    for step in range(1, max_steps + 1):

        # Find a valid proposal (retry up to `patience` times)
        change: GraphChange | None = None
        for _ in range(patience):
            change = _propose_intelligent_double_edge_swap(graph_state, current_annd, target_annd, _knn_norm, rng, current_ecc=current_ecc, target_ecc=target_eccentricity,)
            if change is not None:
                break

        if change is None:
            if debug:
                print(f"Aborting: no valid proposal found after {patience} attempts at step {step}.")
            break

        # Evaluate proposed state
        # Snapshot current knn_norm so we can restore it cheaply on reject.
        prev_knn_norm = _knn_norm
        graph_state.apply_change(change)
        proposed_annd, _knn_norm = graph_state.get_annd()
        proposed_ecc = graph_state.get_eccentricity() if target_eccentricity is not None else None

        proposed_annd_errors = _compute_annd_errors(proposed_annd, target_annd)
        proposed_ecc_errors = _compute_eccentricity_errors(proposed_ecc, target_eccentricity) if proposed_ecc is not None else None
        proposed_error = _combined_error(proposed_annd_errors, proposed_ecc_errors)

        # Acceptance
        if proposed_error < best_state["error"]:
            # Accept: strict improvement to the global best — always keep.
            current_error = proposed_error
            current_annd = proposed_annd
            current_ecc = proposed_ecc
            best_state.update(error=current_error, graph_state=graph_state.copy(), step=step)
            steps_without_improvement = 0
            if debug:
                print(f"Improvement at step {step}: error = {current_error:.4f}, annd = {current_annd}")
        elif rng.random() < np.exp((best_state["error"] - proposed_error) / temperature):
            # Metropolis criterion: accept a non-improving move for exploration.
            current_error = proposed_error
            current_annd = proposed_annd
            current_ecc = proposed_ecc
            steps_without_improvement += 1
        else:
            # Reject: revert graph and restore pre-change knn_norm to avoid
            # a redundant KNN call on the next step.
            graph_state.revert_change(change)
            _knn_norm = prev_knn_norm
            steps_without_improvement += 1

        # Temperature decay
        temperature *= cooling

        if debug:
            errors_history.append(current_error)
            assortativity_errors_history.append(abs(calculate_degree_assortativity(graph_state.get_graph()) - _DEBUG_TARGET_ASSORTATIVITY))

        if steps_without_improvement >= patience:
            if debug:
                print(f"Early stopping at step {step} — {patience} steps without improvement. Best error: {best_state['error']:.6f} at step {best_state['step']}.")
            break

        if best_state["error"] < _GOOD_ENOUGH_THRESHOLD:
            if debug:
                print(f"Early stopping at step {step} — error below threshold: {best_state['error']:.6f}.")
            break

    # ---- Debug summary & plot -----------------------------------------------
    if debug:
        print(f"{'Final ANND error:':<40} {best_state['error']:.6f}")
        _plot_optimization_trajectory(errors_history, assortativity_errors_history)

    best_gs = best_state["graph_state"]
    return igraph_to_networkx(best_gs.get_graph()), _build_info(best_gs)


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------

def _plot_optimization_trajectory(errors_history: list[float], assortativity_errors_history: list[float]) -> None:
    """Plot the objective-function and assortativity-error trajectories.

    Draws the combined ANND (± eccentricity) error on the left y-axis in blue
    and, when *assortativity_errors_history* is non-empty, the absolute
    assortativity error on a secondary right y-axis in red.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(errors_history, color="#2563eb", linewidth=1.5, label="Objective Function")
    ax1.set_xlabel("MCMC Step", fontsize=12)
    ax1.set_ylabel("Total Error", fontsize=12, color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, linestyle="--", alpha=0.7)

    if assortativity_errors_history:
        ax2 = ax1.twinx()
        ax2.plot(assortativity_errors_history, color="#dc2626", linewidth=1.5, label="Assortativity Error")
        ax2.set_ylabel("Assortativity Absolute Error", fontsize=12, color="#dc2626")
        ax2.tick_params(axis="y", labelcolor="#dc2626")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.title("ANNDG Optimization Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
