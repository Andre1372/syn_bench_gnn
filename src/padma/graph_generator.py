""" Graph Generator for PADMA (Probabilistic Annealing for Degree Moments Alignment) """

import numpy as np
import networkx as nx
from scipy import stats
import logging
from typing import Any

from src.padma.maxent_optimizer import maxent_optimize_discrete

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_skewness_bounds(mean_norm: float, var_norm: float, n_nodes: int) -> tuple[float, float]:
    """Compute the feasible range of skewness for a bounded discrete distribution.

    Uses the Pearson bounds for a distribution on [0, 1] with given mean and
    variance.  The result is later scaled to the actual degree support.

    Args:
        mean_norm: Normalized mean degree in ``[0, 1]``.
        var_norm: Normalized variance (must be positive).
        n_nodes: Number of nodes (used to determine the integer support size).
    Returns:
        ``(min_skew, max_skew)`` - feasible skewness bounds.
    """
    std_norm = np.sqrt(var_norm)
    if std_norm < 1e-10:
        return 0.0, 0.0

    # Skewness is equal to E[(X - mu)^3] / sigma^3
    # Since (-mu) < (X - mu) < (1 - mu)
    # E[skew] <= (1-mu) / sigma^3 * E[(X-mu)^2] = (1-mu) / sigma
    # E[skew] >= (-mu) / sigma^3 * E[(X-mu)^2] = (-mu) / sigma
    # min_skew = (-mean_norm) / std_norm
    # max_skew = (1 - mean_norm) / std_norm
    min_skew = -1
    max_skew = 1

    # Pearson's bound for finite samples
    sqrt_n = np.sqrt(n_nodes)
    min_skew = max(min_skew, -sqrt_n)
    max_skew = min(max_skew, sqrt_n)

    return float(min_skew), float(max_skew)


def _find_kurtosis_bounds(mean_norm: float, var_norm: float, skewness: float, n_nodes: int) -> tuple[float, float]:
    """Compute the feasible range of (excess) kurtosis given lower moments.

    Args:
        mean_norm: Normalized mean degree in ``[0, 1]``.
        var_norm: Normalized variance.
        skewness: Skewness of the distribution.
        n_nodes: Number of nodes.
    Returns:
        ``(min_kurt, max_kurt)`` - feasible raw (non-excess) kurtosis bounds.
    """
    # Pearson lower bound for raw kurtosis: kurt ≥ skew² + 1
    min_kurt = skewness ** 2 + 1.0

    # Kurtosis is equal to E[(X - mu)^4] / sigma^4
    # Since (X - mu)^2 <= max(mu^2, (1 - mu)^2)
    # E[kurt] <= max(mu^2, (1 - mu)^2) / sigma^4 * E[(X-mu)^2]
    # max_kurt = max(mean_norm**2, (1 - mean_norm)**2) / var_norm
    max_kurt = float('inf')

    # Upper bound for finite samples
    max_kurt = min(max_kurt, n_nodes)

    return float(min_kurt), float(max_kurt)


def _eg_aware_repair(degree_seq: np.ndarray) -> np.ndarray:
    """Repair a non-graphical degree sequence using the Erdős-Gallai algorithm.

    Iteratively reduces the largest degree until the Erdős-Gallai condition is
    satisfied.  Keeps modifications minimal so that the overall distribution
    shape is preserved as much as possible.

    Args:
        degree_seq: Integer degree sequence (will be copied, not modified in place).
    Returns:
        A repaired integer degree sequence that satisfies the Erdős-Gallai conditions (i.e., is graphical).
    """
    seq = degree_seq.copy().astype(int)
    n = len(seq)

    for _ in range(n * 10):  # safety iteration cap
        seq = np.clip(seq, 0, n - 1)

        # Erdős-Gallai check
        if _is_graphical(seq):
            break

        # Reduce the maximum degree by 1
        idx_max = int(np.argmax(seq))
        seq[idx_max] = max(0, seq[idx_max] - 1)

        # Fix parity if needed
        if seq.sum() % 2 == 1:
            idx_fix = int(np.argmax(seq))
            seq[idx_fix] = max(0, seq[idx_fix] - 1)

    return seq


def _is_graphical(seq: np.ndarray) -> bool:
    """Check whether a degree sequence is graphical via Erdős-Gallai.

    Args:
        seq: Integer array of degrees.
    Returns:
        ``True`` if the sequence can be realised as a simple graph.
    """
    s = sorted(seq.tolist(), reverse=True)
    n = len(s)
    total = sum(s)
    if total % 2 != 0:
        return False
    for k in range(1, n + 1):
        lhs = sum(s[:k])
        rhs = k * (k - 1) + sum(min(s[i], k) for i in range(k, n))
        if lhs > rhs:
            return False
    return True


# ---------------------------------------------------------------------------
# Core degree-sequence generator
# ---------------------------------------------------------------------------

def _generate_degree_sequence(
    n_nodes: int,
    mean_deg_norm: float,
    var_deg_norm: float,
    skewness: float,
    kurtosis: float,
    rng: np.random.Generator,
    normalize_by_size: bool = True,
) -> np.ndarray:
    """Generate an integer degree sequence using `maxent_optimize_discrete`.

    Converts normalized moments (computed on the ``[0,1]`` scale) to the
    actual degree domain, calls the optimizer, and returns the resulting
    sequence.

    Args:
        n_nodes: Number of nodes in the graph.
        mean_deg_norm: Mean degree normalized to ``[0, 1]``.
        var_deg_norm: Variance normalized to ``[0, 1]``.
        skewness: Target skewness (real scale, not normalized).
        kurtosis: Target raw kurtosis (≥ 1).
        normalize_by_size: If ``True``, degrees are on the ``[0, n-1]`` scale; if ``False``, they remain on ``[0, 1]``.
    Returns:
        Integer degree sequence of length ``n_nodes``.
    """
    max_degree = n_nodes - 1

    # --- Denormalize to actual degree scale ---
    if normalize_by_size:
        mean_deg = mean_deg_norm * max_degree
        # Variance: scale from [0,1]² to [0, n-1]² proportionally
        max_var_deg = mean_deg * (max_degree - mean_deg)
        moment_2_norm = var_deg_norm / max(mean_deg_norm * (1 - mean_deg_norm), 1e-10)
        var_deg = moment_2_norm * max_var_deg
    else:
        mean_deg = mean_deg_norm
        max_var_deg = mean_deg_norm * (1.0 - mean_deg_norm)
        moment_2_norm = var_deg_norm / max(max_var_deg, 1e-10)
        var_deg = moment_2_norm * max_var_deg

    var_deg = max(var_deg, 0.0)

    # --- Target sum (must be even for graph realization) ---
    target_sum = int(round(mean_deg * n_nodes))
    if target_sum % 2 == 1:
        target_sum += 1
    # Safety: target_sum cannot exceed n_nodes * max_degree
    target_sum = min(target_sum, n_nodes * max_degree)
    target_sum = max(target_sum, 0)

    # --- Zero-variance edge case: point distribution ---
    if var_deg < 1e-10:
        deg = int(round(mean_deg))
        degrees = np.full(n_nodes, deg, dtype=int)
        return degrees

    # --- MaxEnt discrete optimization ---
    _, degrees = maxent_optimize_discrete(
        target_sum=target_sum,
        target_var=var_deg,
        n_samples=n_nodes,
        max_value=max_degree,
        target_skew=skewness,
        target_kurt=kurtosis,
        rng=rng,
    )

    return degrees.astype(int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_graph(target_stats: dict[str, Any], rng: np.random.Generator, normalize_by_size: bool = True, debug: bool = False) -> tuple[nx.Graph, dict]:
    """Generate a synthetic graph matching the target statistics.

    Pipeline:
        1. Extract target moments from stats.
        2. Clamping/Bounds checking for higher-order moments.
        3. Generate a max-entropy degree sequence.
        4. Repair parity and Erdős-Gallai feasibility.
        5. Build the graph with Havel-Hakimi (configuration_model fallback).
        6. Randomize via double edge swaps.

    Args:
        target_stats: Dictionary containing 'n_nodes' and 'normalized_degree_moments'.
        rng: NumPy random generator.
        normalize_by_size: If True, assumes moments were normalized on [0, 1] scale.
        debug: If True, logs detailed diagnostics.

    Returns:
        A tuple (G_syn, info).
    """
    n = target_stats["n_nodes"]
    if n < 2: raise ValueError(f"Graph must have at least 2 nodes, got {n}.")

    moments = target_stats["normalized_degree_moments"]
    n_moments = len(moments)

    # ── Step 1: extract moments ──────────────────────────────────────────────
    mean_norm = float(moments[0])
    var_norm = float(moments[1]) if n_moments > 1 else 0.0
    target_skew = float(moments[2]) if n_moments > 2 else 0.0
    target_kurt = float(moments[3]) if n_moments > 3 else 3.0

    # ── Step 2: bounds checking ───────────────────────────────────────────────
    if n_moments >= 3 and var_norm > 1e-10:
        min_s, max_s = _find_skewness_bounds(mean_norm, var_norm, n)
        clamped_skew = float(np.clip(target_skew, min_s, max_s))
        if debug and target_skew != clamped_skew:
            logger.warning(f"Target skewness {target_skew:.3f} was out of bounds [{min_s:.3f}, {max_s:.3f}]. Clamped to {clamped_skew:.3f}")
        target_skew = clamped_skew

    if n_moments >= 4 and var_norm > 1e-10:
        min_k, max_k = _find_kurtosis_bounds(mean_norm, var_norm, target_skew, n)
        clamped_kurt = float(np.clip(target_kurt, min_k, max_k))
        if debug and target_kurt != clamped_kurt:
            logger.warning(f"Target raw kurtosis {target_kurt:.3f} was out of bounds [{min_k:.3f}, {max_k:.3f}]. Clamped to {clamped_kurt:.3f}")
        target_kurt = clamped_kurt

    target_moments = {"mean_norm": mean_norm, "var_norm": var_norm, "skew": target_skew, "kurt": target_kurt}

    # ── Step 3: generate max-entropy degree sequence ──────────────────────────
    degree_seq = _generate_degree_sequence(
        n_nodes=n,
        mean_deg_norm=mean_norm,
        var_deg_norm=var_norm,
        skewness=target_skew if n_moments >= 3 else 0.0,
        kurtosis=target_kurt if n_moments >= 4 else 3.0,
        rng=rng,
        normalize_by_size=normalize_by_size,
    )

    # ── Step 4: parity fix + Erdős-Gallai repair ──────────────────────────────
    if degree_seq.sum() % 2 == 1:
        # Increment the smallest non-max degree node to fix parity
        candidates = np.where(degree_seq < n - 1)[0]
        if len(candidates) > 0:
            degree_seq[candidates[0]] += 1
        else:
            degree_seq[np.argmax(degree_seq)] -= 1

    pre_repair_seq = degree_seq.copy()
    degree_seq = _eg_aware_repair(degree_seq)
    
    if debug and not np.array_equal(pre_repair_seq, degree_seq):
        diff = np.abs(pre_repair_seq - degree_seq).sum()
        logger.warning(f"Erdős-Gallai condition was violated. Sequence repaired by modifying {diff} degrees.")

    if debug:
        logger.info(f"Sequence after Erdős-Gallai repair. Graphical: {_is_graphical(degree_seq)}, total degrees: {degree_seq.sum()}")

    # ── Step 5: build graph (Havel-Hakimi preferred) ──────────────────────────
    try:
        G_syn = nx.havel_hakimi_graph(degree_seq.tolist())
    except Exception as e:
        if debug:
            logger.warning(f"nx.havel_hakimi_graph failed: {e}. Falling back to configuration_model.")
        G_multi = nx.configuration_model(degree_seq.tolist(), seed=int(rng.integers(0, 2**31)))
        G_syn = nx.Graph(G_multi)
        G_syn.remove_edges_from(nx.selfloop_edges(G_syn))

    # Ensure all original node IDs exist (Havel-Hakimi may drop isolated nodes)
    G_syn.add_nodes_from(range(n))

    # ── Step 6: double edge swaps for randomization ───────────────────────────
    m_actual = G_syn.number_of_edges()
    if m_actual > 0:
        n_swaps = 10 * m_actual
        try:
            nx.double_edge_swap(G_syn, nswap=n_swaps, max_tries=n_swaps * 10, seed=int(rng.integers(0, 2**31)))
        except (nx.NetworkXError, Exception) as e:
            # Categorize as warning regardless of type since edge swaps are best-effort
            # and the graph remains valid even if not perfectly randomized.
            logger.warning(
                "nx.double_edge_swap failed (partially randomized graph). "
                "Swaps requested: %d, Error: %s", n_swaps, e
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    actual_degs = np.array([G_syn.degree(v) for v in G_syn.nodes()], dtype=float)
    actual_moments = {
        "mean": float(np.mean(actual_degs)),
        "var": float(np.var(actual_degs, ddof=0)),
        "skew": float(stats.skew(actual_degs, bias=False)) if np.var(actual_degs) > 1e-10 else 0.0,
        "kurt": float(stats.kurtosis(actual_degs, bias=False)) + 3.0 if np.var(actual_degs) > 1e-10 else 3.0,
    }
    info = {
        "target_moments": target_moments,
        "actual_moments": actual_moments,
        "degree_seq": degree_seq,
    }

    return G_syn, info
