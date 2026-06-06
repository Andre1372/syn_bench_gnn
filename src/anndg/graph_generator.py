"""Graph generator for the ANNDG (Average Nearest Neighbor Degree) pipeline."""

import numpy as np
import networkx as nx
from typing import Any

from src.padma.graph_generator import generate_graph as padma_generate_graph

from src.data_utils import networkx_to_igraph

from src.anndg.optimizer import optimizer


def generate_graph(
    target_stats: dict[str, Any], 
    replicate_eccentricity: bool = False,
    rng: np.random.Generator = None,
    debug: bool = False
) -> tuple[nx.Graph, dict]:
    """Generate a synthetic graph whose ANND profile matches *target_stats*.

    The function first delegates to the PADMA generator to produce a graph
    whose degree distribution matches the target, then passes that graph to
    the ANNDG optimizer which adjusts edge connections via simulated-annealing
    double-edge swaps to minimise the distance to the target ANND profile.
    When *replicate_eccentricity* is ``True``, the eccentricity profile is
    included as an additional term in the optimizer's objective function.

    Args:
        target_stats: Dictionary of target graph statistics.  Must contain at
            least ``"annd"`` (per-bin ANND array) and the keys expected by
            the PADMA generator.  If *replicate_eccentricity* is ``True``,
            ``"eccentricity"`` must also be present.
        replicate_eccentricity: When ``True``, the eccentricity profile stored
            in ``target_stats["eccentricity"]`` is included in the optimizer
            objective alongside the ANND term.
        rng: Random number generator.  A fresh ``numpy`` default RNG is created
            when ``None`` is passed.
        debug: When ``True``, the optimizer prints progress messages and
            displays an error-trajectory plot at the end.
    Returns:
        A ``(graph, info)`` tuple where *graph* is the optimised
        ``networkx.Graph`` and *info* is a dictionary that merges the PADMA
        generation metadata with the optimizer diagnostics (e.g.
        ``"best_error"``, ``"best_annd"``).
    """
    if rng is None: rng = np.random.default_rng()

    nx_graph, info = padma_generate_graph(target_stats, rng)

    target_eccentricity = target_stats["eccentricity"] if replicate_eccentricity else None    
    best_graph, opt_info = optimizer(
        networkx_to_igraph(nx_graph), 
        target_stats["annd"], 
        target_eccentricity=target_eccentricity,
        rng=rng,
        debug=debug
    )
    info.update(opt_info)
    
    return best_graph, info
