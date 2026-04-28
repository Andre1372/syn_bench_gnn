""" Graph Generator for ANNDG (Average Nearest Neighbor Degree Generator) """

import numpy as np
import igraph as ig
import networkx as nx
from typing import Any

from src.padma.graph_generator import generate_graph as padma_generate_graph

from src.data_utils import networkx_to_igraph, igraph_to_networkx

from src.anndg.optimizer import optimizer
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_graph(
    target_stats: dict[str, Any], 
    replicate_diameter: bool = False, 
    rng: np.random.Generator = None,
    debug: bool = False
) -> tuple[nx.Graph, dict]:
    """
    Generate a graph using the ANNDG algorithm.

    Args:
        target_stats: Dictionary containing the target statistics.
        replicate_diameter: Whether to include diameter in the objective function.
        rng: Random number generator.
        debug: Whether to show optimization progress and plots.
    Returns:
        Tuple of (graph, info)
    """
    if rng is None: rng = np.random.default_rng()

    nx_graph, info = padma_generate_graph(target_stats, rng)

    target_diameter = target_stats["diameter"] if replicate_diameter else None    
    best_state, best_error = optimizer(
        networkx_to_igraph(nx_graph), 
        target_stats["annd"], 
        target_diameter=target_diameter, 
        rng=rng,
        debug=debug
    )
    info["best_objective"] = best_error

    best_graph = best_state.get_graph()
    best_graph = igraph_to_networkx(best_graph)
    
    return best_graph, info
