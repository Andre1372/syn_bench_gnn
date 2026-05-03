""" Graph Generator for ANNDG (Average Nearest Neighbor Degree Generator) """

import numpy as np
import networkx as nx
from typing import Any

from src.padma.graph_generator import generate_graph as padma_generate_graph

from src.data_utils import networkx_to_igraph

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
    replicate_ecc_moments: bool = False,
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
    target_ecc_moments = target_stats["ecc_moments"] if replicate_ecc_moments else None    
    best_graph, opt_info = optimizer(
        networkx_to_igraph(nx_graph), 
        target_stats["annd"], 
        target_diameter=target_diameter, 
        target_ecc_moments=target_ecc_moments,
        rng=rng,
        debug=debug
    )
    info.update(opt_info)
    
    return best_graph, info
