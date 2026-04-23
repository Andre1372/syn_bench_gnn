import sys
from pathlib import Path
# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import igraph as ig
import numpy as np
from typing import Any
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import TUDataset
from src.data_utils import igraph_to_networkx, pytorch_to_networkx, pytorch_to_igraph, networkx_to_igraph
from src.anndg.graph_state import GraphState, GraphChange
from src.padma.graph_generator import generate_graph as padma_generate_graph

PROJECT_ROOT = project_root

def get_degree_distribution(graph: ig.Graph) -> np.ndarray:
    """Calculates the normalized degree distribution of the graph using igraph.
    
    Returns:
        A numpy array where entry i is the fraction of nodes with degree i.
        The length of the array is always max_degree + 1.
    """
    n = graph.vcount()
    if n == 0: return np.array([], dtype=float)
    
    degrees = graph.degree()
    counts = np.bincount(degrees)
    return counts / n


def optimizer(initial_graph: ig.Graph, target_annd: np.ndarray, rng: np.random.Generator=None, debug: bool = False):
    """
    Args:
        initial_graph: The initial graph.
        target_annd: The target mean degree for each degree in the original graph.
        rng: The random number generator.
    """

    if rng is None: rng = np.random.default_rng()
    
    graph_state = GraphState(initial_graph)
    degree_distribution = get_degree_distribution(initial_graph)[1:] # exclude degree = 0
    
    def _compute_objective(actual_annd: np.ndarray, target_annd: np.ndarray, weights: np.ndarray) -> float:
        """Compute the objective function."""
        len_actual = len(actual_annd)
        len_target = len(target_annd)
        max_k = max(len_actual, len_target)
        
        # Align lengths by padding with zeros to ensure element-wise operations work.
        v_actual = np.pad(actual_annd, (0, max_k - len_actual))
        v_target = np.pad(target_annd, (0, max_k - len_target))
        v_weights = np.pad(weights, (0, max_k - len(weights)))

        diff = v_actual - v_target
        
        error = np.sqrt(np.dot(v_weights, diff * diff))
        
        return float(error)

    def _propose_change(graph_state: GraphState, rng: np.random.Generator) -> GraphChange:
        """Propose a random change to the graph."""
        if graph_state.num_edges < 2:
            return None
            
        e1 = graph_state.get_random_edge(rng)
        e2 = graph_state.get_random_edge(rng)
        
        u, v = e1
        x, y = e2

        # Validity check (1): no shared nodes among sampled edges
        if len({u, v, x, y}) != 4:
            return None

        # Coin flip for symmetric proposals (Cross vs Parallel)
        rand_flip = rng.integers(0, 2)
        if rand_flip == 0:
            proposed_edges = [(u, y), (x, v)]
        else:
            proposed_edges = [(u, x), (v, y)]

        # Validity check (2): proposed new edges do not already exist
        for a, b in proposed_edges:
            if graph_state.has_edge(a, b):
                return None

        # Build valid proposal
        return GraphChange(
            edges_to_add=proposed_edges,
            edges_to_remove=[e1, e2]
        )
        
    # Initial error
    current_error = _compute_objective(graph_state.get_annd(), target_annd, weights=degree_distribution)
    if debug:
        print(f"Initial ANND error: {current_error:.6f}")
    best_error = current_error
    best_state = graph_state.copy()
    
    errors_history = [current_error] if debug else None

    max_steps = 1000
    
    # Main loop
    for step in range(max_steps):
        change = _propose_change(graph_state, rng)
        if change is None: 
            if debug: errors_history.append(current_error)
            continue

        graph_state.apply_change(change)
        current_error = _compute_objective(graph_state.get_annd(), target_annd, weights=degree_distribution)
        
        if current_error < best_error:
            best_error = current_error
            best_state = graph_state.copy()
        else:
            if rng.random() < np.exp((best_error - current_error) / 0.5):
                graph_state.revert_change(change)
                current_error = best_error # reset current error
        
        if debug:
            errors_history.append(current_error)
            
        if debug and (step + 1) % 200 == 0:
            print(f"Step {step+1}/{max_steps} - Best error: {best_error:.6f}")
        
    if debug:
        print(f"Final ANND error: {best_error:.6f}")
    
    # Plot and save errors
    if debug:
        plt.figure(figsize=(10, 6))
        plt.plot(errors_history, color='#2563eb', linewidth=1.5)
        plt.title("ANNDG Optimization Progress", fontsize=14, fontweight='bold')
        plt.xlabel("MCMC Step", fontsize=12)
        plt.ylabel("ANND Error", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure results directory exists
        results_dir = project_root / 'results'
        results_dir.mkdir(exist_ok=True)
        
        save_path = results_dir / 'anndg_optimization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Optimization plot saved to: {save_path}")

    return best_state

if __name__ == "__main__":
    
    def generate_graph(target_stats: dict[str, Any], rng: np.random.Generator=None) -> tuple[nx.Graph, dict]:
        """
        Generate a graph using the ANNDG algorithm.

        Args:
            target_stats: Dictionary containing the target statistics.
            rng: Random number generator.
        Returns:
            Tuple of (graph, info)
        """
        if rng is None: rng = np.random.default_rng()

        nx_graph, info = padma_generate_graph(target_stats, rng)

        best_state = optimizer(networkx_to_igraph(nx_graph), target_stats["annd"], rng)

        best_graph = best_state.get_graph()
        best_graph = igraph_to_networkx(best_graph)
        
        return best_graph, info

    DATASET = "DHFR"
    IDX = 5

    dataset = TUDataset(root=PROJECT_ROOT / 'data', name=DATASET)
    GRAPH = pytorch_to_networkx(dataset[IDX])

    from src.graph_analysis import analyze_single_graph

    target_stats = analyze_single_graph(networkx_to_igraph(GRAPH))
    
    graph_gen, info = generate_graph(target_stats, rng=np.random.default_rng(42))