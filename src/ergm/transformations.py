"""Module defining MCMC state transformation strategies."""

import abc
import numpy as np

from src.ergm.graph_state import GraphChange, GraphState


class TransformationStrategy(abc.ABC):
    """Abstract base class for graph transformation strategies.

    Strategies must be stateless and propose valid GraphChange objects
    based only on the current state and random variables.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        """Initializes the transformation strategy."""
        self._rng = rng

    @abc.abstractmethod
    def propose_change(self, current_state: GraphState) -> GraphChange | None:
        """Proposes a graph mutation without altering the state directly."""
        pass

    @abc.abstractmethod
    def expected_changes(self, current_state: GraphState) -> float:
        """Returns the expected number of (edges removed + edges added) in every proposed move."""
        pass


class DoubleEdgeSwapStrategy(TransformationStrategy):
    """Symmetric link switch strategy preserving degree sequences."""

    def propose_change(self, current_state: GraphState) -> GraphChange | None:
        """Proposes a degree-preserving edge swap.

        Args:
            current_state: The current GraphState representing the graph.
        Returns:
            A GraphChange with the proposed swap. None if the move
            is invalid (e.g., node collisions or edge existence).
        """
        if current_state.num_edges < 2:
            return None
            
        e1 = current_state.get_random_edge(self._rng)
        e2 = current_state.get_random_edge(self._rng)
        
        u, v = e1
        x, y = e2

        # Validity check (1): no shared nodes among sampled edges
        if len({u, v, x, y}) != 4:
            return None

        # Coin flip for symmetric proposals (Cross vs Parallel)
        rand_flip = self._rng.integers(0, 2)
        if rand_flip == 0:
            proposed_edges = [(u, y), (x, v)]
        else:
            proposed_edges = [(u, x), (v, y)]

        # Validity check (2): proposed new edges do not already exist
        for a, b in proposed_edges:
            if current_state.has_edge(a, b):
                return None

        # Build valid proposal
        return GraphChange(
            edges_to_add=proposed_edges,
            edges_to_remove=[e1, e2]
        )

    def expected_changes(self, current_state: GraphState) -> float:
        """Returns the expected number of (edges removed + edges added) in every proposed move.

        Args:
            current_state: The current GraphState representing the graph.
        Returns:
            A float representing the expected number of (edges removed + edges added) in every proposed move.
        """
        return 4.0


class MultipleEdgeSwapStrategy(TransformationStrategy):
    """Symmetric link switch strategy preserving degree sequences."""

    def __init__(self, rng: np.random.Generator, p: float = 0.5) -> None:
        """Initializes the transformation strategy.

        Args:
            rng: The random number generator instance for reproducibility.
            p: The probability of increasing the number of edges in the swap.
        """
        super().__init__(rng)
        self._p: float = p

    def propose_change(self, current_state: GraphState) -> GraphChange | None:
        """Proposes a degree-preserving edge swap of k edges following a geometric distribution.

        Args:
            current_state: The current GraphState representing the graph.
        Returns:
            A GraphChange with the proposed swap. None if the move
            is invalid (e.g., node collisions or edge existence).
        """
        k = self._rng.geometric(p=self._p) + 1  # Ensure k >= 2 for a valid cycle swap
        if current_state.num_edges < k:
            return None

        # Sample k edges independently
        sampled_edges = [current_state.get_random_edge(self._rng) for _ in range(k)]

        # Validity check (1): all sampled edges are different
        if len(set(sampled_edges)) != k:
            return None

        # Random direction assignment
        sources, targets = [], []
        for i, (u, v) in enumerate(sampled_edges):
            if self._rng.random() < 0.5:
                sources.append(u)
                targets.append(v)
            else:
                sources.append(v)
                targets.append(u)

        # Proposal
        edges_to_add = []
        for i in range(k):
            u_i = sources[i]
            v_next = targets[(i + 1) % k]
            
            # Validity check (2): no self loops
            if u_i == v_next:
                return None
            
            # Validity check (3): no existing edges
            if current_state.has_edge(u_i, v_next):
                return None
            
            # Validity check (4): no duplicate adds
            if (u_i, v_next) in edges_to_add or (v_next, u_i) in edges_to_add:
                return None

            edges_to_add.append((u_i, v_next))

        return GraphChange(
            edges_to_add=edges_to_add,
            edges_to_remove=sampled_edges
        )

    def expected_changes(self, current_state: GraphState) -> float:
        """Returns the expected number of (edges removed + edges added) in every proposed move.

        Args:
            current_state: The current GraphState representing the graph.
        Returns:
            A float representing the expected number of (edges removed + edges added) in every proposed move.
        """
        return 2.0 * (1.0 + 1.0 / self._p)
