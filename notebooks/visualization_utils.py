"""Utility functions for analyzing and visualizing results in notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from src.data_utils import networkx_to_igraph
from src.graph_analysis import calculate_annd


def plot_graph(
    graph: nx.Graph, 
    ax: plt.Axes, 
    dataset_name: str,
    graph_index: int,
    fixed_pos: dict[int, tuple[float, float]] | None = None
) -> None:
    """Visualizes a single graph on a given Axes with its motif counts.

    Args:
        graph: The networkx graph to render.
        ax: The matplotlib Axes to draw onto.
        dataset_name: Name of the dataset for the title.
        graph_index: Index of the graph for the title.
        fixed_pos: Optional dictionary of node positions for consistent layout.
    """
    num_nodes: int = graph.number_of_nodes()
    num_edges: int = graph.number_of_edges()

    # Use the fixed positions from the original graph for consistency
    pos = fixed_pos if fixed_pos is not None else nx.spring_layout(graph, seed=42)

    nx.draw(graph, pos, ax=ax, node_size=30, node_color="skyblue", edge_color='gray', with_labels=False)
    ax.set_title(
        f"Dataset: {dataset_name}  |  Index: {graph_index}\n"
        f"Nodes: {num_nodes}   Edges: {num_edges}",
        fontsize=10,
        pad=12,
    )
    ax.axis("off")


def plot_annd(graph: nx.Graph, ax: plt.Axes, title: str = "Average Nearest Neighbor Degree", label: str = 'Empirical $k_{nn}(k)$') -> None:
    """Visualizes the average nearest neighbor degree for a given graph.

    Args:
        graph: The networkx graph to analyze.
        ax: The matplotlib Axes to draw onto.
        title: Title for the plot.
        label: Label for the legend.
    """
    # Convert to igraph to use the shared analysis logic
    ig_graph = networkx_to_igraph(graph)
    annd_values = calculate_annd(ig_graph)
    
    if len(annd_values) == 0:
        return
    
    # calculate_annd returns an array where index k corresponds to degree k+1
    degrees = np.arange(1, len(annd_values) + 1)
    values = annd_values
    
    # Filter out entries where ANND is 0 (meaning degree not present or no neighbors)
    mask = values > 0
    degrees = degrees[mask]
    values = values[mask]

    if len(degrees) == 0:
        return

    ax.scatter(degrees, values, color='#2c3e50', s=100, alpha=0.8, edgecolors='white', linewidth=1.5, label=label)
    ax.plot(degrees, values, color='#3498db', linestyle='-', linewidth=2, alpha=0.5)

    ax.set_xlabel("Degree $k$", fontsize=12, fontweight='bold')
    ax.set_ylabel("Avg. Neighbor Degree $k_{nn}(k)$", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if label:
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)


def add_baseline_guide(
    ax: plt.Axes,
    vals: pd.Series | np.ndarray,
    color: str = "#5B9BD5",
    label: str = "Original",
    zorder: int = 0,
) -> None:
    """Adds horizontal bands representing the distribution of baseline values.

    Args:
        ax: The matplotlib Axes to draw onto.
        vals: The baseline values to compute statistics from.
        color: The color for the bands and line.
        label: Label suffix for the legend.
        zorder: Drawing order.
    """
    if len(vals) == 0:
        return

    # Calculate statistics
    v_min, v_max = np.min(vals), np.max(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    v_median = np.median(vals)

    # 1. Outer band (100% range)
    ax.axhspan(v_min, v_max, color=color, alpha=0.25, zorder=zorder, label=f"{label} Range (100%)")
    
    # 2. Inner band (IQR - 50% central)
    ax.axhspan(q1, q3, color=color, alpha=0.5, zorder=zorder + 0.05, label=f"{label} IQR")
    
    # 3. Median line
    ax.axhline(v_median, color=color, linestyle="--", linewidth=1.2, alpha=1, zorder=zorder + 0.1)


def plot_performance_distribution(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    palette: dict[str, str] | str | None = None,
    order: list[Any] | None = None,
    hue_order: list[str] | None = None,
) -> plt.Axes:
    """Generates a styled boxplot with a stripplot overlay for performance data.

    Args:
        ax: The matplotlib Axes to draw onto.
        df: The DataFrame containing the data to plot.
        x: The column name for the X-axis.
        y: The column name for the Y-axis.
        hue: The optional column name for color grouping.
        palette: A mapping of groups to colors, or a seaborn palette name.
        order: An explicit order for X-axis categories.
        hue_order: An explicit order for hue/color categories.

    Returns:
        The modified matplotlib Axes.
    """
    # 1. Determine Plotting Order
    x_categories = order if order is not None else df[x].unique().tolist()
    hue_categories = hue_order if hue_order is not None else (df[hue].unique().tolist() if hue else None)
    
    # Intelligent Dodge Detection:
    # Disable dodging if every X position has only 1 hue value
    do_dodge = False
    if hue:
        hues_per_x = df.groupby(x, sort=False, observed=True)[hue].nunique()
        if hues_per_x.max() > 1:
            do_dodge = True

    # 2. Boxplot
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=x_categories,
        hue_order=hue_categories,
        palette=palette,
        dodge=do_dodge,
        linewidth=1.2,
        fliersize=0,
        showmeans=True,
        meanline=True,
        boxprops={"edgecolor": "none"},
        medianprops={"linewidth": 0},
        meanprops={"color": "#646464", "linewidth": 1.5, "linestyle": "-"},
        ax=ax,
    )

    # 3. Stripplot Overlay
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=x_categories,
        hue_order=hue_categories,
        dodge=do_dodge,
        alpha=0.6,
        palette="dark:black",
        size=4,
        jitter=0.15,
        legend=False,
        ax=ax,
    )

    # 4. Aesthetics
    # Internal position columns (prefixed with '_') have no meaningful label.
    x_label = "" if x.startswith("_") else x.replace("_", " ").title()
    ax.set_xlabel(x_label, fontsize=10)
    ylabel = y.replace("test_", "").upper().replace("_", "-")
    ax.set_ylabel(ylabel, fontsize=10)
            
    # Clean up duplicate legends generated by seaborn
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        
    return ax


def set_dynamic_ylim(ax: plt.Axes, data: np.ndarray | list[np.ndarray], percentile: float = 95, expansion: float = 1.25) -> None:
    """Sets the Y-axis limits dynamically based on the distribution of data.
    
    Useful for filtering out extreme initial spikes in optimization logs.
    
    Args:
        ax: The matplotlib Axes to modify.
        data: A single array or a list of arrays to compute percentiles from.
        percentile: The percentile to use for the limit (e.g., 95 filters top 5%).
        expansion: Multiplier to add some padding around the calculated limits.
    """
    if isinstance(data, list):
        # Flatten and remove NaNs for multi-sequence estimation
        all_data = np.concatenate([np.array(d).flatten() for d in data])
    else:
        all_data = data.flatten()
        
    valid_data = all_data[~np.isnan(all_data)]
    if len(valid_data) == 0:
        return

    # Calculate asymmetric or symmetric bounds based on data range
    v_min = np.percentile(valid_data, 100 - percentile)
    v_max = np.percentile(valid_data, percentile)
    
    # Handle zero range
    if v_max == v_min:
        v_min -= 0.5
        v_max += 0.5

    # Apply expansion
    center = (v_max + v_min) / 2
    half_range = (v_max - v_min) / 2 * expansion
    
    ax.set_ylim(center - half_range, center + half_range)