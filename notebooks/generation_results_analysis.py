"""
This script is the python replica of a notebook, so it is not meant to be run as a script.
"""

# Cell 0 - Imports
import re
import sys
from pathlib import Path
from typing import Any, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
import seaborn as sns

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import DatasetPT, load_all_synthetic_variants, pytorch_to_igraph
from src.graph_analysis import calculate_moments_error, calculate_annd_error, calculate_eccentricity_error
from notebooks.visualization_utils import plot_performance_distribution, set_dynamic_ylim


# Cell 1 - Global Variables & Data Loading
DATASET_NAMES = ["BZR", "DHFR", "Mutagenicity", "MUTAG"]
METHODS = ["padma", "anndg", "anndgE", "nextGen"]
IGNORED_METRICS = ["diameter"]
PALETTE = {
    "nextGen": "#5B9BD5", 
    "dummyNodes": "#DA7CF7",
    "dummyEdges": "#98C379", 
    "padma": "#F5C431", 
    "anndg": "#E06C75",
    "anndgD": "#5CE9FF", 
    "anndgE": "#C47900",
    "anndgED": "#5242D1", 
    }

def load_generation_data() -> pd.DataFrame:
    """Load and preprocess all original datasets and synthetic variants saving all per-graph statistics."""
    rows = []
    original_stats_map = {}

    def extract_errors(stats):
        """Extracts stats as numpy arrays for error calculation."""
        deg_moments = np.array(stats.get("normalized_degree_moments", [0]*4))
        annd = np.array(stats.get("annd", [0]*4))
        ecc_moments = np.array(stats.get("ecc_moments", [0]*4))
        return deg_moments, annd, ecc_moments

    for dataset_name in DATASET_NAMES:
        # Load original
        orig_pt_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_original.pt"
        if orig_pt_path.exists():
            dataset_obj = DatasetPT(orig_pt_path)
            per_graph_stats = dataset_obj.metadata.get("per_graph_statistics", [])
            for i, stats in enumerate(per_graph_stats):
                deg_moments, annd, ecc_moments = extract_errors(stats)
                original_stats_map[(dataset_name, i)] = {
                    "deg_moments": deg_moments,
                    "annd": annd,
                    "ecc_moments": ecc_moments,
                }
                
                row = {
                    "dataset": dataset_name,
                    "method": "original",
                    "graph_idx": i,
                    "variant_idx": -1,
                    "seed": pd.NA,
                    "deg_moments_error": 0.0,
                    "annd_error": 0.0,
                    "ecc_moments_error": 0.0,
                }
                # Flatten the stats dictionary
                for key, value in stats.items():
                    if isinstance(value, (list, np.ndarray)):
                        for idx, val in enumerate(value):
                            row[f"{key}_{idx}"] = val
                    else:
                        row[key] = value
                rows.append(row)
        else:
            print(f"Warning: Original dataset not found at {orig_pt_path}")
            
        # Load synthetic variants
        for method in METHODS:
            method_dir = PROJECT_ROOT / "synthetic_data" / dataset_name / method
            if not method_dir.exists():
                continue
            
            variant_paths = load_all_synthetic_variants(method_dir, dataset_name)
            for v_idx, v_path in enumerate(variant_paths):
                dataset_obj = DatasetPT(v_path)
                per_graph_stats = dataset_obj.metadata.get("per_graph_statistics", [])
                seeds = dataset_obj.metadata.get("seeds", [])
                for i, stats in enumerate(per_graph_stats):
                    # Get target stats
                    orig = original_stats_map.get((dataset_name, i))
                    if orig:
                        deg_moments, annd, ecc_moments = extract_errors(stats)
                        
                        dm_err = calculate_moments_error(deg_moments, orig["deg_moments"])
                        annd_err = calculate_annd_error(annd, orig["annd"])
                        ecc_err = calculate_eccentricity_error(ecc_moments, orig["ecc_moments"])
                    else:
                        dm_err = annd_err = ecc_err = np.nan

                    row = {
                        "dataset": dataset_name,
                        "method": method,
                        "graph_idx": i,
                        "variant_idx": v_idx,
                        "deg_moments_error": dm_err,
                        "annd_error": annd_err,
                        "ecc_moments_error": ecc_err,
                        "seed": seeds[i] if i < len(seeds) else pd.NA,
                    }
                    # Flatten the stats dictionary
                    for key, value in stats.items():
                        if isinstance(value, (list, np.ndarray)):
                            for idx, val in enumerate(value):
                                row[f"{key}_{idx}"] = val
                        else:
                            row[key] = value
                    rows.append(row)
            
    df = pd.DataFrame(rows)
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype("Int64")
    return df

def compute_generation_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates absolute deltas for all statistics between synthetic and original graphs.
    
    Returns a DataFrame where each row is a synthetic graph variant and columns 
    contain the absolute difference |orig - synth| for each metric.
    """
    # 1. Separate original and synthetic data
    df_orig = df[df["method"] == "original"].set_index(["dataset", "graph_idx"])
    df_synth = df[df["method"] != "original"].copy()
    
    # 2. Identify columns to compute deltas for (all stats columns)
    metadata_cols = ["dataset", "method", "graph_idx", "variant_idx", "seed",
                     "deg_moments_error", "annd_error", "ecc_moments_error"]
    stat_cols = [c for c in df.columns if c not in metadata_cols and c not in IGNORED_METRICS]
    
    # 3. Vectorized delta calculation for each statistic
    error_rows = []
    # Grouping by dataset and graph_idx to match variants to their originals
    for (dataset, g_idx), group in df_synth.groupby(["dataset", "graph_idx"], observed=True):
        if (dataset, g_idx) not in df_orig.index:
            continue
        orig_row = df_orig.loc[(dataset, g_idx)]
        
        for _, synth_row in group.iterrows():
            err_row = {
                "dataset": dataset,
                "method": synth_row["method"],
                "graph_idx": g_idx,
                "variant_idx": synth_row["variant_idx"],
                "seed": synth_row.get("seed", np.nan),
                "deg_moments_error": synth_row["deg_moments_error"],
                "annd_error": synth_row["annd_error"],
                "ecc_moments_error": synth_row["ecc_moments_error"]
            }
            # Compute absolute delta for each topological statistic
            for col in stat_cols:
                if col in synth_row and col in orig_row:
                    err_row[col] = abs(synth_row[col] - orig_row[col])
            error_rows.append(err_row)
            
    df_errs = pd.DataFrame(error_rows)
    if "seed" in df_errs.columns:
        df_errs["seed"] = df_errs["seed"].astype("Int64")
    return df_errs

# Load data and compute errors
df = load_generation_data()
df_errors = compute_generation_errors(df)

# Global config variables used in analysis
filtered_src = df[df["method"] != "original"].sort_values("method")
METHODS_PER_DATASET = filtered_src.groupby("dataset", observed=True)["method"].unique().apply(list).to_dict()
DATASETS = [d for d in DATASET_NAMES if d in df["dataset"].unique()]



# Cell 2 - Error Analysis
def analyze_generation_quality(dataset_name: str, stats_to_plot: list[str]) -> None:
    """Analyzes the quality of generation by plotting the distribution of absolute errors for various statistics.
    
    Each statistic is plotted on its own axis with dynamic limits focused on the tightest distribution among methods.
    """
    # 1. Filter and prepare data from the error dataframe
    dataset_errs = df_errors[df_errors["dataset"] == dataset_name].copy()
    
    if dataset_errs.empty:
        print(f"No error data found for dataset {dataset_name}.")
        return

    # 2. Reshape data for plotting (melt)
    melted_df = dataset_errs.melt(
        id_vars=["dataset", "method", "graph_idx", "variant_idx"],
        value_vars=[s for s in stats_to_plot if s in dataset_errs.columns],
        var_name="stat",
        value_name="error"
    )
    
    if melted_df.empty:
        print(f"No data to plot for dataset {dataset_name} and stats {stats_to_plot}")
        return

    # 3. Grid plotting logic: each stat with its own dynamic Y-axis
    n_stats = len(stats_to_plot)
    max_cols = 5
    n_cols = min(n_stats, max_cols)
    n_rows = (n_stats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, stat in enumerate(stats_to_plot):
        r, c = divmod(i, n_cols)
        ax = axes_flat[i]
        df_stat = melted_df[melted_df["stat"] == stat].dropna(subset=["error"])
        
        if df_stat.empty: continue

        present_methods = [m for m in METHODS if m in df_stat["method"].unique()]
        
        plot_performance_distribution(
            ax=ax,
            df=df_stat,
            x="method",
            y="error",
            hue="method",
            palette=PALETTE,
            order=present_methods,
            hue_order=present_methods
        )
        
        # 4. Find the "tightest" distribution among methods to set Y-axis limits
        method_stats = {}
        percentile = 100
        for m in present_methods:
            m_errors = df_stat[df_stat["method"] == m]["error"].dropna().values
            if len(m_errors) > 0:
                v_max = np.percentile(m_errors, percentile)
                method_stats[m] = {"max": v_max, "data": m_errors}
        
        if method_stats:
            tightest_method = min(method_stats.keys(), key=lambda k: method_stats[k]["max"])
            set_dynamic_ylim(ax, method_stats[tightest_method]["data"], percentile=percentile, expansion=1.1)
        
        # Aesthetics
        ax.set_title(stat.replace("_", " ").title(), fontsize=12, fontweight='bold')
        if c == 0:
            ax.set_ylabel("Abs Error $|x_{orig} - x_{synth}|$", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([]) 
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Remove empty axes
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes_flat[j])

    # 5. Global Legend (Top, under title)
    available_methods = [m for m in METHODS if m in dataset_errs["method"].unique()]
    legend_patches = [mpatches.Patch(color=PALETTE[m], label=m) for m in available_methods]
    
    # Place legend horizontally at the top
    fig.legend(handles=legend_patches, title="Methods", loc='upper center', 
               bbox_to_anchor=(0.5, 0.94), ncol=len(available_methods), 
               fontsize=11, title_fontsize=12, frameon=False)

    plt.suptitle(f"Generation Quality Analysis: {dataset_name}", fontsize=22, fontweight='bold', y=0.99)
    
    # Adjust layout to make room for suptitle and horizontal legend at the top
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    plt.show()


def get_top_errors(metric: str, dataset_name: str, method_name: str, n: int = 5) -> pd.DataFrame:
    """Finds and returns the top n graphs with the highest error for a given metric using df_errors."""
    # 1. Filter from pre-calculated error dataframe
    subset = df_errors[(df_errors["dataset"] == dataset_name) & (df_errors["method"] == method_name)].copy()
    
    if subset.empty:
        print(f"No error data found for dataset {dataset_name} and method {method_name}.")
        return pd.DataFrame()

    if metric not in subset.columns:
        print(f"Metric '{metric}' not found in the error dataframe.")
        return pd.DataFrame()

    # 2. Sort and take top n
    top_n = subset.sort_values(by=metric, ascending=False).head(n)
    
    print(f"\n--- Top {n} errors for metric '{metric}' (Dataset: {dataset_name}, Method: {method_name}) ---")
    
    # 3. Display results
    display_cols = ["dataset", "method", "graph_idx", "variant_idx", "seed", metric, "annd_0", "annd_1", "annd_2", "annd_3", "annd_error"]
    with pd.option_context('display.expand_frame_repr', False, 'display.width', 1000, 'display.max_columns', None):
        display(top_n[display_cols])
        
    return top_n


# Example Analysis
plot_stats = ["modularity", "clustering", "assortativity", "efficiency", "diameter",  
 "normalized_degree_moments_0", "normalized_degree_moments_1", "normalized_degree_moments_2", "normalized_degree_moments_3", "deg_moments_error", 
 "annd_0", "annd_1", "annd_2", "annd_3", "annd_error", 
 "ecc_moments_0", "ecc_moments_1", "ecc_moments_2", "ecc_moments_3", "ecc_moments_error"]

plot_stats = [s for s in plot_stats if s not in IGNORED_METRICS]

for d in DATASETS:
    analyze_generation_quality(d, plot_stats)
    # get_top_errors("assortativity", d, "anndg", n=5)



# Cell 3 - Global Summary
def generate_global_summary(df_errs: pd.DataFrame, metrics: list[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Generates a summary of errors per dataset/method and computes a global replication score."""
    valid_metrics = [m for m in metrics if m in df_errs.columns]
    
    # 1. Plotting global distributions (aggregated across all datasets)
    melted_df = df_errs.melt(
        id_vars=["dataset", "method", "graph_idx", "variant_idx"],
        value_vars=valid_metrics,
        var_name="stat",
        value_name="error"
    )
    
    n_stats = len(valid_metrics)
    max_cols = 5
    n_cols = min(n_stats, max_cols)
    n_rows = (n_stats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, stat in enumerate(valid_metrics):
        ax = axes_flat[i]
        df_stat = melted_df[melted_df["stat"] == stat].dropna(subset=["error"])
        if df_stat.empty: continue

        present_methods = [m for m in METHODS if m in df_stat["method"].unique()]
        plot_performance_distribution(
            ax=ax, df=df_stat, x="method", y="error", hue="method",
            palette=PALETTE, order=present_methods, hue_order=present_methods
        )
        
        # Set dynamic Y limits based on tightest method
        method_stats = {}
        for m in present_methods:
            m_errors = df_stat[df_stat["method"] == m]["error"].dropna().values
            if len(m_errors) > 0:
                method_stats[m] = {"max": np.percentile(m_errors, 100), "data": m_errors}
        
        if method_stats:
            tightest_m = min(method_stats.keys(), key=lambda k: method_stats[k]["max"])
            set_dynamic_ylim(ax, method_stats[tightest_m]["data"], percentile=100, expansion=1.1)
        
        ax.set_title(stat.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_ylabel("Abs Error" if i % n_cols == 0 else "")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for j in range(i + 1, n_rows * n_cols): fig.delaxes(axes_flat[j])

    available_methods = [m for m in METHODS if m in df_errs["method"].unique()]
    legend_patches = [mpatches.Patch(color=PALETTE[m], label=m) for m in available_methods]
    fig.legend(handles=legend_patches, title="Methods", loc='upper center', 
               bbox_to_anchor=(0.5, 0.94), ncol=len(available_methods), frameon=False)
    plt.suptitle("Global Generation Quality (All Datasets)", fontsize=22, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    
    # 2. Global Replication Score
    # Mean error of each method per metric (averaged over all datasets and graphs)
    method_mean_errors = df_errs.groupby('method', observed=True)[valid_metrics].mean()
    
    # Mean error across all methods for each metric
    metric_means_across_methods = method_mean_errors.mean()
    
    # Normalize: E_{m,k} / mu_k
    # We add a small epsilon to avoid division by zero
    normalized_errors = method_mean_errors / (metric_means_across_methods + 1e-12)
    
    # Final score: mean of normalized errors across all metrics
    global_scores = normalized_errors.mean(axis=1).rename("Global_Replication_Score").sort_values()
    
    print("\n=== Global Replication Score (Lower is better) ===")
    display(global_scores.to_frame())
    
    return summary_df, global_scores

summary_df, global_scores = generate_global_summary(df_errors, plot_stats)