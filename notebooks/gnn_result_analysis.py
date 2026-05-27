"""
This script is the python replica of a notebook, so it is not meant to be run as a script.
"""

# Cell 0 - Imports
import re
import sys
from pathlib import Path

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

from notebooks.visualization_utils import add_baseline_guide, plot_performance_distribution



# Cell 1 - Global Variables & Data Loading
# Exactly ONE of the three variables should be None at a time; the other two
# must be set to a single valid string from the canonical orders below.
FIXED_METHODS:  str | None = "anndgE"        # set to None to vary methods
FIXED_SAMPLERS: str | None = "nosampler"     # set to None to vary samplers
FIXED_FEATURES: str | None = None            # set to None to vary features

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_NAMES = [
    "BZR", "DHFR", "Mutagenicity", "MUTAG", "AIDS", "PROTEINS", "Letter-low", "Letter-med"
]

# Methods
METHOD_ORDER = ["dummyNodes", "dummyEdges", "padma", "anndg", "anndgD", "anndgE", "anndgED"]
METHOD_PALETTE = {
    "original":   "#5B9BD5",
    "dummyNodes": "#DA7CF7",
    "dummyEdges": "#98C379",
    "padma":      "#F5C431",
    "anndg":      "#E06C75",
    "anndgD":     "#5CE9FF",
    "anndgE":     "#C47900",
    "anndgED":    "#5242D1",
}

# Samplers (None is stored as "nosampler" in filenames)
SAMPLER_ORDER = ["nosampler", "moments", "percentile", "percentile_corr", "gmcm"]
SAMPLER_PALETTE = {
    "nosampler":       "#5B9BD5",
    "moments":         "#DA7CF7",
    "percentile":      "#98C379",
    "percentile_corr": "#F5C431",
    "gmcm":            "#E06C75",
}

# Features
FEATURE_ORDER = ["constant", "log_bin_deg", "random_sample", "degree_ordered", "neighbor_degree_ordered"]
FEATURE_PALETTE = {
    "constant":                  "#5CE9FF",
    "log_bin_deg":               "#DA7CF7",
    "random_sample":             "#98C379",
    "degree_ordered":            "#C47900",
    "neighbor_degree_ordered":   "#FF8C69",
}

def _canonical_order(values, order):
    """Return values sorted by canonical order, with unknown values appended."""
    present = set(values)
    return [v for v in order if v in present] + sorted(v for v in present if v not in order)

def load_experiment_data() -> pd.DataFrame:
    """Loads and preprocesses main GNN evaluation data."""
    sampler_alts = "|".join(re.escape(s) for s in SAMPLER_ORDER)
    results_pattern = re.compile(
        r"gnn_global_"
        r"(?P<dname>[A-Za-z][A-Za-z0-9\-]*)_"          # dataset name
        r"(?P<mname>[A-Za-z][A-Za-z0-9]*)_"             # method name
        r"(?P<sampler>" + sampler_alts + r")_"           # sampler (exact match)
        r"(?P<feature>[a-zA-Z0-9_]+)"                    # feature strategy
        r"\.csv$"
    )

    dfs_results = []
    for csv_path in RESULTS_DIR.glob("gnn_global_*.csv"):
        match = results_pattern.match(csv_path.name)
        if match is None:
            continue
        dname   = match.group("dname")
        mname   = match.group("mname")
        sampler = match.group("sampler")
        feature = match.group("feature")

        if dname not in DATASET_NAMES:
            continue

        df = pd.read_csv(csv_path)
        df["method"]  = mname
        df["sampler"] = sampler
        df["feature"] = feature
        dfs_results.append(df)

    if not dfs_results:
        raise FileNotFoundError(f"No results found in {RESULTS_DIR} for datasets: {DATASET_NAMES}.")

    df_raw = pd.concat(dfs_results, ignore_index=True)
    for col in ["val_best_f1", "test_f1", "test_acc", "test_roc_auc"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # source_base: strip variant index suffix (_0, _1, …) from synthetic sources
    df_raw["source_base"] = df_raw["source"].apply(lambda s: s.rsplit("_", 1)[0] if s != "original" else s)

    # Apply canonical categorical order for source_base (method axis)
    all_methods = set(df_raw["source_base"]) - {"original"}
    ordered_m   = _canonical_order(all_methods, METHOD_ORDER)
    df_raw["source_base"] = pd.Categorical(df_raw["source_base"], categories=["original"] + ordered_m, ordered=True)

    return df_raw

df_raw = load_experiment_data()

# Derived globals
MODELS = sorted(df_raw["model"].unique().tolist())
DATASETS = [d for d in DATASET_NAMES if d in df_raw["dataset"].unique()]
_filtered_src = df_raw[df_raw["source_base"] != "original"].sort_values("source_base")
METHODS_PER_DATASET = _filtered_src.groupby("dataset", observed=True)["source_base"].unique().apply(list).to_dict()



# Cell 2 - Utilities functions
def _resolve_vary_axis() -> str:
    """Returns 'method', 'sampler', or 'feature' — the free (varying) axis."""
    free = [
        dim for dim, val in [
            ("method",  FIXED_METHODS),
            ("sampler", FIXED_SAMPLERS),
            ("feature", FIXED_FEATURES),
        ] if val is None
    ]
    if len(free) != 1:
        raise ValueError(f"Exactly one of FIXED_METHODS, FIXED_SAMPLERS, FIXED_FEATURES must be None. Currently None: {free}")
    return free[0]

def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filters df to rows matching the two fixed dimensions."""
    mask = pd.Series(True, index=df.index)
    if FIXED_METHODS  is not None: mask &= (df["method"]  == FIXED_METHODS)
    if FIXED_SAMPLERS is not None: mask &= (df["sampler"] == FIXED_SAMPLERS)
    if FIXED_FEATURES is not None: mask &= (df["feature"] == FIXED_FEATURES)
    return df[mask].copy()

def _vary_col_and_meta() -> tuple[str, list, dict, str]:
    """Returns (col_name, ordered_values, palette, title_suffix) for the free axis."""
    vary = _resolve_vary_axis()
    if vary == "method":
        present = set(df_raw["source_base"]) - {"original"}
        vals    = _canonical_order(present, METHOD_ORDER)
        pal     = METHOD_PALETTE
        suffix  = f"Fixed Sampler={FIXED_SAMPLERS}, Feature={FIXED_FEATURES}"
        col     = "source_base"
    elif vary == "sampler":
        present = set(df_raw["sampler"].unique())
        vals    = _canonical_order(present, SAMPLER_ORDER)
        pal     = SAMPLER_PALETTE
        suffix  = f"Fixed Method={FIXED_METHODS}, Feature={FIXED_FEATURES}"
        col     = "sampler"
    else:  # vary == "feature"
        present = set(df_raw["feature"].unique())
        vals    = _canonical_order(present, FEATURE_ORDER)
        pal     = FEATURE_PALETTE
        suffix  = f"Fixed Method={FIXED_METHODS}, Sampler={FIXED_SAMPLERS}"
        col     = "feature"
    return col, vals, pal, suffix



# Cell 3 - Per-Dataset Validation (Trajectories & Detailed Statistics)
def plot_performance_trajectory(df: pd.DataFrame) -> None:
    """Plots GNN performance trajectories across all datasets.

    The free axis (determined by which FIXED_* is None) becomes the row
    dimension.  Each dataset × model pair is one column.
    """
    vary_col, row_values, row_pal, title_suffix = _vary_col_and_meta()
    df_f = _filter_df(df)

    datasets = [d for d in DATASETS if d in df_f["dataset"].unique()]
    if not datasets or not row_values:
        print("No data to plot.")
        return

    n_models = len(MODELS)
    n_rows   = len(row_values)
    n_cols   = n_models * len(datasets)

    fig = plt.figure(figsize=(5.5 * n_cols, 4.4 * n_rows))
    gs  = fig.add_gridspec(n_rows, n_cols)
    fig.suptitle(f"Performance Trajectories ({title_suffix})", fontsize=16, fontweight="bold", y=1.02)

    for row_idx, row_val in enumerate(row_values):
        row_color = row_pal.get(row_val, "#AAAAAA")
        row_label = str(row_val).replace("_", " ").upper()

        # For the method axis the variant sources live in "source"; for others
        # we filter by the varying column value.
        for ds_idx, dataset in enumerate(datasets):
            df_ds = df_f[df_f["dataset"] == dataset].copy()

            if vary_col == "source_base":
                # method is varying: list synthetic variants for this method
                method_name = row_val
                variant_sources = sorted(
                    [s for s in df_ds["source"].unique() if s.startswith(f"{method_name}_")],
                    key=lambda s: int(s.rsplit("_", 1)[-1])
                )
            else:
                # sampler or feature is varying: all synthetic sources for the fixed method
                method_name = FIXED_METHODS
                variant_sources = sorted(
                    [s for s in df_ds["source"].unique()
                     if s != "original" and s.startswith(f"{method_name}_")],
                    key=lambda s: int(s.rsplit("_", 1)[-1])
                )

            x_pos = list(range(len(variant_sources)))
            v_pal = {s: row_color for s in variant_sources}

            for model_idx, model in enumerate(MODELS):
                col_idx = ds_idx * n_models + model_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Original baseline
                base_f1 = df_ds.loc[(df_ds["source"] == "original") & (df_ds["model"] == model), "test_f1"]
                if not base_f1.empty:
                    add_baseline_guide(ax, base_f1, color=METHOD_PALETTE["original"], label="Original")

                # Trajectory
                if variant_sources:
                    mask = (
                        (df_ds["source"].isin(variant_sources)) &
                        (df_ds["model"] == model) &
                        (df_ds[vary_col] == row_val) if vary_col != "source_base"
                        else (df_ds["source"].isin(variant_sources)) & (df_ds["model"] == model)
                    )
                    plot_df = df_ds.loc[mask].copy()
                    if not plot_df.empty:
                        plot_df["_xpos"] = plot_df["source"].map({s: p for s, p in zip(variant_sources, x_pos)})
                        plot_performance_distribution(
                            ax=ax, df=plot_df, x="_xpos", y="test_f1",
                            hue="source", palette=v_pal, order=x_pos
                        )

                if row_idx == 0:
                    ax.set_title(f"{dataset}\n{model}", fontsize=11, fontweight="bold", pad=8)
                if ds_idx == 0 and model_idx == 0:
                    ax.set_ylabel(row_label, fontsize=11, fontweight="bold")
                else:
                    ax.set_ylabel("")

                if variant_sources:
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([s.rsplit("_", 1)[-1] for s in variant_sources], fontsize=8)
                ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=METHOD_PALETTE["original"], alpha=0.4, label="Original Baseline")]
    handles += [
        mpatches.Patch(color=row_pal.get(v, "#AAAAAA"), alpha=0.8, label=str(v).replace("_", " ").upper())
        for v in row_values
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=1 + len(handles), frameon=True, shadow=True, fontsize=10)
    fig.tight_layout()
    plt.show()

plot_performance_trajectory(df_raw)



# Cell 4 - Aggregated Overview Summaries
def compute_delta_df(df: pd.DataFrame) -> pd.DataFrame:
    baseline  = df[df["source_base"] == "original"].groupby(["dataset", "model"])["test_f1"].mean()
    df_synth  = df[df["source_base"] != "original"].copy()
    df_synth["delta_test_f1"] = df_synth.apply(lambda row: abs(row["test_f1"] - baseline.loc[(row["dataset"], row["model"])]), axis=1)
    return df_synth

df_synth = compute_delta_df(df_raw)


def plot_performance_overview(df: pd.DataFrame, df_synth: pd.DataFrame) -> None:
    vary_col, categories_all, pal, title_suffix = _vary_col_and_meta()
    df_f       = _filter_df(df)
    df_synth_f = _filter_df(df_synth)

    n_ds = len(DATASETS)
    fig, axes = plt.subplots(
        nrows=n_ds, ncols=5, figsize=(20, 5 * n_ds + 1.0),
        sharex=False, sharey=False, squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 0.2, 1, 1]}
    )
    fig.suptitle(f"GNN Performance — Aggregated Overview ({title_suffix})", fontsize=16, fontweight="bold", y=1.02)

    for r_idx, dataset in enumerate(DATASETS):
        df_ds       = df_f[df_f["dataset"] == dataset]
        df_synth_ds = df_synth_f[df_synth_f["dataset"] == dataset]
        axes[r_idx, 2].set_visible(False)

        present    = set(df_synth_ds[vary_col].dropna().astype(str).unique())
        categories = [v for v in categories_all if v in present]
        palette_map = pal

        for c_idx, model in enumerate(MODELS):
            ax_box  = axes[r_idx, c_idx]
            base_f1 = df_ds.loc[(df_ds["source"] == "original") & (df_ds["model"] == model), "test_f1"]
            if not base_f1.empty:
                add_baseline_guide(ax_box, base_f1, color=METHOD_PALETTE["original"], label="Original")

            plot_df = df_ds[df_ds[vary_col].isin(categories) & (df_ds["model"] == model)].copy()
            if not plot_df.empty:
                plot_performance_distribution(
                    ax_box, plot_df, x=vary_col, y="test_f1",
                    hue=vary_col, palette=palette_map, order=categories
                )

            ax_box.set_xlabel("")
            ax_box.set_ylabel("")
            ax_box.set_xticks(range(len(categories)))
            ax_box.set_xticklabels([str(c).replace("_", " ").upper() for c in categories], fontsize=10)
            ax_box.grid(axis="y", linestyle="--", alpha=0.4)
            if c_idx == 1: ax_box.tick_params(labelleft=False)

            if r_idx == 0:
                ax_box.set_title(f"F1-Score: {model}", fontsize=12, fontweight="bold", pad=8)
                if c_idx == 0:
                    handles = (
                        [mpatches.Patch(color=METHOD_PALETTE["original"], alpha=0.4, label="Original Baseline")]
                        + [mpatches.Patch(color=palette_map.get(c, "#AAAAAA"), alpha=0.8,
                                          label=str(c).replace("_", " ").upper()) for c in categories]
                    )
                    leg = ax_box.legend(
                        handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.1),
                        ncol=1 + len(categories), frameon=True, shadow=True, fontsize=10
                    )
                    leg.set_in_layout(False)
            if c_idx == 0:
                ax_box.text(-0.25, 0.5, dataset, transform=ax_box.transAxes, fontsize=12, fontweight="bold", va="center", ha="center", rotation=90)

        delta_agg = df_synth_ds.groupby([vary_col, "model"], observed=True)["delta_test_f1"].agg(["mean", "std"])
        for c_idx, model in enumerate(MODELS):
            ax_bar = axes[r_idx, 3 + c_idx]
            bw     = 0.6 / max(len(categories), 1)

            for m_idx, cat in enumerate(categories):
                offset = (m_idx - (len(categories) - 1) / 2) * bw
                try:
                    m_val = float(delta_agg.loc[(cat, model), "mean"])
                    s_val = float(delta_agg.loc[(cat, model), "std"])
                except (KeyError, TypeError):
                    m_val, s_val = 0.0, 0.0

                ax_bar.bar(offset, m_val, width=bw, color=palette_map.get(cat, "#AAAAAA"), zorder=2)
                ax_bar.errorbar(
                    offset, m_val, yerr=s_val, fmt="none",
                    color="#646464", capsize=4, capthick=1.4, linewidth=1.4, zorder=3
                )
                run_deltas = df_synth_ds.loc[
                    (df_synth_ds[vary_col] == cat) & (df_synth_ds["model"] == model),
                    "delta_test_f1"
                ].to_numpy(dtype=np.float64)
                jitter = np.random.default_rng(42).uniform(-bw * 0.22, bw * 0.22, size=len(run_deltas))
                ax_bar.scatter(
                    offset + jitter, run_deltas,
                    color="black", alpha=0.55, s=22, zorder=4, linewidths=0
                )

            ax_bar.set_xticks([])
            ax_bar.set_ylim(bottom=0.0)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
            if c_idx == 1:
                ax_bar.tick_params(labelleft=False)

            if r_idx == 0:
                ax_bar.set_title(f"|ΔF1|: {model}", fontsize=12, fontweight="bold", pad=8)
                if c_idx == 0:
                    handles = [
                        mpatches.Patch(color=palette_map.get(c, "#AAAAAA"), alpha=0.8,
                                       label=str(c).replace("_", " ").upper())
                        for c in categories
                    ]
                    leg = ax_bar.legend(
                        handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.1),
                        ncol=len(categories), frameon=True, shadow=True, fontsize=10
                    )
                    leg.set_in_layout(False)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()

def _compute_aggregated_summary_df(df: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    vary_col, categories_all, _, _ = _vary_col_and_meta()
    df_f       = _filter_df(df)
    df_synth_f = _filter_df(df_synth)

    f1_agg    = df_f.groupby(["dataset", "source_base", "feature", "model"], observed=True)["test_f1"].agg(["mean", "std"])
    delta_agg = df_synth_f.groupby(["dataset", vary_col, "model"], observed=True)["delta_test_f1"].agg(["mean", "std"])

    c_tuples = [(m_lbl, m) for m_lbl in ["F1-Score", "|Δ F1|"] for m in MODELS]
    rows, r_tuples = [], []

    for ds in DATASETS:
        df_ds       = df_f[df_f["dataset"] == ds]
        df_synth_ds = df_synth_f[df_synth_f["dataset"] == ds]

        present    = set(df_synth_ds[vary_col].dropna().astype(str).unique())
        categories = ["original"] + [v for v in categories_all if v in present]

        for cat in categories:
            r_tuples.append((ds, str(cat).replace("_", " ").upper()))
            row = []

            for model in MODELS:
                if cat == "original":
                    base = df_ds.loc[(df_ds["source"] == "original") & (df_ds["model"] == model), "test_f1"]
                    row.append(f"{base.mean():.3f} ± {base.std():.3f}" if not base.empty else "N/A")
                else:
                    try:
                        src_base = cat if vary_col == "source_base" else FIXED_METHODS
                        feat_val = cat if vary_col == "feature" else FIXED_FEATURES
                        idx = (ds, src_base, feat_val, model)
                        row.append(f"{f1_agg.loc[idx, 'mean']:.3f} ± {f1_agg.loc[idx, 'std']:.3f}")
                    except KeyError:
                        row.append("N/A")

            for model in MODELS:
                if cat == "original":
                    row.append("-")
                else:
                    try:
                        idx = (ds, cat, model)
                        row.append(f"{delta_agg.loc[idx, 'mean']:.3f} ± {delta_agg.loc[idx, 'std']:.3f}")
                    except KeyError:
                        row.append("N/A")

            rows.append(row)

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(r_tuples, names=["Dataset", "Category"]),
        columns=pd.MultiIndex.from_tuples(c_tuples, names=["Metric", "Model"])
    )

def display_aggregated_summary(df: pd.DataFrame, df_synth: pd.DataFrame) -> None:
    df_res = _compute_aggregated_summary_df(df, df_synth)
    if df_res.empty:
        return

    def _styler(styler):
        c_tuples = df_res.columns.tolist()
        categories = [c[0] for c in c_tuples]
        for i in range(1, len(categories)):
            if categories[i] != categories[i - 1]:
                styler.set_table_styles([{"selector": f".col{i}", "props": [("border-left", "2px solid #969696")]}], overwrite=False)
        idx_lvl = df_res.index.get_level_values(0)
        for i in range(1, len(df_res)):
            if idx_lvl[i] != idx_lvl[i - 1]:
                styler.set_table_styles([{"selector": f"tbody tr:nth-child({i+1})", "props": [("border-top", "2px solid #969696")]}], overwrite=False)
        styler.set_table_styles([{"selector": "th", "props": [("text-align", "center"), ("font-size", "10px")]}, {"selector": "td", "props": [("text-align", "center"), ("font-size", "10px")]}], overwrite=False)
        methods = df_res.index.get_level_values(1)
        styler.apply(lambda c: ["font-weight: bold" if methods[i] == "ORIGINAL" else "" for i in range(len(c))], axis=0)
        return styler

    display(_styler(df_res.style))

plot_performance_overview(df_raw, df_synth)
display_aggregated_summary(df_raw, df_synth)



# Cell 5 - Final aggregated line plot (|ΔF1| trend)
def plot_final_results_comparison(df_synth: pd.DataFrame) -> None:
    """Plots the aggregated GNN classification performance gap across all datasets."""
    vary_col, categories, pal, title_suffix = _vary_col_and_meta()
    df_perf = _filter_df(df_synth)

    if df_perf.empty:
        print(f"No performance data found matching current FIXED_* settings.")
        return

    df_perf = df_perf.copy()
    df_perf[vary_col] = df_perf[vary_col].astype(str)
    present = set(df_perf[vary_col].unique())
    ordered_cats = [v for v in categories if v in present]
    df_perf[vary_col] = pd.Categorical(df_perf[vary_col], categories=ordered_cats, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Global Summary: GNN Performance Gap ({title_suffix})", fontsize=15, fontweight="bold", y=0.98)

    sns.lineplot(
        data=df_perf,
        x=vary_col,
        y="delta_test_f1",
        hue="model",
        marker="o",
        markersize=8,
        linewidth=2.5,
        ax=ax,
        errorbar=None
    )

    ax.set_title("Aggregated GNN Performance Gap (|ΔF1|)", fontsize=12, fontweight="bold")
    ax.set_xlabel(vary_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean |ΔF1| (across all datasets)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="GNN Model", frameon=True, shadow=True)

    ax.set_xticks(range(len(ordered_cats)))
    ax.set_xticklabels([str(c).replace("_", " ").upper() for c in ordered_cats], fontsize=10)

    plt.tight_layout()
    plt.show()

plot_final_results_comparison(df_synth)