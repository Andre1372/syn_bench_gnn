"""Diagnostic script for log-binned degree features.

For every dataset listed in DATASETS:
  1. Loads the raw TUDataset (topology only).
  2. Computes log-bin edges with compute_log_bin_edges.
  3. Applies log-bin features and samples a few graphs.
  4. Plots a figure with four panels:
       (A) Global degree histogram + bin boundaries (vertical lines).
       (B) Fraction of total nodes per bin (bar chart).
       (C) Feature matrices (one-hot) for a handful of sampled graphs.
       (D) Summary table: bin index → degree range → node count → fraction.

Run with:
    venv_sbg/bin/python inspect_log_bins.py [--datasets MUTAG BZR ...] [--out_dir figs/]
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering – swap to "TkAgg" if you want an interactive window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import torch

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from torch_geometric.datasets import TUDataset
from src.data_utils import (
    remove_features,
    compute_log_bin_edges,
    apply_log_bin_features,
    pytorch_to_igraph,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASETS = ["MUTAG", "BZR", "DHFR", "Mutagenicity"]
DATA_ROOT = Path("data")
SAMPLE_GRAPHS = 5          # how many individual graph feature-matrices to show in panel C
LOG_BASE = 2.0
MIN_TAIL_FRAC = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_degrees(dataset) -> list[int]:
    """Return all node degrees (from edge_index) across the whole dataset."""
    degrees: list[int] = []
    for data in dataset:
        if data.edge_index is not None and data.edge_index.numel() > 0:
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
            deg.scatter_add_(0, data.edge_index[0],
                             torch.ones(data.edge_index.size(1), dtype=torch.long))
            degrees.extend(deg.tolist())
        else:
            degrees.extend([0] * data.num_nodes)
    return degrees


def bin_label(edges: list[float], i: int) -> str:
    """Human-readable label for bin i, e.g. '[4, 8)'."""
    lo = int(edges[i]) if edges[i] != float("inf") else "∞"
    hi = int(edges[i + 1]) if edges[i + 1] != float("inf") else "∞"
    if hi == "∞":
        return f"[{lo}, ∞)"
    return f"[{lo}, {hi})"


def assign_bin_indices(degrees: list[int], bin_edges: list[float]) -> list[int]:
    """Map each degree to its bin index."""
    num_bins = len(bin_edges) - 1
    upper = torch.tensor(bin_edges[1:], dtype=torch.float32)
    deg_t = torch.tensor(degrees, dtype=torch.float32)
    return torch.bucketize(deg_t, upper, right=True).clamp(0, num_bins - 1).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset_figure(dataset_name: str, out_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    raw = TUDataset(root=str(DATA_ROOT), name=dataset_name)
    stripped = [remove_features(d) for d in raw]

    # ── Compute bins ──────────────────────────────────────────────────────────
    bin_edges = compute_log_bin_edges(stripped, base=LOG_BASE, min_tail_fraction=MIN_TAIL_FRAC)
    num_bins = len(bin_edges) - 1

    degrees = collect_all_degrees(stripped)
    total_nodes = len(degrees)
    bin_indices = assign_bin_indices(degrees, bin_edges)
    bin_counts = Counter(bin_indices)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Graphs       : {len(stripped)}")
    print(f"  Total nodes  : {total_nodes}")
    print(f"  Max degree   : {max(degrees)}")
    print(f"  Num bins     : {num_bins}")
    print(f"  Bin edges    : {[e if e == float('inf') else int(e) for e in bin_edges]}")
    print()
    print(f"  {'Bin':>4}  {'Range':>12}  {'Nodes':>8}  {'Fraction':>9}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*8}  {'-'*9}")
    for b in range(num_bins):
        cnt = bin_counts.get(b, 0)
        print(f"  {b:>4}  {bin_label(bin_edges, b):>12}  {cnt:>8}  {cnt / total_nodes:>9.3%}")

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"Log-binned Degree Features — {dataset_name}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax_hist  = fig.add_subplot(gs[0, 0])   # A: degree histogram
    ax_bar   = fig.add_subplot(gs[0, 1])   # B: per-bin fractions
    ax_feat  = fig.add_subplot(gs[1, 0])   # C: sample feature matrices
    ax_table = fig.add_subplot(gs[1, 1])   # D: summary table

    palette = plt.cm.tab10.colors

    # ── Panel A: Degree histogram + bin edges ─────────────────────────────────
    max_deg = max(degrees)
    counts_hist, bin_hist_edges = np.histogram(degrees, bins=range(0, max_deg + 2))
    ax_hist.bar(bin_hist_edges[:-1], counts_hist, color="#4a90d9", alpha=0.75,
                label="Node count", width=1.0, edgecolor="white", linewidth=0.4)

    for b, edge in enumerate(bin_edges[:-1]):   # skip the last +inf
        color = palette[b % len(palette)]
        ax_hist.axvline(edge, color=color, linewidth=1.6, linestyle="--",
                        label=f"Bin {b} start = {int(edge)}" if edge > 0 else None)

    ax_hist.set_xlabel("Node degree")
    ax_hist.set_ylabel("# nodes")
    ax_hist.set_title("(A) Global degree distribution + bin edges")
    ax_hist.legend(fontsize=7, ncol=2)
    ax_hist.set_yscale("log")
    ax_hist.yaxis.set_minor_formatter(mticker.NullFormatter())

    # ── Panel B: Per-bin fraction bar ─────────────────────────────────────────
    fracs = [bin_counts.get(b, 0) / total_nodes for b in range(num_bins)]
    bar_labels = [bin_label(bin_edges, b) for b in range(num_bins)]
    x_pos = np.arange(num_bins)
    bars = ax_bar.bar(x_pos, fracs,
                      color=[palette[b % len(palette)] for b in range(num_bins)],
                      edgecolor="white", linewidth=0.6)
    ax_bar.axhline(MIN_TAIL_FRAC, color="red", linewidth=1.4, linestyle=":",
                   label=f"min_tail_fraction = {MIN_TAIL_FRAC:.0%}")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=8)
    ax_bar.set_ylabel("Fraction of total nodes")
    ax_bar.set_title("(B) Per-bin node fraction")
    ax_bar.legend(fontsize=8)
    ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # Annotate bars with exact counts
    for rect, frac, b in zip(bars, fracs, range(num_bins)):
        cnt = bin_counts.get(b, 0)
        ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.002,
                    f"{cnt}", ha="center", va="bottom", fontsize=7)

    # ── Panel C: Sample feature matrices (one-hot heatmaps) ───────────────────
    # Pick SAMPLE_GRAPHS graphs spread across the dataset
    n_graphs = len(stripped)
    sample_idx = np.linspace(0, n_graphs - 1, min(SAMPLE_GRAPHS, n_graphs), dtype=int)
    matrices = []
    labels_str = []
    for idx in sample_idx:
        data = stripped[idx]
        g = pytorch_to_igraph(data)
        x = apply_log_bin_features(g, bin_edges)   # (num_nodes, num_bins)
        matrices.append(x.numpy())
        labels_str.append(f"G[{idx}]  n={data.num_nodes}")

    # Stack with small separators
    SEP_H = 1
    SEP_VAL = 0.5
    rows = []
    for mx in matrices:
        rows.append(mx)
        rows.append(np.full((SEP_H, num_bins), SEP_VAL))
    combined = np.vstack(rows[:-1])   # drop last separator

    im = ax_feat.imshow(combined, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                        interpolation="nearest")
    # Y-tick positions at the middle of each graph block
    ytick_pos = []
    ytick_lbl = []
    cursor = 0
    for i, mx in enumerate(matrices):
        mid = cursor + mx.shape[0] / 2 - 0.5
        ytick_pos.append(mid)
        ytick_lbl.append(labels_str[i])
        cursor += mx.shape[0] + SEP_H

    ax_feat.set_yticks(ytick_pos)
    ax_feat.set_yticklabels(ytick_lbl, fontsize=8)
    ax_feat.set_xticks(range(num_bins))
    ax_feat.set_xticklabels([f"Bin {b}\n{bin_label(bin_edges, b)}" for b in range(num_bins)],
                             fontsize=7, rotation=30, ha="right")
    ax_feat.set_title(f"(C) One-hot feature matrices — {len(sample_idx)} sample graphs")
    plt.colorbar(im, ax=ax_feat, fraction=0.04, pad=0.02)

    # ── Panel D: Summary table ────────────────────────────────────────────────
    ax_table.axis("off")
    col_headers = ["Bin", "Degree range", "Nodes", "Fraction"]
    table_data = []
    for b in range(num_bins):
        cnt = bin_counts.get(b, 0)
        table_data.append([str(b), bin_label(bin_edges, b), str(cnt), f"{cnt / total_nodes:.2%}"])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)

    # Colour the header row
    for col in range(len(col_headers)):
        tbl[0, col].set_facecolor("#2c3e50")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # Colour rows by bin colour
    for b in range(num_bins):
        color = palette[b % len(palette)]
        tbl[b + 1, 0].set_facecolor(color + (0.25,))

    ax_table.set_title("(D) Bin summary", fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"logbin_{dataset_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize log-binned degree features.")
    p.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="TUDataset names to inspect.",
    )
    p.add_argument(
        "--out_dir", default="figs/logbin",
        help="Output directory for figures.",
    )
    p.add_argument(
        "--min_tail_frac", type=float, default=MIN_TAIL_FRAC,
        help="Minimum tail fraction for bin merging (default: 0.01).",
    )
    p.add_argument(
        "--base", type=float, default=LOG_BASE,
        help="Logarithm base for bin edges (default: 2.0).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override module-level constants with CLI values so helpers pick them up
    MIN_TAIL_FRAC = args.min_tail_frac
    LOG_BASE = args.base

    out_dir = Path(args.out_dir)
    for ds in args.datasets:
        try:
            make_dataset_figure(ds, out_dir)
        except Exception as exc:
            print(f"[ERROR] {ds}: {exc}", file=sys.stderr)

    print(f"\nAll figures written to: {out_dir.resolve()}")
