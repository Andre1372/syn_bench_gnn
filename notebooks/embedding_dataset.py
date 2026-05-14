"""
This script is the python replica of a notebook, so it is not meant to be run as a script.
"""

# Cell 0 - Imports
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
import seaborn as sns
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import DatasetPT
from src.enc_dec_dataset import MomentsEncoderDecoder, PercentileEncoderDecoder, GMCMEncoderDecoder



# Cell 1 - Global Variables & Data Loading
DATASET_NAMES = ["BZR", "DHFR", "Mutagenicity", "MUTAG"]
DISCRETE_FEATURES = ["n_nodes", "n_edges"]
CONTINUOUS_FEATURES = [
    "degree_moments_0", "degree_moments_1", "degree_moments_2", "degree_moments_3", 
    "annd_0", "annd_1", "annd_2", "annd_3",
    "eccentricity_0", "eccentricity_1", "eccentricity_2", "eccentricity_3"
]
FEATURES = DISCRETE_FEATURES + CONTINUOUS_FEATURES
IS_DISCRETE = np.array([feat in DISCRETE_FEATURES for feat in FEATURES])
RNG = np.random.default_rng(seed=42)

def load_generation_data() -> pd.DataFrame:
    """Load and preprocess all original datasets and synthetic variants saving all per-graph statistics."""
    rows = []
    original_stats_map = {}

    for dataset_name in DATASET_NAMES:
        # Load original
        orig_pt_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_original.pt"
        if orig_pt_path.exists():
            dataset_obj = DatasetPT(orig_pt_path)
            per_graph_stats = dataset_obj.metadata.get("per_graph_statistics", [])

            for i, stats in enumerate(per_graph_stats):                
                row = {
                    "dataset": dataset_name,
                    "graph_idx": i,
                    "class_id": dataset_obj[i].y.item(),
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
                        
    df = pd.DataFrame(rows)
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype("Int64")
    return df

# Load data
df = load_generation_data()
display(df.head(10))

# Initialize Encoder/Decoder with the correct number of classes and discrete info
num_classes = int(df["class_id"].max() + 1)
# ENC_DEC = MomentsEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, k=4, rng=RNG)
# ENC_DEC = PercentileEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, percentile_size=0.1, replicate_correlation=True, rng=RNG)
ENC_DEC = GMCMEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, percentile_size=0.1, n_components=3, rng=RNG)

DATASETS = [d for d in DATASET_NAMES if d in df["dataset"].unique()]



# Cell 2 - Encode and Sample dataset
def encode_and_sample_dataset(df_dataset: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encodes original statistics and samples synthetic ones for all classes in a dataset."""
    embeddings_rows = []
    samples_list = []
    
    for class_id, group in df_dataset.groupby("class_id"):
        # --- Encode ---
        stat_matrix = group[FEATURES].values
        ENC_DEC.encode_features(stat_matrix, class_id)

        # Save encodings for this class
        emb_row = {"class_id": class_id}
        encoding = ENC_DEC._encodings[class_id]
        for i, feat in enumerate(FEATURES):
            emb_row[feat] = encoding[:, i]
        embeddings_rows.append(emb_row)

        # --- Sample ---
        n_samples = len(group)
        samples_matrix = ENC_DEC.sample_features(n_samples, class_id)
    
        # Save samples for this class
        class_samples_df = pd.DataFrame(samples_matrix, columns=FEATURES)
        class_samples_df["class_id"] = class_id
        class_samples_df[DISCRETE_FEATURES] = class_samples_df[DISCRETE_FEATURES].astype(int)
        samples_list.append(class_samples_df)
        
    # Construct final DataFrames
    df_emb = pd.DataFrame(embeddings_rows)
    df_samples = pd.concat(samples_list, ignore_index=True)
    
    df_emb.insert(0, "dataset", dataset_name)
    df_samples.insert(0, "dataset", dataset_name)
    
    return df_emb, df_samples

dataset_embeddings = []
dataset_samples = []

for dataset_name in DATASETS:
    df_subset = df[df["dataset"] == dataset_name]
    df_emb, df_s = encode_and_sample_dataset(df_subset, dataset_name)
    
    dataset_embeddings.append(df_emb)
    dataset_samples.append(df_s)
    
    print(f"\n--- {dataset_name} ---")
    # Get embedding size for the first class (all classes have same embedding size)
    emb_size = ENC_DEC.get_embedding(df_subset["class_id"].iloc[0]).shape[0]
    print(f"Embedding vector dimension: {emb_size}")
    display(df_emb.head())
    display(df_s.head())

df_embeddings = pd.concat(dataset_embeddings, ignore_index=True)
df_synthetic = pd.concat(dataset_samples, ignore_index=True)



# Cell 3 - Verify sampling quality
def plot_dataset_comparison(df_orig: pd.DataFrame, df_synth: pd.DataFrame, dataset_name: str) -> None:
    """Plots distributions of all metrics for original vs synthetic datasets."""
    n_rows = 4  # 1 row for discrete + rows for continuous
    
    fig = plt.figure(figsize=(24, 5 * n_rows))
    fig.suptitle(f"Metric Comparison: {dataset_name} (Original vs Synthetic)", fontsize=22, fontweight="bold", y=0.92)
    gs = fig.add_gridspec(n_rows, 4, hspace=0.4, wspace=0.3)
    
    # Define mapping of feature to subplot
    metrics_map = []
    # Row 0: Discrete features (spanning 2 columns each)
    metrics_map.append((DISCRETE_FEATURES[0], fig.add_subplot(gs[0, :2])))
    metrics_map.append((DISCRETE_FEATURES[1], fig.add_subplot(gs[0, 2:])))    
    # Rows 1+: Continuous features (4 per row)
    for i, feat in enumerate(CONTINUOUS_FEATURES):
        row, col = divmod(i, 4)
        metrics_map.append((feat, fig.add_subplot(gs[row + 1, col])))
        
    for feat, ax in metrics_map:
        if feat not in df_orig.columns:
            ax.axis("off")
            continue
            
        # Plot distributions
        for data, label, color in [(df_orig, "Original", "#5B9BD5"), (df_synth, "Synthetic", "#F5C431")]:
            sns.histplot(
                data=data, x=feat, ax=ax, label=label,
                color=color, alpha=0.45, stat="density", element="bars",
                linewidth=0.5, edgecolor="white"
            )
        
        ax.set_title(feat.replace("_", " ").title(), fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=12, alpha=0.7)
        ax.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax)
        
    plt.show()

def compute_wassestrain_distance(df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
    """Computes the Wasserstein distance for each feature using anchored Min-Max scaling.
    
    Args:
        df_orig: Original dataset DataFrame
        df_synth: Synthetic dataset DataFrame
    Returns:
        Dictionary mapping feature names to their Wasserstein distance
    """
    results = {}
    for feat in FEATURES:
        if feat not in df_orig.columns or feat not in df_synth.columns:
            raise ValueError(f"Feature {feat} not found in both datasets.")
            
        orig_values = df_orig[feat].values
        synth_values = df_synth[feat].values
        
        # 1. Anchored Min-Max Scaling
        min_val = np.min(orig_values)
        max_val = np.max(orig_values)
        range_val = max_val - min_val
        
        if range_val > 0:
            orig_scaled = (orig_values - min_val) / range_val
            synth_scaled = (synth_values - min_val) / range_val
        else:
            orig_scaled = orig_values - min_val
            synth_scaled = synth_values - min_val
            
        # 2. Wasserstein Distance calculation
        dist = wasserstein_distance(orig_scaled, synth_scaled)
        results[feat] = dist
        
    return results


all_distances_results = []

for dataset_name in DATASETS:
    df_orig = df[df["dataset"] == dataset_name]
    df_synth = df_synthetic[df_synthetic["dataset"] == dataset_name]
    
    # Visualization
    plot_dataset_comparison(df_orig, df_synth, dataset_name)
    
    # Statistical Evaluation
    distances = compute_wassestrain_distance(df_orig, df_synth)
    global_score = np.mean(list(distances.values()))
    
    # Store results
    res_row = {"Dataset": dataset_name}
    res_row.update(distances)
    res_row["GLOBAL_SCORE"] = global_score
    all_distances_results.append(res_row)

# Final Summary Aggregation
print("\n" + "="*100)
print("FINAL CONSOLIDATED SUMMARY (Mean Wasserstein Distances across all datasets)")
print("="*100)

df_summary = pd.DataFrame(all_distances_results).set_index("Dataset")
# Add a "MEAN" row to aggregate results across all datasets
df_summary.loc["MEAN"] = df_summary.mean()

display(df_summary)

final_avg_global = df_summary.loc["MEAN", "GLOBAL_SCORE"]
print(f"\n>>> FINAL PERFORMANCE (Average Global Score): {final_avg_global:.4f} <<<")



# Cell 4 - Verify correlation between metrics
def plot_correlation_matrix(df_orig: pd.DataFrame, df_synth: pd.DataFrame, dataset_name: str) -> list[np.ndarray]:
    """Plots correlation heatmaps between original and synthetic metrics for each class.
    
    Args:
        df_orig: Original dataset DataFrame
        df_synth: Synthetic dataset DataFrame
        dataset_name: Name of the dataset
    Returns:
        List of correlation difference matrices (one per class)
    """
    classes = sorted(df_orig["class_id"].unique())
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_classes, 3, figsize=(20, 6 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, 3)
        
    fig.suptitle(f"Feature Correlations: {dataset_name} (Original vs Synthetic)", fontsize=22, fontweight="bold", y=0.99)
    
    diff_matrices = []
    
    for i, class_id in enumerate(classes):
        # Filter by class
        orig_cls = df_orig[df_orig["class_id"] == class_id][FEATURES]
        synth_cls = df_synth[df_synth["class_id"] == class_id][FEATURES]
        
        # Calculate Pearson correlations
        corr_orig = orig_cls.corr().fillna(0)
        corr_synth = synth_cls.corr().fillna(0)
        corr_diff = corr_synth - corr_orig
        
        diff_matrices.append(corr_diff.values)
        
        # Heatmap styling
        heatmap_kwargs = {
            "center": 0,
            "annot": False,
            "square": True,
            "cbar_kws": {"shrink": 0.8}
        }
        
        # 1. Original Matrix
        sns.heatmap(corr_orig, ax=axes[i, 0], cmap="coolwarm", vmin=-1, vmax=1, **heatmap_kwargs)
        axes[i, 0].set_title(f"Class {class_id} - Original", fontsize=15, fontweight="bold")
        
        # 2. Synthetic Matrix
        sns.heatmap(corr_synth, ax=axes[i, 1], cmap="coolwarm", vmin=-1, vmax=1, **heatmap_kwargs)
        axes[i, 1].set_title(f"Class {class_id} - Synthetic", fontsize=15, fontweight="bold")
        
        # 3. Difference Matrix (Synthetic - Original)
        # Use a different colormap to highlight discrepancies
        sns.heatmap(corr_diff, ax=axes[i, 2], cmap="PiYG", vmin=-0.5, vmax=0.5, **heatmap_kwargs)
        axes[i, 2].set_title(f"Class {class_id} - Difference (S - O)", fontsize=15, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    return diff_matrices


all_corr_errors = []

for dataset_name in DATASETS:
    df_orig = df[df["dataset"] == dataset_name]
    df_synth = df_synthetic[df_synthetic["dataset"] == dataset_name]
    
    # Correlation Analysis
    diff_mats = plot_correlation_matrix(df_orig, df_synth, dataset_name)
    
    # Compute global correlation error
    avg_corr_err = np.mean([np.abs(m).mean() for m in diff_mats])
    all_corr_errors.append({"Dataset": dataset_name, "Correlation_MAE": avg_corr_err})

# Final Correlation Summary Table
print("\n" + "="*100)
print("FINAL CORRELATION FIDELITY SUMMARY (Mean Absolute Error)")
print("="*100)
df_corr_summary = pd.DataFrame(all_corr_errors).set_index("Dataset")
df_corr_summary.loc["MEAN"] = df_corr_summary.mean()

display(df_corr_summary)

final_avg_corr = df_corr_summary.loc["MEAN", "Correlation_MAE"]
print(f"\n>>> FINAL CORRELATION PERFORMANCE (Average MAE): {final_avg_corr:.4f} <<<")



# Cell 5 - Adversarial Discriminator Check
def run_adversarial_check(df_orig: pd.DataFrame, df_synth: pd.DataFrame, dataset_name: str, ax=None) -> dict:
    """Trains a discriminator to distinguish between real and synthetic data."""
    
    # Assign label 0 to original, 1 to synthetic
    df_orig_eval = df_orig[FEATURES].copy()
    df_orig_eval["is_synthetic"] = 0
    df_synth_eval = df_synth[FEATURES].copy()
    df_synth_eval["is_synthetic"] = 1
    
    # Combine datasets
    df_eval = pd.concat([df_orig_eval, df_synth_eval], ignore_index=True)
    X = df_eval[FEATURES]
    y = df_eval["is_synthetic"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train Discriminator
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    acc_score = accuracy_score(y_test, y_pred)
    
    # Extract Feature Importance
    importances = pd.Series(classifier.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    # Plotting Feature Importance if an axis is provided
    if ax is not None:
        sns.barplot(x=importances.values[:10], y=importances.index[:10], ax=ax, palette="viridis", hue=importances.index[:10])
        ax.set_title(f"{dataset_name} Discriminator\nAUC: {auc_score:.3f} (Target: 0.5)", fontweight="bold")
        ax.set_xlabel("Feature Importance")
        sns.despine(ax=ax)
        
    return {
        "Dataset": dataset_name,
        "Discriminator_AUC": auc_score,
        "Discriminator_Accuracy": acc_score,
        "Top_Giveaway_Feature": importances.index[0]
    }

all_adversarial_results = []
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
if len(DATASETS) == 1: axes = [axes]

for i, dataset_name in enumerate(DATASETS):
    df_o = df[df["dataset"] == dataset_name]
    df_s = df_synthetic[df_synthetic["dataset"] == dataset_name]
    
    res = run_adversarial_check(df_o, df_s, dataset_name, ax=axes[i])
    all_adversarial_results.append(res)

plt.tight_layout()
plt.show()

# Final Summary Table
df_adv_summary = pd.DataFrame(all_adversarial_results).set_index("Dataset")
df_adv_summary.loc["MEAN"] = df_adv_summary[["Discriminator_AUC", "Discriminator_Accuracy"]].mean()

display(df_adv_summary)

final_avg_auc = df_adv_summary.loc["MEAN", "Discriminator_AUC"]
print(f"\n>>> FINAL DISCRIMINATOR AUC: {final_avg_auc:.4f} (Closer to 0.5 is better) <<<")



# Cell 6 - Encoder/Decoder Benchmark Comparison
N_REPEATS = 5

def _make_encoder_decoder_configs(num_classes: int) -> list[tuple[str, "FeatureEncoderDecoder"]]:
    """Instantiates all encoder/decoder variants to benchmark."""
    rng = np.random.default_rng(seed=0)
    return [
        ("Moments", MomentsEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, k=4, rng=rng)),
        ("Percentile(no_corr)",PercentileEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, percentile_size=0.1, replicate_correlation=False, rng=rng)),
        ("Percentile(corr)",PercentileEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, percentile_size=0.1, replicate_correlation=True, rng=rng)),
        ("GMCM", GMCMEncoderDecoder(num_classes=num_classes, is_discrete=IS_DISCRETE, percentile_size=0.1, n_components=10, rng=rng)),
    ]


def _compute_mean_wasserstein_per_class(df_orig_cls: pd.DataFrame, df_synth_cls: pd.DataFrame) -> float:
    """Mean Wasserstein distance across all marginal distributions for a single class.

    Applies anchored Min-Max scaling (from the original distribution) so that
    all features contribute on a comparable scale.

    Args:
        df_orig_cls: Original statistics DataFrame (single class).
        df_synth_cls: Synthetic statistics DataFrame (single class).
    Returns:
        Mean Wasserstein distance across all features.
    """
    distances = []
    for feat in FEATURES:
        orig_vals = df_orig_cls[feat].values.astype(float)
        synth_vals = df_synth_cls[feat].values.astype(float)
        min_val, max_val = orig_vals.min(), orig_vals.max()
        range_val = max_val - min_val
        if range_val > 0:
            orig_scaled = (orig_vals - min_val) / range_val
            synth_scaled = (synth_vals - min_val) / range_val
        else:
            orig_scaled = orig_vals - min_val
            synth_scaled = synth_vals - min_val
        distances.append(wasserstein_distance(orig_scaled, synth_scaled))
    return float(np.mean(distances))


def _compute_correlation_mae_per_class(df_orig_cls: pd.DataFrame, df_synth_cls: pd.DataFrame) -> float:
    """Mean absolute error on the Pearson correlation matrix for a single class.

    Computes the global correlation matrix for original and synthetic samples,
    takes the element-wise absolute difference, and returns its mean.

    Args:
        df_orig_cls: Original statistics DataFrame (single class).
        df_synth_cls: Synthetic statistics DataFrame (single class).
    Returns:
        Mean absolute element-wise correlation error.
    """
    corr_orig = df_orig_cls[FEATURES].corr().fillna(0).values
    corr_synth = df_synth_cls[FEATURES].corr().fillna(0).values
    return float(np.abs(corr_synth - corr_orig).mean())


def _compute_discriminator_auc_per_class(df_orig_cls: pd.DataFrame, df_synth_cls: pd.DataFrame) -> float:
    """AUC-ROC of a Random Forest discriminator (original vs synthetic) for a single class.

    Trains the classifier on 70 % of pooled data and evaluates on the remaining
    30 %. Returns NaN when samples are insufficient for a stratified split.

    Args:
        df_orig_cls: Original statistics DataFrame (single class).
        df_synth_cls: Synthetic statistics DataFrame (single class).
    Returns:
        AUC-ROC score (closer to 0.5 means synthetic and original are indistinguishable).
    """
    df_o = df_orig_cls[FEATURES].copy()
    df_o["label"] = 0
    df_s = df_synth_cls[FEATURES].copy()
    df_s["label"] = 1
    df_combined = pd.concat([df_o, df_s], ignore_index=True)
    X = df_combined[FEATURES].values
    y = df_combined["label"].values

    if len(np.unique(y)) < 2 or len(y) < 6:
        return float("nan")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    if len(np.unique(y_test)) < 2:
        return float("nan")

    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def _run_single_repeat(enc_dec: "FeatureEncoderDecoder", df_dataset: pd.DataFrame, class_id: int) -> dict:
    """One encode-decode cycle for a (enc_dec, class_id) pair; returns raw metrics.

    Args:
        enc_dec: Encoder/decoder instance to evaluate.
        df_dataset: Full dataset DataFrame for one dataset name.
        class_id: The class ID to encode and then sample from.
    Returns:
        Dict with keys 'wasserstein', 'corr_mae', 'auc_roc'.
    """
    df_cls = df_dataset[df_dataset["class_id"] == class_id]
    stat_matrix = df_cls[FEATURES].values

    enc_dec.encode_features(stat_matrix, int(class_id))

    samples_matrix = enc_dec.sample_features(len(df_cls), int(class_id))
    df_synth_cls = pd.DataFrame(samples_matrix, columns=FEATURES)
    df_synth_cls[DISCRETE_FEATURES] = df_synth_cls[DISCRETE_FEATURES].astype(int)

    return {
        "wasserstein": _compute_mean_wasserstein_per_class(df_cls, df_synth_cls),
        "corr_mae": _compute_correlation_mae_per_class(df_cls, df_synth_cls),
        "auc_roc": _compute_discriminator_auc_per_class(df_cls, df_synth_cls),
    }


def run_benchmark(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Full benchmark over all encoder/decoder classes, datasets, and class IDs."""
    num_classes = int(df["class_id"].max() + 1)
    rows = []

    # Prepare all triplets for progress bar
    tasks = []
    for dataset_name in datasets:
        df_dataset = df[df["dataset"] == dataset_name]
        class_ids = sorted(df_dataset["class_id"].unique())
        enc_dec_configs = _make_encoder_decoder_configs(num_classes)
        for enc_dec_label, enc_dec in enc_dec_configs:
            for class_id in class_ids:
                tasks.append((dataset_name, enc_dec_label, enc_dec, class_id))

    for dataset_name, enc_dec_label, enc_dec, class_id in tqdm(tasks, desc="Benchmarking Encoders"):
        df_dataset = df[df["dataset"] == dataset_name].copy()
        repeat_metrics: list[dict] = []
        for rep in range(N_REPEATS):
            try:
                metrics = _run_single_repeat(enc_dec, df_dataset, class_id)
                repeat_metrics.append(metrics)
            except Exception as exc:
                print(
                    f"  [WARN] {dataset_name} | {enc_dec_label} | class={class_id} | rep={rep}: {exc}"
                )

        if not repeat_metrics:
            continue

        mean_wasserstein = float(np.mean([m["wasserstein"] for m in repeat_metrics]))
        mean_corr_mae = float(np.mean([m["corr_mae"] for m in repeat_metrics]))
        valid_aucs = [m["auc_roc"] for m in repeat_metrics if not np.isnan(m["auc_roc"])]
        mean_auc_roc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        rows.append(
            {
                "dataset_name": dataset_name,
                "enc_dec": enc_dec_label,
                "class_id": int(class_id),
                "mean_wasserstein": mean_wasserstein,
                "mean_corr_mae": mean_corr_mae,
                "mean_auc_roc": mean_auc_roc,
            }
        )

    return pd.DataFrame(rows)


df_benchmark = run_benchmark(df, DATASETS)

# --- Full triplet-level results ---
print("\n" + "=" * 100)
print("FULL BENCHMARK RESULTS  (dataset_name, enc_dec, class_id)")
print("=" * 100)
display(df_benchmark)

# --- Aggregated summary by encoder/decoder ---
num_classes = int(df["class_id"].max() + 1)
model_order = [cfg[0] for cfg in _make_encoder_decoder_configs(num_classes)]

df_bench_agg = (
    df_benchmark
    .groupby("enc_dec")[["mean_wasserstein", "mean_corr_mae", "mean_auc_roc"]]
    .mean()
    .reindex(model_order)
)
print("\n" + "=" * 100)
print("AGGREGATED SUMMARY (mean across all datasets and classes)")
print("=" * 100)
display(df_bench_agg)