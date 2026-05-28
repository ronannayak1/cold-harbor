#!/usr/bin/env python3
"""
evaluate_album_residuals.py
===========================
Diagnostic evaluation of the LightGBM Tweedie album first-week model.

Stratified by Volume_Tier to isolate heavy-tail (megahit) vs floor (micro-indie).

Generates:
  1. Volume-tier facet scatter (log scale) — predicted vs actual per tier.
  2. Percentage-error bias boxplot per volume tier.
  3. Console table of MAE and median % error by volume tier.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_PATH = Path("models/album_firstweek/lgb_album_model.txt")
FEATURE_PATH = Path(
    "singles_artifacts/lead_pre_album/streaming_equivalent/album_meta_features.parquet"
)
TARGET_PATH = Path("data/78_pre_album_albums_firstweek.csv")
PLOT_DIR = Path("models/album_firstweek/plots")

TARGET_COL = "FIRST_WEEK_TOTAL_AE"
JOIN_KEY = "ALBUM_MRELG_ID"

FEATURE_COLS = [
    "Archetype_Cluster",
    "count_pre_release_tracks",
    "max_single_peak_volume",
    "total_pre_release_auc",
    "terminal_velocity",
    "composite_peak_momentum",
    "expected_week1_carryover",
    "catalog_momentum_slope",
    "norm_terminal_velocity",
    "norm_total_auc",
    "hype_concentration",
    "peak_proximity_weeks",
    "cannibalization_ratio",
]

VOLUME_BINS = [0, 50, 500, 5_000, 50_000, np.inf]
VOLUME_LABELS = [
    "Tier 5: Micro-Indie (< 50)",
    "Tier 4: Emerging (50 - 500)",
    "Tier 3: Mid-Tier (500 - 5k)",
    "Tier 2: Major (5k - 50k)",
    "Tier 1: Megahit (> 50k)",
]

PCT_ERROR_CLIP = (-1.0, 3.0)


def load_model(path: Path = MODEL_PATH) -> lgb.Booster:
    """Load serialized LightGBM booster."""
    return lgb.Booster(model_file=str(path))


def load_data(
    feature_path: Path = FEATURE_PATH,
    target_path: Path = TARGET_PATH,
) -> pd.DataFrame:
    """Load features + targets, merge, cast Archetype_Cluster as category."""
    features = pd.read_parquet(feature_path)
    features["Archetype_Cluster"] = features["Archetype_Cluster"].astype("category")
    features[JOIN_KEY] = features[JOIN_KEY].astype(str).str.strip()

    suffix = target_path.suffix.lower()
    if suffix == ".parquet":
        targets = pd.read_parquet(target_path)
    else:
        targets = pd.read_csv(target_path, encoding="utf-8-sig")

    if JOIN_KEY not in targets.columns and "MRELG_ID" in targets.columns:
        targets = targets.rename(columns={"MRELG_ID": JOIN_KEY})

    targets[JOIN_KEY] = targets[JOIN_KEY].astype(str).str.strip()
    targets[TARGET_COL] = pd.to_numeric(targets[TARGET_COL], errors="coerce")
    targets = targets[[JOIN_KEY, TARGET_COL]].dropna(subset=[TARGET_COL])
    targets = targets.drop_duplicates(subset=[JOIN_KEY], keep="first")

    df = features.merge(targets, on=JOIN_KEY, how="inner")
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    return df


def run_inference(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions, residual metrics, and volume tier assignment."""
    X = df[FEATURE_COLS].copy()
    preds = model.predict(X)

    df = df.copy()
    df["predicted"] = preds
    df["actual"] = df[TARGET_COL]
    df["residual"] = df["actual"] - df["predicted"]

    df["percentage_error"] = np.where(
        df["actual"].abs() > 1e-6,
        (df["predicted"] - df["actual"]) / df["actual"],
        0.0,
    )

    df["log_actual"] = np.log1p(df["actual"].clip(lower=0.0))
    df["log_predicted"] = np.log1p(df["predicted"].clip(lower=0.0))

    df["Volume_Tier"] = pd.cut(
        df["actual"],
        bins=VOLUME_BINS,
        labels=VOLUME_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


def plot_tier_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 1: Faceted scatter of log_predicted vs log_actual by Volume_Tier."""
    tiers = [t for t in VOLUME_LABELS if t in df["Volume_Tier"].cat.categories]
    n_tiers = len(tiers)
    col_wrap = 3
    nrows = (n_tiers + col_wrap - 1) // col_wrap

    fig, axes = plt.subplots(nrows, col_wrap, figsize=(15, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    global_min = min(df["log_actual"].min(), df["log_predicted"].min())
    global_max = max(df["log_actual"].max(), df["log_predicted"].max())
    palette = sns.color_palette("deep", n_colors=n_tiers)

    for idx, tier in enumerate(tiers):
        ax = axes_flat[idx]
        sub = df[df["Volume_Tier"] == tier]
        ax.scatter(
            sub["log_actual"],
            sub["log_predicted"],
            alpha=0.2,
            s=10,
            edgecolors="none",
            color=palette[idx],
        )
        ax.plot(
            [global_min, global_max],
            [global_min, global_max],
            "r--",
            linewidth=1.2,
            label="y = x",
        )
        ax.set_xlim(global_min - 0.2, global_max + 0.2)
        ax.set_ylim(global_min - 0.2, global_max + 0.2)
        ax.set_xlabel("ln(1 + actual)")
        ax.set_ylabel("ln(1 + predicted)")
        ax.set_title(f"{tier}  (n={len(sub):,})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    for idx in range(n_tiers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Predicted vs Actual (Log Scale) by Volume Tier", fontsize=13, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = output_dir / "tier_scatter_log.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tier_bias_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 2: Horizontal boxplot of percentage_error by Volume_Tier (clipped)."""
    plot_df = df[["Volume_Tier", "percentage_error"]].copy()
    plot_df["pct_error_clipped"] = plot_df["percentage_error"].clip(
        lower=PCT_ERROR_CLIP[0], upper=PCT_ERROR_CLIP[1]
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_df,
        y="Volume_Tier",
        x="pct_error_clipped",
        hue="Volume_Tier",
        orient="h",
        ax=ax,
        palette="deep",
        fliersize=2,
        linewidth=0.8,
        legend=False,
        order=VOLUME_LABELS,
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero bias")
    ax.set_xlabel("Percentage Error  [(predicted − actual) / actual]")
    ax.set_ylabel("Volume Tier")
    ax.set_title("Model Bias by Volume Tier (clipped to [-1, +3])")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "bias_boxplot_by_tier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_tier_summary(df: pd.DataFrame) -> None:
    """Print MAE and median percentage error by Volume_Tier."""
    summary = (
        df.groupby("Volume_Tier", observed=True)
        .agg(
            n_albums=("actual", "size"),
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            mae=("residual", lambda x: x.abs().mean()),
            median_pct_error=("percentage_error", "median"),
        )
        .reset_index()
    )
    summary["mae"] = summary["mae"].round(1)
    summary["median_pct_error"] = (summary["median_pct_error"] * 100).round(2)
    summary["mean_actual"] = summary["mean_actual"].round(1)
    summary["mean_predicted"] = summary["mean_predicted"].round(1)

    print("\n" + "=" * 80)
    print("Volume-Tier Diagnostics")
    print("=" * 80)
    print(
        f"{'Tier':<30} {'N':>7} {'Mean Act':>10} {'Mean Pred':>10} "
        f"{'MAE':>10} {'Med %Err':>9}"
    )
    print("-" * 80)
    for _, row in summary.iterrows():
        print(
            f"{row['Volume_Tier']:<30} "
            f"{int(row['n_albums']):>7,} "
            f"{row['mean_actual']:>10,.1f} "
            f"{row['mean_predicted']:>10,.1f} "
            f"{row['mae']:>10,.1f} "
            f"{row['median_pct_error']:>8.2f}%"
        )
    print("-" * 80)

    overall_mae = df["residual"].abs().mean()
    overall_med_pct = df["percentage_error"].median() * 100
    print(
        f"{'OVERALL':<30} "
        f"{len(df):>7,} "
        f"{df['actual'].mean():>10,.1f} "
        f"{df['predicted'].mean():>10,.1f} "
        f"{overall_mae:>10,.1f} "
        f"{overall_med_pct:>8.2f}%"
    )
    print("=" * 80)


def main() -> None:
    print("=" * 60)
    print("Album Model Residual Diagnostics (by Volume Tier)")
    print("=" * 60)

    print("\n[1/4] Loading model and data...")
    model = load_model()
    df = load_data()
    print(f"  Rows: {len(df):,}")

    print("\n[2/4] Running inference + tier assignment...")
    df = run_inference(model, df)
    tier_counts = df["Volume_Tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        print(f"  {tier}: {cnt:,}")

    print("\n[3/4] Generating diagnostic plots...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_tier_scatter(df, PLOT_DIR)
    plot_tier_bias_boxplot(df, PLOT_DIR)

    print("\n[4/4] Tier summary statistics:")
    print_tier_summary(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
