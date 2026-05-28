#!/usr/bin/env python3
"""
evaluate_temporal_backtest.py
=============================
Walk-forward temporal backtest for the Two-Stage Hurdle model.

Unlike GroupKFold (which mixes eras within folds), each test year is predicted
using models trained only on albums released before that calendar year.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from target_config import TargetConfig, add_target_cli, get_target_config
from run_inference_pipeline import FALLBACK_BASE, apply_hurdle_cutoff
from sklearn.ensemble import VotingRegressor

from hurdle_regressor import build_high_tier_ensemble, predict_log_scale, prepare_regressor_features
from train_hurdle_model import (
    CLASSIFIER_PARAMS,
    REGRESSOR_PARAMS,
    compute_dynamic_threshold,
    load_and_merge_targets,
    load_features,
)

CLASSIFIER_THRESHOLD = 0.50

PLOT_DIR = Path("models/album_firstweek/plots/temporal")

TEST_YEARS = [2022, 2023, 2024, 2025]

VOLUME_BINS = [0, 50, 500, 5_000, 50_000, np.inf]
VOLUME_LABELS = [
    "Tier 5: Micro",
    "Tier 4: Emerging",
    "Tier 3: Mid",
    "Tier 2: Major",
    "Tier 1: Megahit",
]


def prepare_dataset(cfg: TargetConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Load features, merge targets, parse album sale date."""
    features = load_features(cfg.features_path)
    df, feature_cols = load_and_merge_targets(features, cfg)
    df["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(df["ALBUM_FIRST_SALE_DATE"], errors="coerce")
    df = df.dropna(subset=["ALBUM_FIRST_SALE_DATE"]).reset_index(drop=True)
    df["actual"] = df[cfg.target_col].astype(float)
    df["release_year"] = df["ALBUM_FIRST_SALE_DATE"].dt.year
    return df, feature_cols


def _temporal_val_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent fraction of training rows for early stopping."""
    train_sorted = train_df.sort_values("ALBUM_FIRST_SALE_DATE")
    n_val = max(1, int(len(train_sorted) * val_fraction))
    val_df = train_sorted.iloc[-n_val:]
    fit_df = train_sorted.iloc[:-n_val]
    if fit_df.empty:
        fit_df = train_sorted.iloc[:-1]
        val_df = train_sorted.iloc[-1:]
    return fit_df, val_df


def train_fold_models(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    early_stopping_rounds: int = 50,
) -> Tuple[lgb.LGBMClassifier, VotingRegressor, int, int]:
    """Train Stage 1 classifier and Stage 2 high-tier regressor on train_df."""
    dynamic_threshold = compute_dynamic_threshold(train_df, target_col)
    fit_df, val_df = _temporal_val_split(train_df)

    X_fit = fit_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_fit["Archetype_Cluster"] = X_fit["Archetype_Cluster"].astype("category")
    X_val["Archetype_Cluster"] = X_val["Archetype_Cluster"].astype("category")

    y_clf_fit = (fit_df[target_col] >= dynamic_threshold).astype(int).to_numpy()
    y_clf_val = (val_df[target_col] >= dynamic_threshold).astype(int).to_numpy()

    clf = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    clf.fit(
        X_fit,
        y_clf_fit,
        eval_set=[(X_val, y_clf_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    clf_iter = clf.best_iteration_ or CLASSIFIER_PARAMS["n_estimators"]

    high_train = train_df[train_df[target_col] >= dynamic_threshold].copy()
    y_log = np.log1p(high_train["actual"].to_numpy(dtype=float))
    X_h = prepare_regressor_features(high_train[feature_cols])

    if len(high_train) < 10:
        reg = build_high_tier_ensemble(n_estimators=50)
        reg.fit(X_h, y_log)
        return clf, reg, clf_iter, 50

    # Tune LGBM leg n_estimators via early stopping; fit full ensemble on all high-tier rows.
    fit_h, val_h = _temporal_val_split(high_train)
    X_h_fit = prepare_regressor_features(fit_h[feature_cols])
    X_h_val = prepare_regressor_features(val_h[feature_cols])
    y_h_fit = np.log1p(fit_h["actual"].to_numpy(dtype=float))
    y_h_val = np.log1p(val_h["actual"].to_numpy(dtype=float))

    lgb_tune = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    lgb_tune.fit(
        X_h_fit,
        y_h_fit,
        eval_set=[(X_h_val, y_h_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    reg_iter = lgb_tune.best_iteration_ or REGRESSOR_PARAMS["n_estimators"]

    reg = build_high_tier_ensemble(n_estimators=reg_iter)
    reg.fit(X_h, y_log)
    return clf, reg, clf_iter, reg_iter


def hurdle_predict(
    clf: lgb.LGBMClassifier,
    reg: VotingRegressor,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = CLASSIFIER_THRESHOLD,
    fallback_base: float = FALLBACK_BASE,
    dynamic_threshold: float | None = None,
) -> pd.DataFrame:
    """Hard-cutoff hurdle inference on test_df (aligned with run_inference_pipeline)."""
    X = test_df[feature_cols].copy()
    X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")

    carryover = test_df["expected_week1_carryover"].to_numpy(dtype=float)
    prob = clf.predict_proba(X)[:, 1]
    high_tier = np.expm1(predict_log_scale(reg, X))

    predicted, use_regressor, passes_clf, passes_vol = apply_hurdle_cutoff(
        prob,
        high_tier,
        carryover,
        prob_threshold=threshold,
        fallback_base=fallback_base,
        dynamic_threshold=dynamic_threshold,
    )

    out = test_df.copy()
    out["predicted"] = predicted
    out["Classifier_Probability"] = prob
    out["Is_Priority_Rollout"] = use_regressor
    out["Passes_Classifier"] = passes_clf
    out["Passes_Volume_Gate"] = passes_vol
    out["residual"] = out["actual"] - out["predicted"]
    out["percentage_error"] = np.where(
        out["actual"].abs() > 1e-6,
        (out["predicted"] - out["actual"]) / out["actual"],
        0.0,
    )
    out["Volume_Tier"] = pd.cut(
        out["actual"],
        bins=VOLUME_BINS,
        labels=VOLUME_LABELS,
        right=True,
        include_lowest=True,
    )
    return out


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_years: List[int],
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute walk-forward backtest; return per-row preds and yearly summary."""
    all_preds: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    for year in test_years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year + 1}-01-01")

        train_df = df[df["ALBUM_FIRST_SALE_DATE"] < cutoff].copy()
        test_df = df[
            (df["ALBUM_FIRST_SALE_DATE"] >= cutoff) & (df["ALBUM_FIRST_SALE_DATE"] < year_end)
        ].copy()

        if train_df.empty or test_df.empty:
            print(f"  {year}: skipped (train={len(train_df):,}, test={len(test_df):,})")
            continue

        fold_p75 = compute_dynamic_threshold(train_df, target_col)
        print(
            f"  {year}: train={len(train_df):,}, test={len(test_df):,}, "
            f"P75={fold_p75:,.0f} ...",
            end=" ",
        )
        clf, reg, _, _ = train_fold_models(train_df, feature_cols, target_col)
        fold_preds = hurdle_predict(
            clf,
            reg,
            test_df,
            feature_cols,
            threshold=threshold,
            dynamic_threshold=fold_p75,
        )
        fold_preds["test_year"] = year
        all_preds.append(fold_preds)

        mae = float(fold_preds["residual"].abs().mean())
        med_pct = float(fold_preds["percentage_error"].median() * 100)
        summaries.append({
            "test_year": year,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "mae": mae,
            "median_pct_error": med_pct,
            "n_priority": int(fold_preds["Is_Priority_Rollout"].sum()),
        })
        print(f"MAE={mae:,.1f}, Med%Err={med_pct:.2f}%")

    if not all_preds:
        raise ValueError("No walk-forward folds produced predictions.")

    preds_df = pd.concat(all_preds, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    return preds_df, summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print yearly backtest metrics."""
    print("\n" + "=" * 72)
    print("Walk-Forward Temporal Backtest Summary")
    print("=" * 72)
    print(
        f"{'Year':>6} {'Train':>8} {'Test':>8} {'Priority':>10} "
        f"{'MAE':>12} {'Med %Err':>10}"
    )
    print("-" * 72)
    for _, row in summary_df.iterrows():
        print(
            f"{int(row['test_year']):>6} "
            f"{int(row['train_size']):>8,} "
            f"{int(row['test_size']):>8,} "
            f"{int(row['n_priority']):>10,} "
            f"{row['mae']:>12,.1f} "
            f"{row['median_pct_error']:>9.2f}%"
        )
    print("=" * 72)


def plot_error_over_time(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart: MAE and median % error by test year."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    years = summary_df["test_year"].astype(int)

    ax1.plot(years, summary_df["mae"], "o-", color="steelblue", linewidth=2, label="MAE")
    ax1.set_xlabel("Test Year")
    ax1.set_ylabel("MAE", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        summary_df["median_pct_error"],
        "s--",
        color="coral",
        linewidth=2,
        label="Median % Error",
    )
    ax2.set_ylabel("Median % Error", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1)

    fig.suptitle("Hurdle Model Error Over Time (Walk-Forward)", fontsize=13)
    fig.tight_layout()
    path = output_dir / "error_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tier_accuracy_shift(preds_df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of MAE by Volume_Tier x test_year."""
    tier_year = (
        preds_df.groupby(["test_year", "Volume_Tier"], observed=True)
        .agg(mae=("residual", lambda x: x.abs().mean()), n=("actual", "size"))
        .reset_index()
    )
    pivot = tier_year.pivot(index="Volume_Tier", columns="test_year", values="mae")
    pivot = pivot.reindex(VOLUME_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "MAE"},
    )
    ax.set_title("MAE by Volume Tier and Test Year")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Volume Tier")
    fig.tight_layout()
    path = output_dir / "tier_accuracy_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward temporal hurdle backtest")
    add_target_cli(parser)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=TEST_YEARS,
        help="Calendar years to hold out as test folds",
    )
    args = parser.parse_args()

    cfg = get_target_config(args.target)
    output_dir = args.output_dir or (cfg.models_dir / "plots/temporal")

    print("=" * 65)
    print(f"Walk-Forward Temporal Backtest — target={cfg.name}")
    print(f"  Test years: {args.test_years}")
    print(f"  Classifier threshold: {args.threshold}")
    print(f"  Gatekeeper: P75 of {cfg.target_col} (per train fold)")
    print("=" * 65)

    print("\n[1/4] Loading data...")
    df, feature_cols = prepare_dataset(cfg)
    print(f"  Albums with dates + targets: {len(df):,}")
    print(f"  Date range: {df['ALBUM_FIRST_SALE_DATE'].min().date()} – "
          f"{df['ALBUM_FIRST_SALE_DATE'].max().date()}")
    p75_full = compute_dynamic_threshold(df, cfg.target_col)
    print(f"  Full-sample P75 ({cfg.target_col}): {p75_full:,.2f}")

    print("\n[2/4] Walk-forward training & inference...")
    preds_df, summary_df = run_walk_forward(
        df, feature_cols, cfg.target_col, args.test_years, args.threshold
    )

    print("\n[3/4] Summary table:")
    print_summary_table(summary_df)

    print("\n[4/4] Saving plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_error_over_time(summary_df, output_dir)
    plot_tier_accuracy_shift(preds_df, output_dir)

    preds_path = output_dir / "temporal_backtest_predictions.parquet"
    summary_path = output_dir / "temporal_backtest_summary.csv"
    preds_df.to_parquet(preds_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Predictions: {preds_path}")
    print(f"  Summary: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
