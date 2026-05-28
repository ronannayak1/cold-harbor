#!/usr/bin/env python3
"""
train_album_model.py
====================
LightGBM Tweedie regression for album first-week unit volume prediction.

Architecture:
  - Tweedie objective (p=1.5) handles the heavy power-law tail natively.
  - GroupKFold by DISPLAY_ARTIST prevents same-artist leakage across folds.
  - Archetype_Cluster treated as native categorical (no one-hot encoding).
  - Feature importance by gain (variance reduction contribution).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

FEATURE_PATH = Path(
    "singles_artifacts/lead_pre_album/streaming_equivalent/album_meta_features.parquet"
)
TARGET_PATH = Path("data/78_pre_album_albums_firstweek.csv")
OUTPUT_DIR = Path("models/album_firstweek")

TARGET_COL = "FIRST_WEEK_TOTAL_AE"
JOIN_KEY = "ALBUM_MRELG_ID"
GROUP_COL = "DISPLAY_ARTIST"

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

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "tweedie",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "verbose": -1,
}


def load_features(path: Path = FEATURE_PATH) -> pd.DataFrame:
    """Load Phase 2 album meta-features."""
    df = pd.read_parquet(path)
    df["Archetype_Cluster"] = df["Archetype_Cluster"].astype("category")
    return df


def load_and_merge_targets(
    feature_df: pd.DataFrame,
    target_path: Path = TARGET_PATH,
) -> pd.DataFrame:
    """
    Merge target variable onto features.

    Supports both CSV (current: 78_pre_album_albums_firstweek.csv with MRELG_ID key)
    and parquet (future: album_first_week_targets.parquet with ALBUM_MRELG_ID key).
    """
    suffix = target_path.suffix.lower()
    if suffix == ".parquet":
        targets = pd.read_parquet(target_path)
    else:
        targets = pd.read_csv(target_path, encoding="utf-8-sig")

    # Normalize join key
    if JOIN_KEY not in targets.columns and "MRELG_ID" in targets.columns:
        targets = targets.rename(columns={"MRELG_ID": JOIN_KEY})

    targets[JOIN_KEY] = targets[JOIN_KEY].astype(str).str.strip()
    targets[TARGET_COL] = pd.to_numeric(targets[TARGET_COL], errors="coerce")
    targets = targets[[JOIN_KEY, TARGET_COL]].dropna(subset=[TARGET_COL])
    targets = targets.drop_duplicates(subset=[JOIN_KEY], keep="first")

    feature_df[JOIN_KEY] = feature_df[JOIN_KEY].astype(str).str.strip()
    merged = feature_df.merge(targets, on=JOIN_KEY, how="inner")
    merged = merged[merged[TARGET_COL] > 0].reset_index(drop=True)
    return merged


def run_group_kfold_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
) -> Tuple[List[float], List[int]]:
    """
    GroupKFold CV grouped by DISPLAY_ARTIST to prevent artist leakage.
    Returns per-fold Tweedie scores and best iterations.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = df[GROUP_COL].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: List[float] = []
    best_iters: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = model.best_iteration_
        best_score = model.best_score_["valid_0"]["tweedie"]
        fold_scores.append(best_score)
        best_iters.append(best_iter)

        print(
            f"  Fold {fold_idx}/{n_splits}: "
            f"tweedie={best_score:.6f}, best_iter={best_iter}"
        )

    return fold_scores, best_iters


def train_final_model(
    df: pd.DataFrame,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    """Train on the full dataset with the optimal iteration count from CV."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)

    params = {**LGB_PARAMS, "n_estimators": n_estimators}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def save_feature_importance(
    model: lgb.LGBMRegressor,
    output_dir: Path,
) -> pd.DataFrame:
    """Save feature importance by gain to CSV and print ranking."""
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM Tweedie album model")
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH)
    parser.add_argument("--target-path", type=Path, default=TARGET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("Album First-Week Prediction — LightGBM Tweedie (p=1.5)")
    print("=" * 60)

    print("\n[1/5] Loading features...")
    features = load_features(args.feature_path)
    print(f"  Feature rows: {len(features):,}")

    print("\n[2/5] Merging targets...")
    df = load_and_merge_targets(features, target_path=args.target_path)
    print(f"  Rows with valid target: {len(df):,}")
    print(f"  Unique artists: {df[GROUP_COL].nunique():,}")
    print(f"  Target stats — median: {df[TARGET_COL].median():.1f}, "
          f"mean: {df[TARGET_COL].mean():.1f}, "
          f"max: {df[TARGET_COL].max():.1f}")

    print(f"\n[3/5] GroupKFold CV ({args.n_splits} folds, grouped by {GROUP_COL})...")
    fold_scores, best_iters = run_group_kfold_cv(
        df,
        n_splits=args.n_splits,
        early_stopping_rounds=args.early_stopping,
    )
    avg_score = float(np.mean(fold_scores))
    avg_iter = int(np.mean(best_iters))
    print(f"\n  Average CV Tweedie loss: {avg_score:.6f}")
    print(f"  Average best iteration: {avg_iter}")

    print(f"\n[4/5] Training final model (n_estimators={avg_iter})...")
    final_model = train_final_model(df, n_estimators=avg_iter)

    print(f"\n[5/5] Saving artifacts to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = args.output_dir / "lgb_album_model.txt"
    final_model.booster_.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    importance = save_feature_importance(final_model, args.output_dir)
    print("\n  Feature importance (gain):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:30s}  {row['importance_gain']:.1f}")

    meta = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "cv_folds": args.n_splits,
        "avg_cv_tweedie_loss": avg_score,
        "avg_best_iteration": avg_iter,
        "n_training_rows": len(df),
        "n_unique_artists": int(df[GROUP_COL].nunique()),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
    }
    with open(args.output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {args.output_dir / 'model_meta.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
