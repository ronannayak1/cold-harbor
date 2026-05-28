#!/usr/bin/env python3
"""
train_hurdle_model.py
=====================
Two-Stage Hurdle Architecture for album first-week prediction (multi-target).

Stage 1 (Gatekeeper): Binary classifier — album clears the 75th percentile of the
    training target (dynamic threshold, saved to hurdle_model_meta.json).
Stage 2 (High-Tier Regressor): VotingRegressor(LGBM + Ridge) on log1p target,
    albums >= dynamic threshold (joblib artifact).

Use --target to switch total_ae | streaming | product | song.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hurdle_regressor import prepare_regressor_features, regressor_feature_importance, regressor_model_path
from target_config import (
    TargetConfig,
    add_target_cli,
    build_training_feature_cols,
    get_target_config,
    validate_target_data_dir,
)

JOIN_KEY = "ALBUM_MRELG_ID"
GROUP_COL = "DISPLAY_ARTIST"
GATEKEEPER_PERCENTILE = 75

CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
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
    "is_unbalance": True,
}

REGRESSOR_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "verbose": -1,
}


def compute_dynamic_threshold(df: pd.DataFrame, target_col: str) -> float:
    """75th percentile of the target column (Stage 1 gatekeeper cutoff)."""
    values = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No valid values for target column {target_col!r}")
    return float(np.percentile(values, GATEKEEPER_PERCENTILE))


def load_features(path: Path) -> pd.DataFrame:
    """Load Phase 2 album meta-features."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. "
            "Run singles_archetypes.py --phase 2 for this --target first."
        )
    df = pd.read_parquet(path)
    df["Archetype_Cluster"] = df["Archetype_Cluster"].astype("category")
    return df


def load_and_merge_targets(
    feature_df: pd.DataFrame,
    cfg: TargetConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge target variable onto features; return df and feature column list."""
    target_path = cfg.targets_path
    suffix = target_path.suffix.lower()
    if suffix == ".parquet":
        targets = pd.read_parquet(target_path)
    else:
        targets = pd.read_csv(target_path, encoding="utf-8-sig")

    if JOIN_KEY not in targets.columns and "MRELG_ID" in targets.columns:
        targets = targets.rename(columns={"MRELG_ID": JOIN_KEY})

    if cfg.target_col not in targets.columns:
        raise ValueError(
            f"{target_path} missing target column {cfg.target_col} for --target {cfg.name}"
        )

    targets[JOIN_KEY] = targets[JOIN_KEY].astype(str).str.strip()
    targets[cfg.target_col] = pd.to_numeric(targets[cfg.target_col], errors="coerce")
    targets = targets[[JOIN_KEY, cfg.target_col]].dropna(subset=[cfg.target_col])
    targets = targets.drop_duplicates(subset=[JOIN_KEY], keep="first")

    feature_df[JOIN_KEY] = feature_df[JOIN_KEY].astype(str).str.strip()
    merged = feature_df.merge(targets, on=JOIN_KEY, how="inner")
    merged = merged[merged[cfg.target_col] > 0].reset_index(drop=True)

    merged["actual"] = merged[cfg.target_col]
    feature_cols = build_training_feature_cols(merged, cfg)
    return merged, feature_cols


def cv_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
) -> Tuple[List[float], List[float], List[int], float]:
    """GroupKFold CV for Stage 1 binary classifier."""
    dynamic_threshold = compute_dynamic_threshold(df, target_col)
    X = df[feature_cols].copy()
    X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")
    y = (df[target_col] >= dynamic_threshold).astype(int).to_numpy()
    groups = df[GROUP_COL].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    fold_logloss: List[float] = []
    fold_auc: List[float] = []
    best_iters: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
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
        best_score = model.best_score_["valid_0"]["binary_logloss"]
        fold_logloss.append(best_score)
        best_iters.append(best_iter)

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        fold_auc.append(auc)

        print(
            f"  Fold {fold_idx}/{n_splits}: "
            f"logloss={best_score:.5f}, AUC={auc:.4f}, best_iter={best_iter}"
        )

    return fold_logloss, fold_auc, best_iters, dynamic_threshold


def cv_regressor(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
) -> Tuple[List[float], List[int]]:
    """GroupKFold CV for Stage 2 high-tier log-RMSE regressor."""
    dynamic_threshold = compute_dynamic_threshold(df, target_col)
    high_df = df[df[target_col] >= dynamic_threshold].copy()

    X = high_df[feature_cols].copy()
    X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")
    y = np.log1p(high_df[target_col].to_numpy(dtype=float))
    groups = high_df[GROUP_COL].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: List[float] = []
    best_iters: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
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
        best_score = model.best_score_["valid_0"]["rmse"]
        fold_scores.append(best_score)
        best_iters.append(best_iter)

        print(
            f"  Fold {fold_idx}/{n_splits}: "
            f"rmse={best_score:.4f}, best_iter={best_iter}"
        )

    return fold_scores, best_iters


def train_final_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int,
) -> lgb.LGBMClassifier:
    """Train final Stage 1 classifier on full data."""
    dynamic_threshold = compute_dynamic_threshold(df, target_col)
    X = df[feature_cols].copy()
    X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")
    y = (df[target_col] >= dynamic_threshold).astype(int).to_numpy()
    params = {**CLASSIFIER_PARAMS, "n_estimators": n_estimators}
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def train_final_regressor(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int,
) -> VotingRegressor:
    """Train Stage 2 VotingRegressor (LGBM + scaled Ridge) on high-tier log1p target."""
    dynamic_threshold = compute_dynamic_threshold(df, target_col)
    high_df = df[df[target_col] >= dynamic_threshold].copy()
    X = prepare_regressor_features(high_df[feature_cols])
    y = np.log1p(high_df[target_col].to_numpy(dtype=float))

    lgb_model = LGBMRegressor(
        objective="regression",
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ensemble = VotingRegressor(
        estimators=[("lgbm", lgb_model), ("ridge", ridge_model)],
    )
    ensemble.fit(X, y)
    return ensemble


def print_importance(model, label: str, feature_cols: List[str]) -> pd.DataFrame:
    """Print feature importance by gain."""
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    print(f"\n  Feature Importance ({label}, by gain):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:30s}  {row['gain']:>12,.1f}")
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Two-Stage Hurdle Model (classifier + high-tier regressor)"
    )
    add_target_cli(parser)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--target-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=50)
    args = parser.parse_args()

    cfg = get_target_config(args.target)
    if not cfg.use_firstweekae_baseline:
        validate_target_data_dir(cfg)

    feature_path = args.feature_path or cfg.features_path
    output_dir = args.output_dir or cfg.models_dir
    target_col = cfg.target_col

    print("=" * 65)
    print(f"Two-Stage Hurdle — target={cfg.name}")
    print(f"  Stage 1: dynamic gatekeeper = P{GATEKEEPER_PERCENTILE} of {target_col}")
    print(f"  Stage 2: VotingRegressor LGBM+Ridge (trained on >= dynamic threshold only)")
    print(f"  Target column: {target_col}")
    print("=" * 65)

    print("\n[1/6] Loading features...")
    features = load_features(feature_path)
    print(f"  Feature rows: {len(features):,}")
    print(f"  From: {feature_path}")

    print("\n[2/6] Merging targets...")
    df, feature_cols = load_and_merge_targets(features, cfg)
    dynamic_threshold = compute_dynamic_threshold(df, target_col)
    n_major = int((df[target_col] >= dynamic_threshold).sum())
    print(f"  Training features ({len(feature_cols)}): {feature_cols}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Dynamic threshold (P{GATEKEEPER_PERCENTILE}): {dynamic_threshold:,.2f}")
    print(f"  Major releases (>= threshold): {n_major:,} ({100*n_major/len(df):.1f}%)")
    print(f"  Sub-threshold: {len(df) - n_major:,}")

    print(f"\n[3/6] Stage 1 — Classifier GroupKFold CV ({args.n_splits} folds)...")
    clf_logloss, clf_auc, clf_iters, _ = cv_classifier(
        df,
        feature_cols,
        target_col,
        n_splits=args.n_splits,
        early_stopping_rounds=args.early_stopping,
    )
    avg_logloss = float(np.mean(clf_logloss))
    avg_auc = float(np.mean(clf_auc))
    avg_clf_iter = int(np.mean(clf_iters))
    print(f"\n  Avg Logloss: {avg_logloss:.5f}")
    print(f"  Avg AUC:     {avg_auc:.4f}")
    print(f"  Avg best iteration: {avg_clf_iter}")

    high_tier_n = int((df[target_col] >= dynamic_threshold).sum())
    print(
        f"\n[4/6] Stage 2 — High-Tier Regressor GroupKFold CV "
        f"({high_tier_n:,} rows, {args.n_splits} folds)..."
    )
    if high_tier_n < 10:
        print("  ERROR: Too few high-tier rows for regressor CV. Add data or check targets.")
        return
    reg_scores, reg_iters = cv_regressor(
        df,
        feature_cols,
        target_col,
        n_splits=args.n_splits,
        early_stopping_rounds=args.early_stopping,
    )
    avg_rmse = float(np.mean(reg_scores))
    avg_reg_iter = int(np.mean(reg_iters))
    print(f"\n  Avg RMSE (log-scale, high-tier): {avg_rmse:.4f}")
    print(f"  Avg best iteration: {avg_reg_iter}")

    print(f"\n[5/6] Training final models...")
    print(f"  Classifier: n_estimators={avg_clf_iter} (full dataset)")
    final_clf = train_final_classifier(
        df, feature_cols, target_col, n_estimators=avg_clf_iter
    )

    print(f"  Regressor:  n_estimators={avg_reg_iter} (high-tier only)")
    final_reg = train_final_regressor(
        df, feature_cols, target_col, n_estimators=avg_reg_iter
    )

    print(f"\n[6/6] Saving artifacts to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    clf_path = output_dir / "lgb_hurdle_classifier.txt"
    reg_path = regressor_model_path(output_dir)
    final_clf.booster_.save_model(str(clf_path))
    joblib.dump(final_reg, reg_path)
    print(f"  Classifier model: {clf_path}")
    print(f"  Regressor model:  {reg_path}")

    clf_imp = print_importance(final_clf, "Stage 1 Classifier", feature_cols)
    reg_imp = regressor_feature_importance(final_reg, feature_cols)
    print(f"\n  Feature Importance (Stage 2 Regressor — LGBM leg, by gain):")
    for _, row in reg_imp.iterrows():
        print(f"    {row['feature']:30s}  {row['gain']:>12,.1f}")

    clf_imp.to_csv(output_dir / "importance_classifier.csv", index=False)
    reg_imp.to_csv(output_dir / "importance_regressor.csv", index=False)

    meta = {
        "architecture": "two_stage_hurdle",
        "target": cfg.name,
        "dynamic_threshold": dynamic_threshold,
        "gatekeeper_percentile": GATEKEEPER_PERCENTILE,
        "stage1": {
            "objective": "binary",
            "avg_cv_logloss": avg_logloss,
            "avg_cv_auc": avg_auc,
            "final_n_estimators": avg_clf_iter,
            "n_training_rows": len(df),
            "positive_rate": n_major / len(df),
        },
        "stage2": {
            "regressor_type": "voting_lgbm_ridge",
            "artifact_format": "joblib",
            "lgbm_n_estimators": avg_reg_iter,
            "ridge_alpha": 1.0,
            "target_transform": "log1p",
            "inference_inverse": "expm1",
            "avg_cv_rmse_lgbm": avg_rmse,
            "final_n_estimators": avg_reg_iter,
            "n_training_rows": high_tier_n,
            "n_unique_artists": int(
                df.loc[df[target_col] >= dynamic_threshold, GROUP_COL].nunique()
            ),
        },
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": GROUP_COL,
        "cv_folds": args.n_splits,
        "features_path": str(feature_path),
        "targets_path": str(cfg.targets_path),
    }
    with open(output_dir / "hurdle_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata: {output_dir / 'hurdle_model_meta.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
