#!/usr/bin/env python3
"""
hurdle_regressor.py
===================
Stage 2 high-tier regressor: VotingRegressor(LGBM + scaled Ridge).

LightGBM interpolates well on the mid-tier; Ridge extrapolates on the heavy tail.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGRESSOR_FILENAME = "lgb_hightier_regressor.joblib"
REGRESSOR_FILENAME_LEGACY = "lgb_hightier_regressor.txt"

DEFAULT_LGBM_N_ESTIMATORS = 100
DEFAULT_RIDGE_ALPHA = 1.0


def regressor_model_path(models_dir: Path) -> Path:
    """Canonical joblib path for the Stage 2 ensemble."""
    return models_dir / REGRESSOR_FILENAME


def resolve_regressor_path(models_dir: Path) -> Path:
    """Prefer joblib ensemble; fall back to legacy LightGBM .txt if present."""
    joblib_path = regressor_model_path(models_dir)
    if joblib_path.exists():
        return joblib_path
    legacy = models_dir / REGRESSOR_FILENAME_LEGACY
    if legacy.exists():
        return legacy
    return joblib_path


def prepare_regressor_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric matrix for VotingRegressor (Ridge cannot use pandas category dtype).

    LightGBM in the ensemble still benefits from integer cluster codes.
    """
    out = X.copy()
    if "Archetype_Cluster" in out.columns:
        if isinstance(out["Archetype_Cluster"].dtype, pd.CategoricalDtype):
            out["Archetype_Cluster"] = out["Archetype_Cluster"].astype(int)
        else:
            out["Archetype_Cluster"] = (
                pd.to_numeric(out["Archetype_Cluster"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
    return out


def build_high_tier_ensemble(n_estimators: int = DEFAULT_LGBM_N_ESTIMATORS) -> VotingRegressor:
    """LGBM (mid-tier structure) + Ridge (linear tail extrapolation)."""
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
    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=DEFAULT_RIDGE_ALPHA)),
        ]
    )
    return VotingRegressor(
        estimators=[("lgbm", lgb_model), ("ridge", ridge_model)],
    )


def load_regressor(path: Path) -> Union[VotingRegressor, object]:
    """
    Load Stage 2 model from joblib (ensemble) or legacy LightGBM booster file.

    Legacy .txt returns a lightgbm.Booster after load via Booster(model_file=...).
    """
    if path.suffix == ".joblib":
        return joblib.load(path)
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def predict_log_scale(
    regressor: object,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict log1p-scale targets; works for ensemble or legacy Booster."""
    X_reg = prepare_regressor_features(X)
    if hasattr(regressor, "predict"):
        return np.asarray(regressor.predict(X_reg), dtype=float)
    return np.asarray(regressor.predict(X_reg), dtype=float)


def save_regressor(ensemble: VotingRegressor, path: Path) -> None:
    """Persist ensemble with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, path)


def regressor_feature_importance(
    ensemble: VotingRegressor,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Feature importance from the LGBM leg of the voting ensemble."""
    lgbm = ensemble.named_estimators_["lgbm"]
    gain = lgbm.booster_.feature_importance(importance_type="gain")
    split = lgbm.booster_.feature_importance(importance_type="split")
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "gain": gain,
            "split": split,
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def cv_ensemble_log_rmse(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int,
) -> float:
    """Fit a fresh ensemble on the train fold and return validation log-RMSE."""
    model = build_high_tier_ensemble(n_estimators=n_estimators)
    model.fit(prepare_regressor_features(X_train), y_train)
    pred = model.predict(prepare_regressor_features(X_val))
    return float(np.sqrt(mean_squared_error(y_val, pred)))
