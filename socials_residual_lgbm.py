#!/usr/bin/env python3
"""
Rebuilt first-week model using artist history + smoothed social momentum.

Target:
- log(FIRST_WEEK_TOTAL_AE + 1)

Core features:
- log(Lag_1_First_Week_AE + 1)
- log(Prior_Avg_First_Week_AE + 1)
- Days_Since_Last_Release

Momentum + scale features (14-day pre-release window):
- IG_Rolling_7D_vs_Previous_7D
- TT_Rolling_7D_vs_Previous_7D
- log(IG_Total_Eng_14d + 1)
- log(TT_Total_Eng_14d + 1)
- TT_Share_of_Total
- log(Absolute_Follower_Count_Day_14 + 1) if present in socials file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc


COHORT_MIN_DATE = pd.Timestamp("2025-08-01")
SOCIAL_LOOKBACK_DAYS = 14
TEST_FRAC = 0.2
PRIOR_AVG_THRESHOLD = 12000.0
EPS = 1e-6

FIRSTWEEKAE_SOCIALS_PATH = Path("data/firstweekae_socials.csv")
SOCIALS_MOMENTUM_PATH = Path("data/socials_momentum.csv")
FIRSTWEEKAE_PATH = Path("data/firstweekae.csv")


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    feature_cols: List[str]


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer / max(denom, EPS))


def _detect_follower_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "FOLLOWER_COUNT",
        "FOLLOWERS",
        "IG_FOLLOWER_COUNT",
        "IG_FOLLOWERS",
        "TOTAL_FOLLOWERS",
        "ABSOLUTE_FOLLOWER_COUNT_DAY_14",
    ]
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_upper:
            return cols_upper[cand]
    return None


def load_release_cohort() -> pd.DataFrame:
    df = pd.read_csv(FIRSTWEEKAE_SOCIALS_PATH, encoding="utf-8-sig")
    required = {"MRELG_ID", "FIRST_SALE_DATE", "SODATONE_ARTIST_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in firstweekae_socials: {sorted(missing)}")

    df = df[["MRELG_ID", "FIRST_SALE_DATE", "SODATONE_ARTIST_ID"]].copy()
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df["MRELG_ID"] = df["MRELG_ID"].astype(str).str.strip()
    df["SODATONE_ARTIST_ID"] = df["SODATONE_ARTIST_ID"].astype(str).str.strip()
    df = df.dropna(subset=["FIRST_SALE_DATE"])
    df = df[(df["FIRST_SALE_DATE"] >= COHORT_MIN_DATE) & (df["MRELG_ID"] != "")]
    return df.drop_duplicates(subset=["MRELG_ID"]).reset_index(drop=True)


def load_socials() -> tuple[pd.DataFrame, str | None]:
    df = pd.read_csv(SOCIALS_MOMENTUM_PATH, encoding="utf-8-sig")
    required = {"SODATONE_ARTIST_ID", "DATE", "IG_ENGAGEMENT", "TT_ENGAGEMENT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in socials_momentum: {sorted(missing)}")

    follower_col = _detect_follower_column(df)
    keep_cols = ["SODATONE_ARTIST_ID", "DATE", "IG_ENGAGEMENT", "TT_ENGAGEMENT"]
    if follower_col is not None:
        keep_cols.append(follower_col)

    df = df[keep_cols].copy()
    df["SODATONE_ARTIST_ID"] = df["SODATONE_ARTIST_ID"].astype(str).str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["IG_ENGAGEMENT"] = pd.to_numeric(df["IG_ENGAGEMENT"], errors="coerce").fillna(0.0)
    df["TT_ENGAGEMENT"] = pd.to_numeric(df["TT_ENGAGEMENT"], errors="coerce").fillna(0.0)
    if follower_col is not None:
        df[follower_col] = pd.to_numeric(df[follower_col], errors="coerce")
    df = df.dropna(subset=["DATE"])
    return df, follower_col


def load_first_week_actuals_and_artist() -> pd.DataFrame:
    df = pd.read_csv(FIRSTWEEKAE_PATH, encoding="utf-8-sig")
    required = {"MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in firstweekae: {sorted(missing)}")

    df = df[["MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"]].copy()
    df["MRELG_ID"] = df["MRELG_ID"].astype(str).str.strip()
    df["DISPLAY_ARTIST"] = df["DISPLAY_ARTIST"].astype(str).str.strip()
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df["FIRST_WEEK_TOTAL_AE"] = pd.to_numeric(df["FIRST_WEEK_TOTAL_AE"], errors="coerce")
    df = df.dropna(subset=["FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"])
    df = df[df["MRELG_ID"] != ""]
    return df


def build_release_history_features(firstweek_df: pd.DataFrame) -> pd.DataFrame:
    df = firstweek_df.sort_values(["DISPLAY_ARTIST", "FIRST_SALE_DATE", "MRELG_ID"]).copy()
    grp = df.groupby("DISPLAY_ARTIST", sort=False)

    df["LAG_1_FIRST_WEEK_AE"] = grp["FIRST_WEEK_TOTAL_AE"].shift(1)
    df["PRIOR_AVG_FIRST_WEEK_AE"] = grp["FIRST_WEEK_TOTAL_AE"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    prev_date = grp["FIRST_SALE_DATE"].shift(1)
    df["DAYS_SINCE_LAST_RELEASE"] = (df["FIRST_SALE_DATE"] - prev_date).dt.days

    return df[
        [
            "MRELG_ID",
            "DISPLAY_ARTIST",
            "FIRST_SALE_DATE",
            "FIRST_WEEK_TOTAL_AE",
            "LAG_1_FIRST_WEEK_AE",
            "PRIOR_AVG_FIRST_WEEK_AE",
            "DAYS_SINCE_LAST_RELEASE",
        ]
    ]


def compute_social_features_for_release(
    sid: str,
    sale_date: pd.Timestamp,
    socials_by_id: Dict[str, pd.DataFrame],
    follower_col: str | None,
) -> Dict[str, float]:
    artist_socials = socials_by_id.get(sid)
    if artist_socials is None or artist_socials.empty:
        return {}

    start = sale_date - pd.Timedelta(days=SOCIAL_LOOKBACK_DAYS)
    end = sale_date - pd.Timedelta(days=1)
    win = artist_socials[(artist_socials["DATE"] >= start) & (artist_socials["DATE"] <= end)].copy()
    if win.empty:
        return {}

    win = win.sort_values("DATE")
    ig = win["IG_ENGAGEMENT"].astype(float)
    tt = win["TT_ENGAGEMENT"].astype(float)

    # Require enough points for 7-vs-previous-7 rolling velocity features.
    if len(win) < 14:
        return {}

    ig_prev7 = float(ig.iloc[0:7].mean())
    ig_last7 = float(ig.iloc[7:14].mean())
    tt_prev7 = float(tt.iloc[0:7].mean())
    tt_last7 = float(tt.iloc[7:14].mean())

    ig_total_14d = float(ig.iloc[0:14].sum())
    tt_total_14d = float(tt.iloc[0:14].sum())
    total_14d = ig_total_14d + tt_total_14d

    follower_day_14 = np.nan
    if follower_col is not None:
        follower_day_14 = float(win[follower_col].iloc[-1]) if not pd.isna(win[follower_col].iloc[-1]) else np.nan

    return {
        "IG_Rolling_7D_vs_Previous_7D": _safe_ratio(ig_last7, ig_prev7),
        "TT_Rolling_7D_vs_Previous_7D": _safe_ratio(tt_last7, tt_prev7),
        "log_IG_Total_Eng_14d_plus1": float(np.log1p(max(ig_total_14d, 0.0))),
        "log_TT_Total_Eng_14d_plus1": float(np.log1p(max(tt_total_14d, 0.0))),
        "TT_Share_of_Total": _safe_ratio(tt_total_14d, total_14d),
        "log_Absolute_Follower_Count_Day_14_plus1": float(
            np.log1p(max(follower_day_14, 0.0)) if not np.isnan(follower_day_14) else 0.0
        ),
    }


def build_modeling_dataset() -> DatasetBundle:
    releases = load_release_cohort()
    socials, follower_col = load_socials()
    firstweek = load_first_week_actuals_and_artist()
    history = build_release_history_features(firstweek)

    socials_by_id = {
        sid: g.sort_values("DATE")[["DATE", "IG_ENGAGEMENT", "TT_ENGAGEMENT"] + ([follower_col] if follower_col else [])]
        for sid, g in socials.groupby("SODATONE_ARTIST_ID", sort=False)
    }

    social_rows: List[Dict[str, float]] = []
    for row in releases.itertuples(index=False):
        sid = str(row.SODATONE_ARTIST_ID).strip()
        if not sid:
            continue
        feats = compute_social_features_for_release(
            sid=sid,
            sale_date=row.FIRST_SALE_DATE,
            socials_by_id=socials_by_id,
            follower_col=follower_col,
        )
        if not feats:
            continue
        feats["MRELG_ID"] = row.MRELG_ID
        feats["SODATONE_ARTIST_ID"] = sid
        feats["FIRST_SALE_DATE"] = row.FIRST_SALE_DATE
        social_rows.append(feats)

    social_df = pd.DataFrame(social_rows)
    if social_df.empty:
        raise RuntimeError("No rows had complete 14-day pre-release socials coverage.")

    model_df = social_df.merge(history, on="MRELG_ID", how="inner", suffixes=("", "_HIST"))
    model_df = model_df.dropna(
        subset=[
            "FIRST_WEEK_TOTAL_AE",
            "LAG_1_FIRST_WEEK_AE",
            "PRIOR_AVG_FIRST_WEEK_AE",
            "DAYS_SINCE_LAST_RELEASE",
        ]
    )
    model_df = model_df[model_df["PRIOR_AVG_FIRST_WEEK_AE"] > PRIOR_AVG_THRESHOLD].copy()

    if model_df.empty:
        raise RuntimeError("No rows left after prior average threshold filter.")

    model_df["TARGET_LOG_FIRST_WEEK"] = np.log1p(model_df["FIRST_WEEK_TOTAL_AE"].clip(lower=0.0))
    model_df["log_Lag_1_First_Week_AE_plus1"] = np.log1p(model_df["LAG_1_FIRST_WEEK_AE"].clip(lower=0.0))
    model_df["log_Prior_Avg_First_Week_AE_plus1"] = np.log1p(model_df["PRIOR_AVG_FIRST_WEEK_AE"].clip(lower=0.0))
    model_df["log_Days_Since_Last_Release_plus1"] = np.log1p(
        model_df["DAYS_SINCE_LAST_RELEASE"].clip(lower=0.0)
    )

    model_df = model_df.sort_values("FIRST_SALE_DATE").reset_index(drop=True)

    feature_cols = [
        "log_Lag_1_First_Week_AE_plus1",
        "log_Prior_Avg_First_Week_AE_plus1",
        "log_Days_Since_Last_Release_plus1",
        "IG_Rolling_7D_vs_Previous_7D",
        "TT_Rolling_7D_vs_Previous_7D",
        "log_IG_Total_Eng_14d_plus1",
        "log_TT_Total_Eng_14d_plus1",
        "TT_Share_of_Total",
        "log_Absolute_Follower_Count_Day_14_plus1",
    ]
    return DatasetBundle(data=model_df, feature_cols=feature_cols)


def time_split(df: pd.DataFrame, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = max(1, int(n * (1 - test_frac)))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/test split produced an empty partition.")
    return train_df, test_df


def evaluate_predictions(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "r2": float(r2_score(actual, pred)),
    }


def main() -> None:
    ds = build_modeling_dataset()
    df = ds.data
    train_df, test_df = time_split(df, TEST_FRAC)

    x_train = train_df[ds.feature_cols]
    y_train_log = train_df["TARGET_LOG_FIRST_WEEK"]
    x_test = test_df[ds.feature_cols]
    y_test_log = test_df["TARGET_LOG_FIRST_WEEK"]

    model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        reg_alpha=1.0,
        reg_lambda=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(x_train, y_train_log)

    pred_log = pd.Series(model.predict(x_test), index=test_df.index)

    baseline_log = test_df["log_Lag_1_First_Week_AE_plus1"]
    log_metrics = evaluate_predictions(y_test_log, pred_log)
    log_baseline_metrics = evaluate_predictions(y_test_log, baseline_log)

    actual_units = test_df["FIRST_WEEK_TOTAL_AE"]
    pred_units = np.expm1(pred_log).clip(lower=0.0)
    baseline_units = test_df["LAG_1_FIRST_WEEK_AE"].clip(lower=0.0)
    unit_metrics = evaluate_predictions(actual_units, pred_units)
    unit_baseline_metrics = evaluate_predictions(actual_units, baseline_units)

    print("=== Rebuilt First-Week LGBM (Log Target) ===")
    print(f"Cohort min first sale date: {COHORT_MIN_DATE.date()}")
    print(f"Filter: PRIOR_AVG_FIRST_WEEK_AE > {PRIOR_AVG_THRESHOLD:,.0f}")
    print(f"Rows in modeling dataset: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print()
    print("Target (log units) performance:")
    print(f"Model MAE:  {log_metrics['mae']:.4f}")
    print(f"Model RMSE: {log_metrics['rmse']:.4f}")
    print(f"Model R2:   {log_metrics['r2']:.4f}")
    print(f"Lag-1 baseline MAE:  {log_baseline_metrics['mae']:.4f}")
    print(f"Lag-1 baseline RMSE: {log_baseline_metrics['rmse']:.4f}")
    print(f"Lag-1 baseline R2:   {log_baseline_metrics['r2']:.4f}")
    print()
    print("Units-scale performance:")
    print(f"Model MAE:  {unit_metrics['mae']:,.2f}")
    print(f"Model RMSE: {unit_metrics['rmse']:,.2f}")
    print(f"Lag-1 baseline MAE:  {unit_baseline_metrics['mae']:,.2f}")
    print(f"Lag-1 baseline RMSE: {unit_baseline_metrics['rmse']:,.2f}")
    print(
        f"MAE delta (model - baseline): {(unit_metrics['mae'] - unit_baseline_metrics['mae']):,.2f}"
    )
    print(
        f"RMSE delta (model - baseline): {(unit_metrics['rmse'] - unit_baseline_metrics['rmse']):,.2f}"
    )
    print()

    importances = pd.Series(model.feature_importances_, index=ds.feature_cols).sort_values(ascending=False)
    print("Top feature importances:")
    for feat, val in importances.items():
        print(f"{feat}: {val:.2f}")

    print()
    print("Example test predictions (first 10 rows):")
    preview = test_df[
        [
            "MRELG_ID",
            "SODATONE_ARTIST_ID",
            "FIRST_SALE_DATE",
            "FIRST_WEEK_TOTAL_AE",
            "LAG_1_FIRST_WEEK_AE",
        ]
    ].copy()
    preview["PRED_LOG_TARGET"] = pred_log
    preview["PRED_FIRST_WEEK_UNITS"] = pred_units
    print(preview.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
