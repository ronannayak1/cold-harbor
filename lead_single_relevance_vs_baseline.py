#!/usr/bin/env python3
"""
Assess whether lead-single → album paths align with first-week volume above history baselines.

Uses:
- data/lead_singles_with_album_within_6mo.csv (single → follow-up album within 6 months)
- data/firstweekae.csv (actual first week + artist timeline)

Baselines (same definitions as socials_residual_lgbm / release history):
- BASELINE_LAG_1: prior release FIRST_WEEK_TOTAL_AE for DISPLAY_ARTIST (chronological)
- BASELINE_PRIOR_AVG: expanding mean of all prior releases for that artist

Outputs row-level flags and lifts. This does not prove the single *caused* the lift;
weekly single volumes will be added later for momentum. For now we only test
whether the album that followed a lead single beat history-based expectations.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


MATCHED_PATH = Path("data/lead_singles_with_album_within_6mo.csv")
FIRSTWEEKAE_PATH = Path("data/firstweekae.csv")
OUTPUT_PATH = Path("data/lead_singles_album_vs_baseline.csv")
BASELINE_EPS = 1.0


def load_firstweekae_history() -> pd.DataFrame:
    df = pd.read_csv(FIRSTWEEKAE_PATH, encoding="utf-8-sig")
    required = {"MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"firstweekae.csv missing columns: {sorted(missing)}")

    df = df[list(required)].copy()
    df["MRELG_ID"] = df["MRELG_ID"].astype(str).str.strip()
    df["DISPLAY_ARTIST"] = df["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df["FIRST_WEEK_TOTAL_AE"] = pd.to_numeric(df["FIRST_WEEK_TOTAL_AE"], errors="coerce")
    df = df.dropna(subset=["FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"])
    df = df[(df["MRELG_ID"] != "") & (df["DISPLAY_ARTIST"] != "")]

    df = df.sort_values(["DISPLAY_ARTIST", "FIRST_SALE_DATE", "MRELG_ID"]).copy()
    grp = df.groupby("DISPLAY_ARTIST", sort=False)
    df["BASELINE_LAG_1"] = grp["FIRST_WEEK_TOTAL_AE"].shift(1)
    df["BASELINE_PRIOR_AVG"] = grp["FIRST_WEEK_TOTAL_AE"].transform(
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
            "BASELINE_LAG_1",
            "BASELINE_PRIOR_AVG",
            "DAYS_SINCE_LAST_RELEASE",
        ]
    ]


def main() -> None:
    matched = pd.read_csv(MATCHED_PATH, encoding="utf-8-sig")
    required = {
        "SINGLE_MRELG_ID",
        "ALBUM_MRELG_ID",
        "SINGLE_FIRST_SALE_DATE",
        "ALBUM_FIRST_SALE_DATE",
    }
    missing = required - set(matched.columns)
    if missing:
        raise ValueError(f"{MATCHED_PATH} missing columns: {sorted(missing)}")

    hist = load_firstweekae_history().rename(
        columns={
            "MRELG_ID": "ALBUM_MRELG_ID",
            "DISPLAY_ARTIST": "ALBUM_DISPLAY_ARTIST_FW",
            "FIRST_SALE_DATE": "ALBUM_FIRST_SALE_DATE_FW",
            "FIRST_WEEK_TOTAL_AE": "ACTUAL_ALBUM_FIRST_WEEK_AE",
        }
    )

    out = matched.merge(hist, on="ALBUM_MRELG_ID", how="left")

    # Prefer album dates from firstweekae when present (canonical release ordering).
    out["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(out["ALBUM_FIRST_SALE_DATE"], errors="coerce")
    if "ALBUM_FIRST_SALE_DATE_FW" in out.columns:
        out["ALBUM_FIRST_SALE_DATE_FW"] = pd.to_datetime(out["ALBUM_FIRST_SALE_DATE_FW"], errors="coerce")
        out["ALBUM_FIRST_SALE_DATE_CANON"] = out["ALBUM_FIRST_SALE_DATE_FW"].fillna(
            out["ALBUM_FIRST_SALE_DATE"]
        )
    else:
        out["ALBUM_FIRST_SALE_DATE_CANON"] = out["ALBUM_FIRST_SALE_DATE"]

    has_actual = out["ACTUAL_ALBUM_FIRST_WEEK_AE"].notna()
    has_lag1 = out["BASELINE_LAG_1"].notna()
    has_prior_avg = out["BASELINE_PRIOR_AVG"].notna()

    actual = out["ACTUAL_ALBUM_FIRST_WEEK_AE"].clip(lower=0.0)
    lag1 = out["BASELINE_LAG_1"]
    prior_avg = out["BASELINE_PRIOR_AVG"]

    denom_lag = lag1.abs().clip(lower=BASELINE_EPS)
    denom_prior = prior_avg.abs().clip(lower=BASELINE_EPS)

    out["RESIDUAL_VS_LAG1"] = (actual - lag1).where(has_lag1)
    out["RESIDUAL_VS_PRIOR_AVG"] = (actual - prior_avg).where(has_prior_avg)
    out["RELATIVE_LIFT_VS_LAG1"] = ((actual - lag1) / denom_lag).where(has_lag1)
    out["RELATIVE_LIFT_VS_PRIOR_AVG"] = ((actual - prior_avg) / denom_prior).where(has_prior_avg)
    out["LOG_LIFT_VS_LAG1"] = (
        actual.map(math.log1p) - lag1.clip(lower=0.0).map(math.log1p)
    ).where(has_lag1)
    out["LOG_LIFT_VS_PRIOR_AVG"] = (
        actual.map(math.log1p) - prior_avg.clip(lower=0.0).map(math.log1p)
    ).where(has_prior_avg)

    out["ALBUM_BEAT_LAG1_BASELINE"] = has_lag1 & (actual > lag1)
    out["ALBUM_BEAT_PRIOR_AVG_BASELINE"] = has_prior_avg & (actual > prior_avg)
    out["ALBUM_BEAT_BOTH_BASELINES"] = (
        out["ALBUM_BEAT_LAG1_BASELINE"].fillna(False)
        & out["ALBUM_BEAT_PRIOR_AVG_BASELINE"].fillna(False)
    )

    # Narrative flag: album outperformed at least one usable baseline (association only).
    out["LEAD_SINGLE_PATH_ALBUM_ABOVE_EXPECTED"] = (
        out["ALBUM_BEAT_LAG1_BASELINE"].fillna(False)
        | out["ALBUM_BEAT_PRIOR_AVG_BASELINE"].fillna(False)
    )

    out.to_csv(OUTPUT_PATH, index=False)

    n = len(out)
    n_matched_fw = int(has_actual.sum())
    n_lag = int(has_lag1.sum())
    n_prior = int(has_prior_avg.sum())

    def pct_true(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) == 0:
            return float("nan")
        return 100.0 * float(s.mean())

    print("=== Lead single path vs first-week baselines ===")
    print(f"Rows in {MATCHED_PATH.name}: {n:,}")
    print(f"Albums found in firstweekae.csv: {n_matched_fw:,}")
    print(f"Rows with lag-1 baseline: {n_lag:,}")
    print(f"Rows with prior-avg baseline: {n_prior:,}")
    print()
    print("Among rows with lag-1 baseline, % beating lag-1:")
    print(f"  {pct_true(out.loc[has_lag1, 'ALBUM_BEAT_LAG1_BASELINE']):.2f}%")
    print("Among rows with prior-avg baseline, % beating prior avg:")
    print(f"  {pct_true(out.loc[has_prior_avg, 'ALBUM_BEAT_PRIOR_AVG_BASELINE']):.2f}%")
    print("Among rows with both baselines, % beating both:")
    both = has_lag1 & has_prior_avg
    if both.any():
        print(
            f"  {pct_true(out.loc[both, 'ALBUM_BEAT_BOTH_BASELINES']):.2f}%"
        )
    print()
    print(
        "Rows where album beat lag-1 OR prior avg (association with lead-single window only):"
    )
    print(f"  {pct_true(out.loc[has_actual, 'LEAD_SINGLE_PATH_ALBUM_ABOVE_EXPECTED']):.2f}% of rows with actuals")
    print()
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
