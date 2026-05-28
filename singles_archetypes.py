#!/usr/bin/env python3
"""
Lead-single pre-album meta-archetype pipeline (album-level).

Phase 1:
  - Join weekly singles to lead-single -> album mapping.
  - Keep rows with WEEK_END_DATE < ALBUM_FIRST_SALE_DATE.
  - Require >= MIN_OBSERVED_WEEKS per single.

Phase 2 (meta-archetyping with 5-step architecture):
  Pre-album rollout curves always use STREAMING_EQUIVALENT on singles.
  Album first-week labels differ by --target (total_ae / product / song); product and
  song artist baselines + cluster probs come from data/sales or data/songs only.
  Step 1: True shape normalization via M_norm(t) (unit-weight stacking).
  Step 2: Cannibalization ratio from gamma residuals on overlapping tracks.
  Step 3: Late-promo drop bypass for tracks with <2 observed weeks.
  Step 4: Expected week-1 carryover via integral of M(t) over [0,1].
  Step 5: Catalog momentum slope from artist_ae_baseline.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from all_data_archetypes_simulator_ae import extract_main_genre, gamma_norm
from historical_single_momentum import attach_historical_single_momentum
from target_config import (
    BASE_ABSOLUTE_EXPORT_COLS,
    CLUSTER_FEATURE_COLS,
    MOMENTUM_BASELINE_INTERACTION_COL,
    MOMENTUM_GROWTH_RATIO_COL,
    TargetConfig,
    add_target_cli,
    compute_momentum_growth_ratio,
    get_target_config,
    merge_target_artist_artifacts,
    validate_target_data_dir,
)


LEAD_SINGLES_PATH = Path("data/lead_singles_with_album_within_6mo.csv")
SINGLES_WEEKLY_PATH = Path("data/78_pre_album_singles_weekly.parquet")
ARTIST_BASELINE_PATH = Path("data/artist_ae_baseline.parquet")
HISTORICAL_ALBUMS_PATH = Path("data/firstweekae.csv")
PHASE1_PANEL_PATH = Path("data/pre_album_singles_weekly_panel.parquet")
MIN_OBSERVED_WEEKS = 2
HORIZON_WEEKS = 27
EPS = 1e-9

# Step 3: Fallback template for late promo drops (<2 observed weeks).
LATE_PROMO_FALLBACK_A = 1.5
LATE_PROMO_FALLBACK_T_PEAK = 1.0

def compute_week_index_pre_album(
    df: pd.DataFrame, horizon_weeks: int, metric_col: str
) -> pd.DataFrame:
    """Single-relative week index (1..horizon) for gamma fitting."""
    out = df.copy()
    out["WEEK_END_DATE"] = pd.to_datetime(out["WEEK_END_DATE"], errors="coerce")
    out["TARGET_METRIC"] = pd.to_numeric(out[metric_col], errors="coerce").fillna(0.0)

    first_dates = out.groupby("MRELG_ID")["WEEK_END_DATE"].transform("min")
    weeks_since_release = ((out["WEEK_END_DATE"] - first_dates).dt.days / 7).round().astype(int)
    out["WEEKS_SINCE_RELEASE"] = weeks_since_release

    min_week = out.groupby("MRELG_ID")["WEEKS_SINCE_RELEASE"].transform("min")
    out["week"] = (out["WEEKS_SINCE_RELEASE"] - min_week) + 1
    out = out[(out["week"] >= 1) & (out["week"] <= horizon_weeks)].copy()
    out["week"] = out["week"].astype(int)
    return out


def gamma_norm_broadcast(
    t: np.ndarray, a: np.ndarray, t_peak: np.ndarray
) -> np.ndarray:
    """Vectorized gamma_norm for broadcast shapes (T, n_tracks)."""
    ratio = np.clip(t / np.maximum(t_peak, EPS), EPS, None)
    return (ratio ** a) * np.exp(-a * (ratio - 1.0))


def fit_track_gamma_params(
    df: pd.DataFrame,
    horizon_weeks: int,
) -> pd.DataFrame:
    """
    Fit gamma_norm(a, t_peak) per pre-release single.

    Step 3 (right-censorship): tracks with <2 observed weeks get the late-promo
    fallback template instead of a dynamic fit.
    """
    meta = (
        df.groupby("MRELG_ID", sort=False)
        .agg(
            ALBUM_MRELG_ID=("ALBUM_MRELG_ID", "first"),
            ALBUM_FIRST_SALE_DATE=("ALBUM_FIRST_SALE_DATE", "first"),
            DISPLAY_ARTIST=("DISPLAY_ARTIST", "first"),
            days_to_album_drop=("days_to_album_drop", "first"),
            weeks_observed_pre_album=("weeks_observed_pre_album", "first"),
        )
        .reset_index()
    )

    weekly = df.groupby(["MRELG_ID", "week"], sort=False)["TARGET_METRIC"].mean().reset_index()

    records: List[Dict[str, Any]] = []
    for mrelg_id, grp in weekly.groupby("MRELG_ID", sort=False):
        g = grp.sort_values("week")
        w = g["week"].to_numpy(dtype=float)
        y = g["TARGET_METRIC"].to_numpy(dtype=float)

        peak_idx = int(np.argmax(y))
        peak_y = float(y[peak_idx])
        if not np.isfinite(peak_y) or peak_y <= 0:
            continue

        # Step 3: bypass for late-promo drops
        if len(w) < 2:
            records.append({
                "MRELG_ID": mrelg_id,
                "peak_volume_V": peak_y,
                "gamma_a": LATE_PROMO_FALLBACK_A,
                "gamma_t_peak": LATE_PROMO_FALLBACK_T_PEAK,
                "peak_week_obs": float(w[peak_idx]),
                "is_late_promo": True,
            })
            continue

        y_norm = y / peak_y
        t_peak0 = float(w[peak_idx])
        a0 = 2.0

        if len(w) >= 5:
            try:
                popt, _ = curve_fit(
                    f=gamma_norm,
                    xdata=w,
                    ydata=y_norm,
                    p0=[a0, t_peak0],
                    bounds=([0.2, 1.0], [20.0, float(horizon_weeks)]),
                    maxfev=10000,
                )
                a_hat, t_peak_hat = float(popt[0]), float(popt[1])
            except Exception:
                a_hat, t_peak_hat = a0, t_peak0
        else:
            a_hat, t_peak_hat = a0, t_peak0

        records.append({
            "MRELG_ID": mrelg_id,
            "peak_volume_V": peak_y,
            "gamma_a": a_hat,
            "gamma_t_peak": t_peak_hat,
            "peak_week_obs": t_peak0,
            "is_late_promo": False,
        })

    if not records:
        raise ValueError("No track-level gamma fits produced.")

    tracks = pd.DataFrame.from_records(records).merge(meta, on="MRELG_ID", how="inner")
    tracks["delta_weeks"] = tracks["days_to_album_drop"].astype(float) / 7.0
    return tracks


def build_album_time_grid(max_pre_weeks: int = HORIZON_WEEKS) -> np.ndarray:
    """Album-relative week grid: negative pre-release weeks through t=0."""
    return np.arange(-int(max_pre_weeks), 1, dtype=float)


def _stack_curves_for_album(
    delta: np.ndarray,
    a: np.ndarray,
    t_peak: np.ndarray,
    v: np.ndarray,
    t_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (M_abs, M_norm) for one album.
    M_abs = sum V_i * gamma_i(...)   (absolute)
    M_norm = sum 1.0 * gamma_i(...)  (unit-weight, pure shape)
    """
    w = t_grid[:, None] + delta[None, :]  # (T, n)
    valid = w >= 1.0
    safe_w = np.where(valid, w, 1.0)
    gamma_vals = gamma_norm_broadcast(safe_w, a[None, :], t_peak[None, :])
    gamma_masked = np.where(valid, gamma_vals, 0.0)

    m_abs = (gamma_masked * v[None, :]).sum(axis=1)
    m_norm = gamma_masked.sum(axis=1)  # unit-weight (Step 1)
    return m_abs, m_norm


def stack_composite_momentum(
    track_params: pd.DataFrame,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Stack per-track gamma curves into album composite M(t) and M_norm(t).
    M(t)      = sum_i V_i * gamma_i(t + delta_i)   [absolute]
    M_norm(t) = sum_i 1   * gamma_i(t + delta_i)   [pure shape, Step 1]
    """
    curve_rows: List[Dict[str, Any]] = []
    for album_id, grp in track_params.groupby("ALBUM_MRELG_ID", sort=False):
        delta = grp["delta_weeks"].to_numpy(dtype=float)
        a = grp["gamma_a"].to_numpy(dtype=float)
        t_peak = grp["gamma_t_peak"].to_numpy(dtype=float)
        v = grp["peak_volume_V"].to_numpy(dtype=float)

        m_abs, m_norm = _stack_curves_for_album(delta, a, t_peak, v, t_grid)
        curve_rows.extend(
            {
                "ALBUM_MRELG_ID": album_id,
                "album_week_t": float(t),
                "composite_momentum": float(ma),
                "composite_momentum_norm": float(mn),
            }
            for t, ma, mn in zip(t_grid, m_abs, m_norm)
        )

    return pd.DataFrame.from_records(curve_rows)


def compute_cannibalization_ratio(
    df: pd.DataFrame,
    track_params: pd.DataFrame,
) -> pd.DataFrame:
    """
    Step 2: For each album, measure how subsequent singles interact.
    When single_j launches, compute residual on all prior singles i during that week:
        residual_i = observed_i(week_j_launch) - expected_gamma_i(week_j_launch)
    Aggregate mean residual per album -> cannibalization_ratio.
    Negative = cannibalization; positive = synergy/halo.
    """
    track_params_sorted = track_params.sort_values(
        ["ALBUM_MRELG_ID", "delta_weeks"], ascending=[True, False]
    ).copy()

    album_residuals: Dict[str, List[float]] = {}

    # Weekly observed data indexed for fast lookup
    weekly = (
        df.groupby(["MRELG_ID", "week"], sort=False)["TARGET_METRIC"]
        .mean()
        .reset_index()
    )
    obs_lookup: Dict[str, Dict[int, float]] = {}
    for mrelg_id, grp in weekly.groupby("MRELG_ID", sort=False):
        obs_lookup[mrelg_id] = dict(zip(grp["week"].astype(int), grp["TARGET_METRIC"].astype(float)))

    for album_id, album_tracks in track_params_sorted.groupby("ALBUM_MRELG_ID", sort=False):
        if len(album_tracks) < 2:
            album_residuals[album_id] = [0.0]
            continue

        tracks_list = album_tracks.sort_values("delta_weeks", ascending=False).to_dict("records")
        residuals: List[float] = []

        for j_idx in range(1, len(tracks_list)):
            track_j = tracks_list[j_idx]
            delta_j = track_j["delta_weeks"]

            for i_idx in range(j_idx):
                track_i = tracks_list[i_idx]
                delta_i = track_i["delta_weeks"]
                # When track_j drops, track_i has been out for (delta_i - delta_j) weeks
                week_i_at_j_launch = max(1.0, delta_i - delta_j)
                week_int = int(round(week_i_at_j_launch))

                obs_dict = obs_lookup.get(track_i["MRELG_ID"], {})
                observed = obs_dict.get(week_int)
                if observed is None:
                    continue

                expected = track_i["peak_volume_V"] * float(
                    gamma_norm(np.array([week_i_at_j_launch]), track_i["gamma_a"], track_i["gamma_t_peak"])[0]
                )
                if expected > EPS:
                    residuals.append((observed - expected) / expected)

        album_residuals[album_id] = residuals if residuals else [0.0]

    result = pd.DataFrame({
        "ALBUM_MRELG_ID": list(album_residuals.keys()),
        "cannibalization_ratio": [float(np.mean(v)) for v in album_residuals.values()],
    })
    return result


def _auc_on_grid(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> float:
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(y[mask], t[mask]))


def _value_at_t(t: np.ndarray, y: np.ndarray, target: float) -> float:
    idx = np.where(t == target)[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(t - target)))])
    return float(y[idx[0]])


def extract_rollout_meta_features(
    t: np.ndarray,
    m_abs: np.ndarray,
    m_norm: np.ndarray,
    count_tracks: int,
    max_single_peak: float,
) -> Dict[str, float]:
    """
    Extract features from both absolute and normalized composite curves.
    Shape features from M_norm; magnitude features from M_abs.
    """
    m_a = np.clip(m_abs.astype(float), 0.0, None)
    m_n = np.clip(m_norm.astype(float), 0.0, None)

    # --- Absolute features from M_abs ---
    total_auc_abs = _auc_on_grid(t, m_a, float(t.min()), 0.0)
    terminal_velocity_abs = (_value_at_t(t, m_a, 0.0) - _value_at_t(t, m_a, -2.0)) / 2.0
    composite_peak_abs = float(np.max(m_a))

    # Step 4: expected week-1 carryover = integral M_abs over [0, 1]
    expected_week1_carryover = _auc_on_grid(t, m_a, 0.0, 1.0)
    # If grid doesn't extend past 0, estimate from M(0) alone
    if expected_week1_carryover == 0.0:
        expected_week1_carryover = float(_value_at_t(t, m_a, 0.0))

    # --- Shape features from M_norm (Step 1: truly dimensionless) ---
    norm_peak = float(np.max(m_n))
    if norm_peak < EPS:
        norm_peak = EPS

    total_auc_norm = _auc_on_grid(t, m_n, float(t.min()), 0.0)
    final_4w_auc_norm = _auc_on_grid(t, m_n, -4.0, 0.0)
    hype_concentration = float(final_4w_auc_norm / max(total_auc_norm, EPS))

    norm_terminal_velocity = (
        (_value_at_t(t, m_n, 0.0) - _value_at_t(t, m_n, -2.0)) / (2.0 * norm_peak)
    )
    norm_total_auc = total_auc_norm / norm_peak

    peak_idx = int(np.argmax(m_n))
    peak_proximity_weeks = float(0.0 - t[peak_idx])

    return {
        # Absolute (downstream XGBoost)
        "count_pre_release_tracks": float(count_tracks),
        "max_single_peak_volume": float(max_single_peak),
        "total_pre_release_auc": total_auc_abs,
        "terminal_velocity": terminal_velocity_abs,
        "composite_peak_momentum": composite_peak_abs,
        "expected_week1_carryover": expected_week1_carryover,
        # Shape (dimensionless, for KMeans)
        "norm_terminal_velocity": norm_terminal_velocity,
        "norm_total_auc": norm_total_auc,
        "hype_concentration": hype_concentration,
        "peak_proximity_weeks": peak_proximity_weeks,
    }


def build_album_meta_features(
    track_params: pd.DataFrame,
    t_grid: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct composite curves and per-album rollout meta-features."""
    curves = stack_composite_momentum(track_params, t_grid)

    album_feats: List[Dict[str, Any]] = []
    for album_id, c_df in curves.groupby("ALBUM_MRELG_ID", sort=False):
        t = c_df["album_week_t"].to_numpy(dtype=float)
        m_abs = c_df["composite_momentum"].to_numpy(dtype=float)
        m_norm = c_df["composite_momentum_norm"].to_numpy(dtype=float)
        grp_tracks = track_params.loc[track_params["ALBUM_MRELG_ID"] == album_id]
        n_tracks = int(len(grp_tracks))
        max_v = float(grp_tracks["peak_volume_V"].max()) if n_tracks else 0.0

        feats = extract_rollout_meta_features(t, m_abs, m_norm, n_tracks, max_v)
        feats["ALBUM_MRELG_ID"] = album_id
        album_feats.append(feats)

    album_df = pd.DataFrame.from_records(album_feats)
    album_meta = (
        track_params.groupby("ALBUM_MRELG_ID", sort=False)
        .agg(
            ALBUM_FIRST_SALE_DATE=("ALBUM_FIRST_SALE_DATE", "first"),
            DISPLAY_ARTIST=("DISPLAY_ARTIST", "first"),
        )
        .reset_index()
    )
    album_df = album_df.merge(album_meta, on="ALBUM_MRELG_ID", how="left")
    return album_df, curves


def compute_catalog_momentum_slope(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Isolate true catalog volume and compute its slope between t=-8 and t=-2.

    catalog_volume = artist_baseline_ae - sum(active_single_ae) for each week.
    Slope is fit by linear regression on catalog_volume over album-relative weeks -8..-2.
    """
    if not ARTIST_BASELINE_PATH.exists():
        print(f"  Warning: {ARTIST_BASELINE_PATH} not found; catalog_momentum_slope set to 0.")
        albums = panel["ALBUM_MRELG_ID"].unique()
        return pd.DataFrame({"ALBUM_MRELG_ID": albums, "catalog_momentum_slope": 0.0})

    baseline = pd.read_parquet(ARTIST_BASELINE_PATH)
    baseline["WEEK_END_DATE"] = pd.to_datetime(baseline["WEEK_END_DATE"], errors="coerce")
    baseline["TOTAL_ALBUM_EQUIVALENTS"] = pd.to_numeric(
        baseline["TOTAL_ALBUM_EQUIVALENTS"], errors="coerce"
    ).fillna(0.0)
    baseline["DISPLAY_ARTIST"] = baseline["DISPLAY_ARTIST"].astype(str).str.strip()
    baseline = baseline.rename(columns={"TOTAL_ALBUM_EQUIVALENTS": "baseline_ae"})

    # Build album-relative week for each panel row
    p = panel[["MRELG_ID", "ALBUM_MRELG_ID", "ALBUM_FIRST_SALE_DATE",
               "DISPLAY_ARTIST", "WEEK_END_DATE"]].copy()
    p["WEEK_END_DATE"] = pd.to_datetime(p["WEEK_END_DATE"], errors="coerce")
    p["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(p["ALBUM_FIRST_SALE_DATE"], errors="coerce")
    p["album_rel_week"] = ((p["WEEK_END_DATE"] - p["ALBUM_FIRST_SALE_DATE"]).dt.days / 7.0).round().astype(int)

    # Merge baseline for artist + week
    p = p.merge(
        baseline[["DISPLAY_ARTIST", "WEEK_END_DATE", "baseline_ae"]],
        on=["DISPLAY_ARTIST", "WEEK_END_DATE"],
        how="left",
    )
    p["baseline_ae"] = p["baseline_ae"].fillna(0.0)

    # Sum single streaming per album + week (singles do not carry product sales).
    metric_col = None
    for cand in ("STREAMING_EQUIVALENT", "TOTAL_ALBUM_EQUIVALENTS"):
        if cand in panel.columns:
            metric_col = cand
            break
    if metric_col is not None:
        weekly_single_vol = (
            panel.groupby(["ALBUM_MRELG_ID", "WEEK_END_DATE"], sort=False)[metric_col]
            .sum()
            .reset_index()
            .rename(columns={metric_col: "singles_ae_sum"})
        )
        weekly_single_vol["WEEK_END_DATE"] = pd.to_datetime(weekly_single_vol["WEEK_END_DATE"], errors="coerce")
        p = p.merge(weekly_single_vol, on=["ALBUM_MRELG_ID", "WEEK_END_DATE"], how="left")
        p["singles_ae_sum"] = p["singles_ae_sum"].fillna(0.0)
    else:
        p["singles_ae_sum"] = 0.0

    p["catalog_volume"] = (p["baseline_ae"] - p["singles_ae_sum"]).clip(lower=0.0)

    # Slope between t=-8 and t=-2
    window = p[(p["album_rel_week"] >= -8) & (p["album_rel_week"] <= -2)].copy()
    agg = (
        window.groupby(["ALBUM_MRELG_ID", "album_rel_week"], sort=False)["catalog_volume"]
        .mean()
        .reset_index()
    )

    slopes: Dict[str, float] = {}
    for album_id, grp in agg.groupby("ALBUM_MRELG_ID", sort=False):
        if len(grp) < 2:
            slopes[album_id] = 0.0
            continue
        x = grp["album_rel_week"].to_numpy(dtype=float)
        y = grp["catalog_volume"].to_numpy(dtype=float)
        # np.polyfit vectorized per album
        coeffs = np.polyfit(x, y, 1)
        slopes[album_id] = float(coeffs[0])

    return pd.DataFrame({
        "ALBUM_MRELG_ID": list(slopes.keys()),
        "catalog_momentum_slope": list(slopes.values()),
    })


def fit_album_archetype_clusters(
    album_df: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], StandardScaler, SimpleImputer, MiniBatchKMeans]:
    """Cluster albums strictly on dimensionless shape features."""
    X = album_df[CLUSTER_FEATURE_COLS].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init="auto",
    )
    km.fit(X_scaled)

    out = album_df.copy()
    out["Archetype_Cluster"] = km.labels_.astype(int)
    model_info = {
        "cluster_feature_cols": CLUSTER_FEATURE_COLS,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "clustering_level": "album_meta_archetype_v2",
    }
    return out, model_info, scaler, imputer, km


def fit_cluster_composite_curves(
    curves: pd.DataFrame,
    albums_with_clusters: pd.DataFrame,
    n_clusters: int,
) -> Dict[str, Any]:
    """Median composite M(t) per cluster -> gamma_norm parameters."""
    merged = curves.merge(
        albums_with_clusters[["ALBUM_MRELG_ID", "Archetype_Cluster"]],
        on="ALBUM_MRELG_ID",
        how="inner",
    )
    params: Dict[str, Any] = {}
    for c in range(n_clusters):
        sub = merged.loc[merged["Archetype_Cluster"] == c]
        if sub.empty:
            continue
        med = (
            sub.groupby("album_week_t", sort=False)["composite_momentum"]
            .median()
            .reset_index()
        )
        t_fit = med["album_week_t"].to_numpy(dtype=float)
        y_fit = med["composite_momentum"].to_numpy(dtype=float)
        y_max = float(np.nanmax(y_fit))
        if not np.isfinite(y_max) or y_max <= 0 or len(t_fit) < 5:
            continue
        y_norm = y_fit / y_max
        t_pos = t_fit - float(t_fit.min()) + 1.0
        peak_idx = int(np.nanargmax(y_norm))
        try:
            popt, _ = curve_fit(
                gamma_norm,
                xdata=t_pos,
                ydata=y_norm,
                p0=[2.0, float(t_pos[peak_idx])],
                bounds=([0.2, 1.0], [1.0, float(max(t_pos.max(), 2.0))]),
                maxfev=10000,
            )
            a_hat, t_peak_hat = float(popt[0]), float(popt[1])
        except Exception:
            a_hat, t_peak_hat = 2.0, float(t_pos[peak_idx])
        params[str(c)] = {"a": a_hat, "t_peak": t_peak_hat}
    return params


def compute_historical_baselines(album_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each album, compute the artist's max historical first-week volume
    from albums released strictly before ALBUM_FIRST_SALE_DATE.

    Returns album_df with two new columns:
      - max_historical_week1_volume: max FIRST_WEEK_TOTAL_AE from prior albums
      - is_debut_studio_album: 1 if no prior albums exist, 0 otherwise
    """
    if not HISTORICAL_ALBUMS_PATH.exists():
        print(f"  Warning: {HISTORICAL_ALBUMS_PATH} not found; filling defaults.")
        album_df = album_df.copy()
        album_df["max_historical_week1_volume"] = 0.0
        album_df["is_debut_studio_album"] = 1
        return album_df

    hist = pd.read_csv(
        HISTORICAL_ALBUMS_PATH,
        encoding="utf-8-sig",
        usecols=["MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"],
    )
    hist["DISPLAY_ARTIST"] = hist["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    hist["FIRST_SALE_DATE"] = pd.to_datetime(hist["FIRST_SALE_DATE"], errors="coerce")
    hist["FIRST_WEEK_TOTAL_AE"] = pd.to_numeric(hist["FIRST_WEEK_TOTAL_AE"], errors="coerce")
    hist = hist.dropna(subset=["FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"])

    album_df = album_df.copy()
    album_df["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(
        album_df["ALBUM_FIRST_SALE_DATE"], errors="coerce"
    )
    album_df["DISPLAY_ARTIST"] = album_df["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()

    # Build per-artist sorted history for vectorized max-before-date lookup
    hist_sorted = hist.sort_values(["DISPLAY_ARTIST", "FIRST_SALE_DATE"]).reset_index(drop=True)

    # Group historical max by artist + date using a merge + filter approach
    merged = album_df[["ALBUM_MRELG_ID", "DISPLAY_ARTIST", "ALBUM_FIRST_SALE_DATE"]].merge(
        hist_sorted[["DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"]],
        on="DISPLAY_ARTIST",
        how="left",
    )
    # Only keep historical records strictly before the album's sale date
    merged = merged[merged["FIRST_SALE_DATE"] < merged["ALBUM_FIRST_SALE_DATE"]]

    # Aggregate max per album
    baselines = (
        merged.groupby("ALBUM_MRELG_ID", sort=False)["FIRST_WEEK_TOTAL_AE"]
        .max()
        .reset_index()
        .rename(columns={"FIRST_WEEK_TOTAL_AE": "max_historical_week1_volume"})
    )

    album_df = album_df.merge(baselines, on="ALBUM_MRELG_ID", how="left")
    album_df["max_historical_week1_volume"] = album_df["max_historical_week1_volume"].fillna(0.0)
    album_df["is_debut_studio_album"] = (album_df["max_historical_week1_volume"] == 0.0).astype(int)
    return album_df


def build_downstream_export_df(
    albums_with_clusters: pd.DataFrame,
    absolute_export_cols: List[str],
) -> pd.DataFrame:
    """Album-level export: cluster label + absolute features + shape features."""
    ordered = [
        "ALBUM_MRELG_ID",
        "Archetype_Cluster",
        "DISPLAY_ARTIST",
        "ALBUM_FIRST_SALE_DATE",
        *absolute_export_cols,
        *CLUSTER_FEATURE_COLS,
    ]
    keep = list(dict.fromkeys(c for c in ordered if c in albums_with_clusters.columns))
    return albums_with_clusters.loc[:, keep].copy()


def phase1_build_panel(output_path: Path = PHASE1_PANEL_PATH) -> pd.DataFrame:
    """Filter singles weekly rows to pre-album window with minimum week coverage."""
    lead = pd.read_csv(LEAD_SINGLES_PATH, encoding="utf-8-sig")

    # Standardize Snowflake columns to uppercase to avoid case-sensitivity errors
    lead.columns = [c.upper() for c in lead.columns]

    # Map the new SQL schema to what the legacy pipeline expects
    if "MRELG_ID_ALBUM" in lead.columns and "MRELG_ID" in lead.columns:
        lead = lead.rename(columns={
            "MRELG_ID": "SINGLE_MRELG_ID",
            "MRELG_ID_ALBUM": "ALBUM_MRELG_ID",
            "FIRST_SALE_DATE_ALBUM": "ALBUM_FIRST_SALE_DATE",
        })
        if "DAYS_TO_NEXT_ALBUM" not in lead.columns:
            single_date = pd.to_datetime(lead.get("FIRST_SALE_DATE"), errors="coerce")
            album_date = pd.to_datetime(lead["ALBUM_FIRST_SALE_DATE"], errors="coerce")
            lead["DAYS_TO_NEXT_ALBUM"] = (album_date - single_date).dt.days

    required = {"SINGLE_MRELG_ID", "ALBUM_MRELG_ID", "ALBUM_FIRST_SALE_DATE", "DAYS_TO_NEXT_ALBUM"}
    missing = required - set(lead.columns)
    if missing:
        raise ValueError(f"{LEAD_SINGLES_PATH} missing columns: {sorted(missing)}")

    mapping = lead[list(required)].copy()
    mapping = mapping.rename(columns={"SINGLE_MRELG_ID": "MRELG_ID"})
    mapping["MRELG_ID"] = mapping["MRELG_ID"].astype(str).str.strip()
    mapping["ALBUM_MRELG_ID"] = mapping["ALBUM_MRELG_ID"].astype(str).str.strip()
    mapping["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(mapping["ALBUM_FIRST_SALE_DATE"], errors="coerce")
    mapping["DAYS_TO_NEXT_ALBUM"] = pd.to_numeric(mapping["DAYS_TO_NEXT_ALBUM"], errors="coerce")
    mapping = mapping.dropna(subset=["MRELG_ID", "ALBUM_FIRST_SALE_DATE"])
    mapping = mapping.drop_duplicates(subset=["MRELG_ID"], keep="first")

    singles = pd.read_parquet(SINGLES_WEEKLY_PATH)
    singles["MRELG_ID"] = singles["MRELG_ID"].astype(str).str.strip()
    singles["WEEK_END_DATE"] = pd.to_datetime(singles["WEEK_END_DATE"], errors="coerce")

    panel = singles.merge(mapping, on="MRELG_ID", how="inner", suffixes=("", "_MAP"))
    panel = panel[panel["WEEK_END_DATE"] < panel["ALBUM_FIRST_SALE_DATE"]].copy()

    week_counts = (
        panel.groupby("MRELG_ID", sort=False)
        .agg(
            weeks_observed_pre_album=("WEEK_END_DATE", "nunique"),
            days_to_album_drop=("DAYS_TO_NEXT_ALBUM", "first"),
        )
        .reset_index()
    )

    valid_ids = week_counts.loc[
        week_counts["weeks_observed_pre_album"] >= MIN_OBSERVED_WEEKS, "MRELG_ID"
    ]
    panel = panel[panel["MRELG_ID"].isin(valid_ids)].copy()
    panel = panel.merge(
        week_counts[["MRELG_ID", "weeks_observed_pre_album", "days_to_album_drop"]],
        on="MRELG_ID",
        how="left",
    )

    panel = panel.sort_values(
        ["MRELG_ID", "WEEKS_SINCE_RELEASE", "WEEK_END_DATE"], kind="mergesort"
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print("=== Phase 1: pre-album panel ===")
    print(f"Lead singles mapped: {len(mapping):,}")
    print(f"Rows after WEEK_END_DATE < ALBUM_FIRST_SALE_DATE: {len(panel):,}")
    print(f"Singles with >= {MIN_OBSERVED_WEEKS} observed weeks: {panel['MRELG_ID'].nunique():,}")
    print(f"Distinct albums in panel: {panel['ALBUM_MRELG_ID'].nunique():,}")
    print(f"Wrote: {output_path}")
    return panel


def phase2_train_metric(
    panel: pd.DataFrame,
    target_cfg: TargetConfig,
    out_dir: Path,
    *,
    n_clusters: int,
    random_state: int,
    batch_size: int,
    horizon_weeks: int,
) -> None:
    """Album-level meta-archetype training (streaming rollout + target-specific album joins)."""
    rollout_col = target_cfg.singles_rollout_metric_col
    print()
    print(
        f"=== Phase 2 (album meta-archetypes v2): "
        f"forecast={target_cfg.name} ({target_cfg.target_col}) | "
        f"singles rollout={rollout_col} ==="
    )

    df = compute_week_index_pre_album(panel, horizon_weeks=horizon_weeks, metric_col=rollout_col)
    n_singles = df["MRELG_ID"].nunique()
    pos_singles = df.groupby("MRELG_ID")["TARGET_METRIC"].max().gt(0).sum()
    print(f"Singles with positive peak ({rollout_col}): {pos_singles:,}/{n_singles:,}")
    if pos_singles == 0:
        print(
            f"ERROR: No positive values for '{rollout_col}' in the Phase 1 panel. "
            f"Check {SINGLES_WEEKLY_PATH} contains streaming data for lead singles.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Step 1+3: fitting per-track gamma curves (with late-promo bypass)...")
    track_params = fit_track_gamma_params(df, horizon_weeks=horizon_weeks)
    n_late = int(track_params["is_late_promo"].sum())
    print(f"  Track gamma fits: {len(track_params):,} (late-promo fallback: {n_late:,})")

    print("Step 2: computing cannibalization ratios...")
    cannibal_df = compute_cannibalization_ratio(df, track_params)

    t_grid = build_album_time_grid(max_pre_weeks=horizon_weeks)
    print("Step 1: stacking composite M(t) + M_norm(t) and extracting features...")
    album_df, composite_curves = build_album_meta_features(track_params, t_grid)
    print(f"  Albums with composite curves: {album_df['ALBUM_MRELG_ID'].nunique():,}")

    # Merge cannibalization
    album_df = album_df.merge(cannibal_df, on="ALBUM_MRELG_ID", how="left")
    album_df["cannibalization_ratio"] = album_df["cannibalization_ratio"].fillna(0.0)

    print("Step 5: computing catalog momentum slope...")
    catalog_df = compute_catalog_momentum_slope(panel)
    album_df = album_df.merge(catalog_df, on="ALBUM_MRELG_ID", how="left")
    album_df["catalog_momentum_slope"] = album_df["catalog_momentum_slope"].fillna(0.0)

    absolute_export_cols = list(BASE_ABSOLUTE_EXPORT_COLS)
    if target_cfg.use_firstweekae_baseline:
        print("Computing historical baselines (firstweekae.csv)...")
        album_df = compute_historical_baselines(album_df)
        absolute_export_cols.extend(
            ["max_historical_week1_volume", "is_debut_studio_album"]
        )
        n_debuts = int(album_df["is_debut_studio_album"].sum())
        print(f"  Debut albums: {n_debuts:,} / {len(album_df):,}")
    else:
        print(f"Merging target artifacts from {target_cfg.target_data_dir}...")
        album_df, extra_cols = merge_target_artist_artifacts(album_df, target_cfg)
        absolute_export_cols.extend(extra_cols)
        n_debuts = int(album_df[target_cfg.debut_col].sum())
        print(f"  Debut albums ({target_cfg.debut_col}): {n_debuts:,} / {len(album_df):,}")

    print(
        "Computing historical single momentum from prior-album lead singles "
        f"({LEAD_SINGLES_PATH})..."
    )
    album_df = attach_historical_single_momentum(
        album_df,
        lead_path=LEAD_SINGLES_PATH,
        weekly_path=SINGLES_WEEKLY_PATH,
        momentum_col=target_cfg.baseline_momentum_col,
    )
    n_with_prior = int((album_df[target_cfg.baseline_momentum_col] > 0).sum())
    print(
        f"  Albums with prior-era single momentum > 0: {n_with_prior:,} / {len(album_df):,}"
    )

    album_df[MOMENTUM_BASELINE_INTERACTION_COL] = (
        pd.to_numeric(album_df[target_cfg.baseline_max_col], errors="coerce").fillna(0.0)
        * pd.to_numeric(album_df["composite_peak_momentum"], errors="coerce").fillna(0.0)
    )
    if MOMENTUM_BASELINE_INTERACTION_COL not in absolute_export_cols:
        absolute_export_cols.append(MOMENTUM_BASELINE_INTERACTION_COL)

    mom_col = target_cfg.baseline_momentum_col
    if mom_col not in album_df.columns:
        album_df[mom_col] = 0.0
    album_df[MOMENTUM_GROWTH_RATIO_COL] = compute_momentum_growth_ratio(
        album_df["composite_peak_momentum"],
        album_df[mom_col],
    )
    if MOMENTUM_GROWTH_RATIO_COL not in absolute_export_cols:
        absolute_export_cols.append(MOMENTUM_GROWTH_RATIO_COL)

    print("Step 3B: album-level MiniBatchKMeans on shape features only...")
    albums_clustered, model_info, scaler, imputer, km = fit_album_archetype_clusters(
        album_df=album_df,
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
    )

    print("Fitting cluster-level composite curve templates...")
    archetype_params = fit_cluster_composite_curves(
        curves=composite_curves,
        albums_with_clusters=albums_clustered,
        n_clusters=n_clusters,
    )

    downstream_df = build_downstream_export_df(albums_clustered, absolute_export_cols)

    os.makedirs(out_dir, exist_ok=True)
    downstream_df.to_parquet(out_dir / "album_meta_features.parquet", index=False)
    downstream_df.to_csv(out_dir / "album_meta_features.csv", index=False)
    composite_curves.to_parquet(out_dir / "album_composite_curves.parquet", index=False)
    track_params.to_parquet(out_dir / "track_gamma_params.parquet", index=False)
    albums_clustered.to_parquet(out_dir / "album_rollout_features_full.parquet", index=False)

    with open(out_dir / "archetype_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"horizon_weeks": horizon_weeks, "archetype_params": archetype_params},
            f,
            indent=2,
        )
    with open(out_dir / "cluster_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": target_cfg.name,
                "target_col": target_cfg.target_col,
                "singles_rollout_metric_col": rollout_col,
                "horizon_weeks": horizon_weeks,
                "min_observed_weeks": MIN_OBSERVED_WEEKS,
                "model_info": model_info,
                "cluster_feature_cols": CLUSTER_FEATURE_COLS,
                "absolute_export_cols": absolute_export_cols,
                "late_promo_fallback": {
                    "a": LATE_PROMO_FALLBACK_A,
                    "t_peak": LATE_PROMO_FALLBACK_T_PEAK,
                },
                "stacking_formula": (
                    "M(t)=sum_i V_i*gamma_i(t+delta_i); "
                    "M_norm(t)=sum_i 1*gamma_i(t+delta_i)"
                ),
            },
            f,
            indent=2,
        )

    print(f"Saved artifacts: {out_dir}")
    print(f"Downstream export rows: {len(downstream_df):,}")
    print(
        "Album cluster counts:\n"
        + downstream_df["Archetype_Cluster"].value_counts().sort_index().to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lead-single pre-album album meta-archetype pipeline (v2)"
    )
    add_target_cli(parser)
    parser.add_argument("--phase", choices=("1", "2", "all"), default="all")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--panel-path", type=Path, default=PHASE1_PANEL_PATH)
    args = parser.parse_args()

    target_cfg = get_target_config(args.target)
    if not target_cfg.use_firstweekae_baseline:
        validate_target_data_dir(target_cfg)

    out_dir = target_cfg.artifacts_dir
    print(f"Target: {target_cfg.name} | Artifacts: {out_dir}")

    panel: Optional[pd.DataFrame] = None
    if args.phase in ("1", "all"):
        panel = phase1_build_panel(output_path=args.panel_path)
    elif args.phase == "2":
        if not args.panel_path.exists():
            raise FileNotFoundError(f"Phase-1 panel not found: {args.panel_path}")
        panel = pd.read_parquet(args.panel_path)

    if args.phase in ("2", "all") and panel is not None:
        phase2_train_metric(
            panel=panel,
            target_cfg=target_cfg,
            out_dir=out_dir,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            batch_size=args.kmeans_batch_size,
            horizon_weeks=HORIZON_WEEKS,
        )


if __name__ == "__main__":
    main()
