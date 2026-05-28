#!/usr/bin/env python3
"""
album_projection.py
===================
Synthesize Phase-2 album feature rows from manual single rollout inputs
for what-if / forward projection scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from all_data_archetypes_simulator_ae import gamma_norm
from singles_archetypes import (
    HISTORICAL_ALBUMS_PATH,
    build_album_time_grid,
    extract_rollout_meta_features,
    gamma_norm_broadcast,
)

FEATURE_ARTIFACTS_PATH = Path(
    "singles_artifacts/lead_pre_album/streaming_equivalent/album_meta_features.parquet"
)
EPS = 1e-9
HORIZON_WEEKS = 27


@dataclass
class ManualSingle:
    """Hypothetical pre-release single with two observed weekly volumes."""

    weeks_before_album: int  # e.g. 8 means single dropped 8 weeks before album
    week1_volume: float
    week2_volume: float


def lookup_artist_baselines(
    artist_name: str,
    album_date: pd.Timestamp,
) -> Dict[str, float]:
    """
    Historical max first-week AE (strictly before album_date) and debut flag.
    Catalog momentum: median from training artifacts for this artist, else 0.
    """
    artist_key = artist_name.strip()
    max_hist = 0.0
    is_debut = 1

    if HISTORICAL_ALBUMS_PATH.exists():
        hist = pd.read_csv(
            HISTORICAL_ALBUMS_PATH,
            encoding="utf-8-sig",
            usecols=["DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"],
        )
        hist["DISPLAY_ARTIST"] = hist["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
        hist["FIRST_SALE_DATE"] = pd.to_datetime(hist["FIRST_SALE_DATE"], errors="coerce")
        hist["FIRST_WEEK_TOTAL_AE"] = pd.to_numeric(
            hist["FIRST_WEEK_TOTAL_AE"], errors="coerce"
        )
        mask = (
            (hist["DISPLAY_ARTIST"].str.lower() == artist_key.lower())
            & (hist["FIRST_SALE_DATE"] < album_date)
        )
        past = hist.loc[mask, "FIRST_WEEK_TOTAL_AE"].dropna()
        if len(past) > 0:
            max_hist = float(past.max())
            is_debut = 0

    catalog_slope = 0.0
    archetype_cluster = 1
    if FEATURE_ARTIFACTS_PATH.exists():
        art = pd.read_parquet(FEATURE_ARTIFACTS_PATH)
        art["DISPLAY_ARTIST"] = art["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
        sub = art[art["DISPLAY_ARTIST"].str.lower() == artist_key.lower()]
        if not sub.empty:
            catalog_slope = float(sub["catalog_momentum_slope"].median())
            archetype_cluster = int(sub["Archetype_Cluster"].mode().iloc[0])

    return {
        "max_historical_week1_volume": max_hist,
        "is_debut_studio_album": is_debut,
        "catalog_momentum_slope": catalog_slope,
        "Archetype_Cluster": archetype_cluster,
    }


def _fit_single_gamma(week1: float, week2: float) -> Dict[str, float]:
    """Two-point gamma fit; peak volume = max(W1, W2)."""
    w = np.array([1.0, 2.0])
    y = np.array([week1, week2], dtype=float)
    peak_idx = int(np.argmax(y))
    peak_v = float(y[peak_idx])
    if peak_v <= 0:
        return {"peak_volume_V": 0.0, "gamma_a": 1.5, "gamma_t_peak": 1.0, "delta_weeks": 0.0}

    y_norm = y / peak_v
    t_peak = float(w[peak_idx])
    a = 2.0
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            gamma_norm,
            w,
            y_norm,
            p0=[a, t_peak],
            bounds=([0.2, 1.0], [20.0, 10.0]),
            maxfev=5000,
        )
        a, t_peak = float(popt[0]), float(popt[1])
    except Exception:
        pass
    return {"peak_volume_V": peak_v, "gamma_a": a, "gamma_t_peak": t_peak}


def synthesize_album_features(
    artist_name: str,
    album_date: datetime | pd.Timestamp | str,
    singles: List[ManualSingle],
) -> pd.DataFrame:
    """
    Build one album-level feature row from manual singles (W1/W2 per single).

    Each single contributes a gamma curve shifted by weeks_before_album.
    """
    album_ts = pd.to_datetime(album_date)
    baselines = lookup_artist_baselines(artist_name, album_ts)

    if not singles:
        raise ValueError("Provide at least one hypothetical single.")

    track_rows: List[Dict[str, float]] = []
    for s in singles:
        g = _fit_single_gamma(s.week1_volume, s.week2_volume)
        g["delta_weeks"] = -float(s.weeks_before_album)
        track_rows.append(g)

    t_grid = build_album_time_grid(HORIZON_WEEKS)
    delta = np.array([r["delta_weeks"] for r in track_rows])
    a = np.array([r["gamma_a"] for r in track_rows])
    t_peak = np.array([r["gamma_t_peak"] for r in track_rows])
    v = np.array([r["peak_volume_V"] for r in track_rows])

    w = t_grid[:, None] + delta[None, :]
    valid = w >= 1.0
    safe_w = np.where(valid, w, 1.0)
    gamma_vals = gamma_norm_broadcast(safe_w, a[None, :], t_peak[None, :])
    gamma_masked = np.where(valid, gamma_vals, 0.0)
    m_abs = (gamma_masked * v[None, :]).sum(axis=1)
    m_norm = gamma_masked.sum(axis=1)

    rollout = extract_rollout_meta_features(
        t_grid,
        m_abs,
        m_norm,
        count_tracks=len(singles),
        max_single_peak=float(v.max()) if len(v) else 0.0,
    )
    rollout["cannibalization_ratio"] = 0.0 if len(singles) < 2 else -0.05
    rollout.update(baselines)
    rollout["DISPLAY_ARTIST"] = artist_name.strip()
    rollout["ALBUM_MRELG_ID"] = f"PROJ_{artist_name.strip().replace(' ', '_')}_{album_ts.date()}"
    rollout["ALBUM_FIRST_SALE_DATE"] = album_ts

    return pd.DataFrame([rollout])
