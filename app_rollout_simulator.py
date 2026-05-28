#!/usr/bin/env python3
"""
app_rollout_simulator.py
========================
Streamlit what-if engine: hypothetical lead singles → Phase 2 features →
bottom-up hurdle ensemble (Streaming + Song + Product) → reconciled Total AE.

Run:
  streamlit run app_rollout_simulator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from all_data_archetypes_simulator_ae import gamma_norm
from album_projection import ManualSingle, synthesize_album_features
from run_inference_pipeline import (
    CLASSIFIER_THRESHOLD,
    FALLBACK_BASE,
    HurdleModel,
    load_hurdle_meta,
    resolve_dynamic_threshold,
)
from historical_single_momentum import (
    DEFAULT_LEAD_PATH,
    DEFAULT_WEEKLY_PATH,
    lookup_historical_single_momentum as lookup_prior_album_single_momentum,
)
from target_config import (
    BOTTOM_UP_TARGETS,
    CLUSTER_FEATURE_COLS,
    HISTORICAL_SINGLE_MOMENTUM_COL,
    MOMENTUM_BASELINE_INTERACTION_COL,
    MOMENTUM_GROWTH_RATIO_COL,
    TargetConfig,
    _pick_stats_column,
    compute_momentum_growth_ratio,
    get_target_config,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STREAMING_FEATURES_PATH = Path("singles_artifacts/streaming/album_meta_features.parquet")
STREAMING_TRACK_PARAMS_PATH = Path("singles_artifacts/streaming/track_gamma_params.parquet")
HISTORICAL_ALBUMS_PATH = Path("data/firstweekae.csv")
SALES_STATS_PATH = Path("data/sales/artist_stats.parquet")
SALES_PROBS_PATH = Path("data/sales/artist_cluster_probs.parquet")
SONGS_STATS_PATH = Path("data/songs/artist_stats.parquet")
SONGS_PROBS_PATH = Path("data/songs/artist_cluster_probs.parquet")

SHAPE_DEFAULT_COLS = [*CLUSTER_FEATURE_COLS, "catalog_momentum_slope"]
PRODUCT_DEBUT_CAP_FRACTION = 0.05
SUPERSTAR_THRESHOLD = 150_000.0
SUPERSTAR_FLOOR_RATIO = 0.65
# Median expected_week1_carryover / max_historical for low single-momentum albums (training).
DEFAULT_CATALOG_CARRYOVER_RATIO = 0.024
GLOBAL_MEDIAN_CARRYOVER = 28.0
EPS = 1e-9
MAX_SINGLES = 3


@dataclass
class HypotheticalSingle:
    """User-entered pre-release single (enabled via checkbox in the UI)."""

    enabled: bool
    weeks_before_album: int
    week1_volume: float
    week2_volume: float
    week2_observed: bool
    week2_extrapolated: bool = False  # set when W2 inferred from artist decay

    @property
    def active(self) -> bool:
        return self.enabled and self.week1_volume > 0


# ---------------------------------------------------------------------------
# Cached asset loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ensemble_models() -> Dict[str, HurdleModel]:
    """Load hurdle classifiers + regressors for streaming, product, song."""
    models: Dict[str, HurdleModel] = {}
    for name in BOTTOM_UP_TARGETS:
        cfg = get_target_config(name)
        meta = load_hurdle_meta(cfg.models_dir)
        if not meta or "feature_cols" not in meta:
            raise FileNotFoundError(
                f"Missing hurdle_model_meta.json for {name} in {cfg.models_dir}"
            )
        models[name] = HurdleModel(
            classifier_path=cfg.classifier_path,
            regressor_path=cfg.regressor_path,
            feature_cols=meta["feature_cols"],
            threshold=CLASSIFIER_THRESHOLD,
            dynamic_threshold=resolve_dynamic_threshold(meta),
        )
    return models


@st.cache_data
def load_model_metas() -> Dict[str, dict]:
    """Per-target training metadata including dynamic_threshold (P75)."""
    metas: Dict[str, dict] = {}
    for name in BOTTOM_UP_TARGETS:
        cfg = get_target_config(name)
        meta = load_hurdle_meta(cfg.models_dir)
        if not meta:
            raise FileNotFoundError(f"Missing meta for {name}: {cfg.models_dir}")
        metas[name] = meta
    return metas


@st.cache_data
def load_artist_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Product/song historical stats and cluster probability tables."""
    for path in (SALES_STATS_PATH, SALES_PROBS_PATH, SONGS_STATS_PATH, SONGS_PROBS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    sales_stats = pd.read_parquet(SALES_STATS_PATH)
    sales_probs = pd.read_parquet(SALES_PROBS_PATH)
    songs_stats = pd.read_parquet(SONGS_STATS_PATH)
    songs_probs = pd.read_parquet(SONGS_PROBS_PATH)
    for df in (sales_stats, sales_probs, songs_stats, songs_probs):
        if "DISPLAY_ARTIST" in df.columns:
            df["DISPLAY_ARTIST"] = df["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    return sales_stats, sales_probs, songs_stats, songs_probs


@st.cache_data
def load_shape_medians() -> Dict[str, float]:
    """Industry median defaults for dimensionless rollout shape features."""
    if not STREAMING_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing training features for medians: {STREAMING_FEATURES_PATH}"
        )
    df = pd.read_parquet(STREAMING_FEATURES_PATH)
    medians: Dict[str, float] = {}
    for col in SHAPE_DEFAULT_COLS:
        if col in df.columns:
            medians[col] = float(df[col].median())
        else:
            medians[col] = 0.0
    return medians


@st.cache_data
def load_track_gamma_params() -> pd.DataFrame:
    """Per-single gamma fits from streaming Phase 2 (for artist decay fallback)."""
    if not STREAMING_TRACK_PARAMS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(STREAMING_TRACK_PARAMS_PATH)
    if "DISPLAY_ARTIST" in df.columns:
        df["DISPLAY_ARTIST"] = df["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    return df


@st.cache_data
def load_training_reference() -> pd.DataFrame:
    """Album-level reference for artist cluster / catalog defaults."""
    if not STREAMING_FEATURES_PATH.exists():
        return pd.DataFrame()
    ref = pd.read_parquet(STREAMING_FEATURES_PATH)
    ref["DISPLAY_ARTIST"] = ref["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    return ref


# ---------------------------------------------------------------------------
# Feature synthesis
# ---------------------------------------------------------------------------
def _gamma_week2_over_week1_ratios(track_df: pd.DataFrame) -> List[float]:
    """W2/W1 from fitted pre-album single gamma shapes (weeks 1 and 2 on curve)."""
    ratios: List[float] = []
    for _, tr in track_df.iterrows():
        a = float(tr.get("gamma_a", 1.5))
        t_peak = float(tr.get("gamma_t_peak", 1.0))
        g1 = float(gamma_norm(np.array([1.0]), a, t_peak)[0])
        g2 = float(gamma_norm(np.array([2.0]), a, t_peak)[0])
        if g1 > EPS:
            ratios.append(g2 / g1)
    return ratios


def lookup_artist_week2_retention_ratio(
    artist_name: str,
    sales_stats: pd.DataFrame,
    track_params: pd.DataFrame,
) -> Tuple[float, str]:
    """
    Week-1 → week-2 retention ratio for exponential decay (W2/W1).

    Prefers this artist's historical single gamma curves (streaming Phase 2),
    then catalog-wide gamma medians, then a conservative default.
    """
    artist_key = artist_name.strip().lower()

    if not track_params.empty and "DISPLAY_ARTIST" in track_params.columns:
        tsub = track_params[track_params["DISPLAY_ARTIST"].str.lower() == artist_key]
        ratios = _gamma_week2_over_week1_ratios(tsub)
        if ratios:
            ratio = float(np.clip(np.median(ratios), 0.05, 1.0))
            return ratio, f"artist singles decay ({len(ratios)} historical tracks)"

    if not track_params.empty:
        ratios = _gamma_week2_over_week1_ratios(track_params)
        if ratios:
            ratio = float(np.clip(np.median(ratios), 0.05, 1.0))
            return ratio, "catalog singles decay (all artists)"

    return 0.90, "default (0.90)"


def resolve_single_volumes(
    single: HypotheticalSingle,
    retention_ratio: float,
) -> HypotheticalSingle:
    """Extrapolate week 2 from artist decay when only week 1 is known."""
    if not single.active:
        return single
    if single.week2_observed and single.week2_volume > 0:
        single.week2_extrapolated = False
        return single
    w1 = max(float(single.week1_volume), 0.0)
    single.week2_volume = w1 * retention_ratio
    single.week2_extrapolated = True
    return single


def rollout_carryover_and_momentum(singles: List[HypotheticalSingle]) -> Tuple[float, float]:
    """
    Exponential decay carryover to album week and composite peak momentum.

    λ = -ln(W2/W1); volume at album week ≈ W1 * exp(-λ * weeks_before_album).
  """
    total_carryover = 0.0
    peak_momentum = 0.0
    for s in singles:
        if not s.active:
            continue
        w1 = max(float(s.week1_volume), 0.0)
        w2 = max(float(s.week2_volume), EPS)
        peak_momentum += w1
        if w1 <= 0:
            continue
        decay = -np.log(w2 / w1) if w2 > EPS else 0.0
        weeks = max(int(s.weeks_before_album), 0)
        total_carryover += w1 * float(np.exp(-decay * weeks))
    return total_carryover, peak_momentum


def lookup_streaming_baseline(
    artist_name: str,
    album_date: pd.Timestamp,
    training_ref: pd.DataFrame,
) -> Dict[str, Any]:
    """max_historical_week1_volume + is_debut_studio_album from firstweekae / training ref."""
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
            hist["DISPLAY_ARTIST"].str.lower() == artist_key.lower()
        ) & (hist["FIRST_SALE_DATE"] < album_date)
        past = hist.loc[mask, "FIRST_WEEK_TOTAL_AE"].dropna()
        if len(past) > 0:
            max_hist = float(past.max())
            is_debut = 0

    catalog_slope = 0.0
    archetype_cluster = 1
    if not training_ref.empty:
        sub = training_ref[training_ref["DISPLAY_ARTIST"].str.lower() == artist_key.lower()]
        if not sub.empty:
            catalog_slope = float(sub["catalog_momentum_slope"].median())
            archetype_cluster = int(sub["Archetype_Cluster"].mode().iloc[0])

    return {
        "max_historical_week1_volume": max_hist,
        "is_debut_studio_album": is_debut,
        "catalog_momentum_slope": catalog_slope,
        "Archetype_Cluster": archetype_cluster,
    }


def _artist_cluster_probs_wide(
    artist_name: str,
    probs_df: pd.DataFrame,
    required_cols: List[str],
) -> Dict[str, float]:
    """Pivot artist cluster probs to cluster_prob_* columns; 0 if unknown."""
    out = {c: 0.0 for c in required_cols if c.startswith("cluster_prob_")}
    artist_key = artist_name.strip().lower()
    sub = probs_df[probs_df["DISPLAY_ARTIST"].str.lower() == artist_key]
    if sub.empty:
        return out
    cluster_col = "Archetype_Cluster"
    prob_col = "prob"
    for _, row in sub.iterrows():
        key = f"cluster_prob_{int(row[cluster_col])}"
        if key in out:
            out[key] = float(row[prob_col])
    return out


def lookup_historical_single_momentum(
    artist_name: str,
    album_date: date,
    *,
    no_lead_singles_rollout: bool = False,
) -> Tuple[float, str]:
    """
    Prior-era momentum anchor for momentum_growth_ratio.

    With lead singles: prior album's lead-single Week-1 sum, or first-week AE if
    that prior album had no singles. Without lead singles: max first-week AE among
    prior albums that also had no lead singles.
    """
    momentum, _, source = lookup_prior_album_single_momentum(
        artist_name,
        pd.Timestamp(album_date),
        lead_path=DEFAULT_LEAD_PATH,
        weekly_path=DEFAULT_WEEKLY_PATH,
        no_lead_singles_rollout=no_lead_singles_rollout,
    )
    return momentum, source


def lookup_artist_baseline_carryover(
    artist_name: str,
    album_ts: pd.Timestamp,
    training_ref: pd.DataFrame,
    max_historical_week1_volume: float,
) -> Tuple[float, str]:
    """
    Expected week-1 carryover when there is no pre-release single rollout.

    Prefers this artist's median carryover in training; else scales historical peak.
    """
    artist_key = artist_name.strip().lower()
    if not training_ref.empty and "expected_week1_carryover" in training_ref.columns:
        sub = training_ref[training_ref["DISPLAY_ARTIST"].str.lower() == artist_key]
        if not sub.empty:
            med = float(sub["expected_week1_carryover"].median())
            if med > 0:
                return med, (
                    f"median catalog carryover from {len(sub):,} training album(s) "
                    "for this artist"
                )

    max_hist = max(float(max_historical_week1_volume), 0.0)
    if max_hist > 0:
        carry = max_hist * DEFAULT_CATALOG_CARRYOVER_RATIO
        return carry, (
            f"catalog proxy ({DEFAULT_CATALOG_CARRYOVER_RATIO:.1%} of historical peak "
            f"{max_hist:,.0f} AE)"
        )

    return GLOBAL_MEDIAN_CARRYOVER, "industry median carryover (debut / no history)"


def compute_momentum_baseline_interaction(
    baseline_max: float,
    composite_peak_momentum: float,
) -> float:
    """Superstar immunity: historical scale × current single momentum (raw product)."""
    return float(baseline_max) * float(composite_peak_momentum)


def lookup_component_baseline(
    artist_name: str,
    cfg: TargetConfig,
    stats_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Product/song baselines from artist_stats + cluster_probs (BASELINE_LAG_1 proxy)."""
    artist_key = artist_name.strip()
    row: Dict[str, Any] = {
        cfg.baseline_max_col: 0.0,
        cfg.baseline_median_col: 0.0,
        cfg.baseline_momentum_col: 0.0,
        cfg.debut_col: 1,
    }
    for c in feature_cols:
        if c.startswith("cluster_prob_"):
            row[c] = 0.0

    sub = stats_df[stats_df["DISPLAY_ARTIST"].str.lower() == artist_key.lower()]
    if not sub.empty:
        stats_row = sub.iloc[0]
        max_src = _pick_stats_column(stats_df, "max")
        med_src = _pick_stats_column(stats_df, "median")
        if max_src:
            row[cfg.baseline_max_col] = float(
                pd.to_numeric(stats_row[max_src], errors="coerce") or 0.0
            )
        elif med_src:
            row[cfg.baseline_max_col] = float(
                pd.to_numeric(stats_row[med_src], errors="coerce") or 0.0
            )
        if med_src:
            row[cfg.baseline_median_col] = float(
                pd.to_numeric(stats_row[med_src], errors="coerce") or 0.0
            )
        else:
            row[cfg.baseline_median_col] = row[cfg.baseline_max_col]
        if max_src is None and med_src and row[cfg.baseline_max_col] == 0.0:
            row[cfg.baseline_max_col] = row[cfg.baseline_median_col]
        row[cfg.debut_col] = int(
            row[cfg.baseline_max_col] == 0.0 and row[cfg.baseline_median_col] == 0.0
        )

    row.update(
        _artist_cluster_probs_wide(
            artist_name,
            probs_df,
            [c for c in feature_cols if c.startswith("cluster_prob_")],
        )
    )
    return row


def _finalize_feature_rows(
    artist_name: str,
    gamma_row: Dict[str, Any],
    streaming_extra: Dict[str, Any],
    peak_momentum: float,
    exp_carryover: float,
    hist_momentum: float,
    model_metas: Dict[str, dict],
    sales_stats: pd.DataFrame,
    sales_probs: pd.DataFrame,
    songs_stats: pd.DataFrame,
    songs_probs: pd.DataFrame,
    *,
    is_baseline_only: bool,
    retention_ratio: float = 0.0,
    retention_source: str = "",
    singles_detail: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Shared per-target row assembly for single-driven and baseline-only rollouts."""
    growth_ratio = float(compute_momentum_growth_ratio(peak_momentum, hist_momentum))
    gamma_row[HISTORICAL_SINGLE_MOMENTUM_COL] = hist_momentum
    gamma_row[MOMENTUM_GROWTH_RATIO_COL] = growth_ratio

    is_debut = bool(
        streaming_extra.get("is_debut_studio_album", 0) == 1
        or gamma_row.get("is_debut_album", 0) == 1
    )

    per_target: Dict[str, pd.DataFrame] = {}
    for name in BOTTOM_UP_TARGETS:
        cfg = get_target_config(name)
        feature_cols: List[str] = model_metas[name]["feature_cols"]
        row = {k: gamma_row[k] for k in feature_cols if k in gamma_row}

        if name == "product":
            row.update(
                lookup_component_baseline(
                    artist_name, cfg, sales_stats, sales_probs, feature_cols
                )
            )
            if row.get(cfg.debut_col, 1) == 1:
                is_debut = True
        elif name == "song":
            row.update(
                lookup_component_baseline(
                    artist_name, cfg, songs_stats, songs_probs, feature_cols
                )
            )
            if row.get(cfg.debut_col, 1) == 1:
                is_debut = True

        baseline_max = float(
            row.get(
                cfg.baseline_max_col,
                gamma_row.get(cfg.baseline_max_col, 0.0),
            )
        )
        row[cfg.baseline_momentum_col] = hist_momentum
        row[MOMENTUM_GROWTH_RATIO_COL] = growth_ratio
        row[MOMENTUM_BASELINE_INTERACTION_COL] = compute_momentum_baseline_interaction(
            baseline_max, peak_momentum
        )

        for col in feature_cols:
            if col not in row:
                row[col] = 0.0 if col.startswith("cluster_prob_") else gamma_row.get(col, 0.0)

        row["DISPLAY_ARTIST"] = artist_name.strip()
        row["ALBUM_MRELG_ID"] = gamma_row.get(
            "ALBUM_MRELG_ID",
            f"SIM_{artist_name.strip().replace(' ', '_')}",
        )
        per_target[name] = pd.DataFrame([row])

    meta = {
        "expected_week1_carryover": exp_carryover,
        "composite_peak_momentum": peak_momentum,
        "historical_single_momentum": hist_momentum,
        "momentum_growth_ratio": growth_ratio,
        "momentum_baseline_interaction": compute_momentum_baseline_interaction(
            float(streaming_extra.get("max_historical_week1_volume", 0.0)),
            peak_momentum,
        ),
        "is_debut": is_debut,
        "is_baseline_only": is_baseline_only,
        "n_singles": 0 if is_baseline_only else len(singles_detail or []),
        "week2_retention_ratio": retention_ratio,
        "week2_retention_source": retention_source,
        "singles_detail": singles_detail or [],
    }
    return per_target, meta


def synthesize_baseline_only_features(
    artist_name: str,
    album_date: date,
    shape_medians: Dict[str, float],
    training_ref: pd.DataFrame,
    sales_stats: pd.DataFrame,
    sales_probs: pd.DataFrame,
    songs_stats: pd.DataFrame,
    songs_probs: pd.DataFrame,
    model_metas: Dict[str, dict],
    historical_single_momentum: float,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Feature rows when there are no pre-release lead singles.

    Uses artist historical peak + catalog carryover; prior-era anchor from albums
    that also shipped without lead singles when available.
    """
    album_ts = pd.Timestamp(album_date)
    streaming_extra = lookup_streaming_baseline(artist_name, album_ts, training_ref)
    max_hist = float(streaming_extra.get("max_historical_week1_volume", 0.0))
    exp_carryover, carry_source = lookup_artist_baseline_carryover(
        artist_name, album_ts, training_ref, max_hist
    )
    peak_momentum = 0.0
    hist_momentum = max(float(historical_single_momentum), 0.0)

    gamma_row: Dict[str, Any] = dict(shape_medians)
    gamma_row.update(streaming_extra)
    gamma_row.update({
        "count_pre_release_tracks": 0.0,
        "max_single_peak_volume": 0.0,
        "total_pre_release_auc": 0.0,
        "terminal_velocity": exp_carryover * 0.1,
        "composite_peak_momentum": peak_momentum,
        "expected_week1_carryover": exp_carryover,
        "cannibalization_ratio": 0.0,
        "DISPLAY_ARTIST": artist_name.strip(),
        "ALBUM_MRELG_ID": (
            f"SIM_{artist_name.strip().replace(' ', '_')}_{album_ts.date()}"
        ),
        "ALBUM_FIRST_SALE_DATE": album_ts,
    })

    return _finalize_feature_rows(
        artist_name,
        gamma_row,
        streaming_extra,
        peak_momentum,
        exp_carryover,
        hist_momentum,
        model_metas,
        sales_stats,
        sales_probs,
        songs_stats,
        songs_probs,
        is_baseline_only=True,
        retention_source=carry_source,
    )


def synthesize_feature_rows(
    artist_name: str,
    album_date: date,
    singles: List[HypotheticalSingle],
    shape_medians: Dict[str, float],
    training_ref: pd.DataFrame,
    sales_stats: pd.DataFrame,
    sales_probs: pd.DataFrame,
    songs_stats: pd.DataFrame,
    songs_probs: pd.DataFrame,
    track_params: pd.DataFrame,
    model_metas: Dict[str, dict],
    historical_single_momentum: float,
    *,
    baseline_only: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Build one feature row per ensemble target from user inputs.

    Returns per-target DataFrames and shared core metadata for display.
    """
    if baseline_only:
        return synthesize_baseline_only_features(
            artist_name=artist_name,
            album_date=album_date,
            shape_medians=shape_medians,
            training_ref=training_ref,
            sales_stats=sales_stats,
            sales_probs=sales_probs,
            songs_stats=songs_stats,
            songs_probs=songs_probs,
            model_metas=model_metas,
            historical_single_momentum=historical_single_momentum,
        )

    active = [s for s in singles if s.active]
    if not active:
        raise ValueError("Enable at least one single and enter Week 1 volume > 0.")

    retention_ratio, retention_source = lookup_artist_week2_retention_ratio(
        artist_name, sales_stats, track_params
    )
    resolved = [
        resolve_single_volumes(s, retention_ratio) for s in active
    ]

    manual = [
        ManualSingle(
            weeks_before_album=s.weeks_before_album,
            week1_volume=s.week1_volume,
            week2_volume=s.week2_volume,
        )
        for s in resolved
    ]
    album_ts = pd.Timestamp(album_date)
    exp_carryover, peak_momentum = rollout_carryover_and_momentum(resolved)

    gamma_row = synthesize_album_features(artist_name, album_ts, manual).iloc[0].to_dict()
    gamma_row["expected_week1_carryover"] = exp_carryover
    gamma_row["composite_peak_momentum"] = peak_momentum
    gamma_row["count_pre_release_tracks"] = float(len(active))
    gamma_row["max_single_peak_volume"] = float(max(s.week1_volume for s in resolved))

    for col, val in shape_medians.items():
        gamma_row[col] = val

    streaming_extra = lookup_streaming_baseline(artist_name, album_ts, training_ref)
    gamma_row.update(streaming_extra)

    hist_momentum = max(float(historical_single_momentum), 0.0)
    singles_detail = [
        {
            "weeks_before_album": s.weeks_before_album,
            "week1": s.week1_volume,
            "week2": s.week2_volume,
            "week2_extrapolated": s.week2_extrapolated,
        }
        for s in resolved
    ]

    return _finalize_feature_rows(
        artist_name,
        gamma_row,
        streaming_extra,
        peak_momentum,
        exp_carryover,
        hist_momentum,
        model_metas,
        sales_stats,
        sales_probs,
        songs_stats,
        songs_probs,
        is_baseline_only=False,
        retention_ratio=retention_ratio,
        retention_source=retention_source,
        singles_detail=singles_detail,
    )


def run_bottom_up_forecast(
    feature_rows: Dict[str, pd.DataFrame],
    models: Dict[str, HurdleModel],
    is_debut: bool,
) -> Dict[str, float]:
    """Hurdle inference per component + debut product cap + Total AE."""
    preds: Dict[str, float] = {}
    for name in BOTTOM_UP_TARGETS:
        final, _, _ = models[name].predict_values(feature_rows[name])
        preds[name] = float(final[0])

    pred_product = preds["product"]
    if is_debut:
        cap = preds["streaming"] * PRODUCT_DEBUT_CAP_FRACTION
        pred_product = float(min(pred_product, cap))
        preds["product_capped"] = pred_product
    else:
        preds["product_capped"] = pred_product

    preds["total_ae"] = preds["streaming"] + preds["song"] + preds["product_capped"]
    return preds


def apply_superstar_floor(
    total_ae: float,
    baseline_max: float,
    *,
    threshold: float = SUPERSTAR_THRESHOLD,
    floor_ratio: float = SUPERSTAR_FLOOR_RATIO,
) -> Tuple[float, bool]:
    """
    Superstar immunity floor for generational artists.

    If historical max is large enough, prevent the final forecast from collapsing
    below a fixed fraction of that historical peak.
    """
    baseline = max(float(baseline_max), 0.0)
    final_val = float(total_ae)
    if baseline >= threshold:
        superstar_floor = baseline * float(floor_ratio)
        if final_val < superstar_floor:
            return superstar_floor, True
    return final_val, False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Rollout Simulator — First Week Forecast",
        layout="wide",
    )
    st.title("Album First-Week Rollout Simulator")
    st.caption(
        "What-if engine: hypothetical lead singles or baseline-only catalog rollout → "
        "bottom-up ensemble (Streaming + Song + Product) → reconciled Total AE."
    )

    try:
        models = load_ensemble_models()
        model_metas = load_model_metas()
        sales_stats, sales_probs, songs_stats, songs_probs = load_artist_artifacts()
        shape_medians = load_shape_medians()
        training_ref = load_training_reference()
        track_params = load_track_gamma_params()
    except FileNotFoundError as exc:
        st.error(f"Startup failed — missing asset: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Release metadata")
        artist_name = st.text_input("Artist name", value="Taylor Swift")
        album_date = st.date_input("Expected album date", value=date.today())
        no_lead_singles = st.checkbox(
            "No lead singles (baseline-only rollout)",
            value=False,
            help=(
                "Album drops without pre-release singles. Forecast uses artist "
                "historical peak + catalog carryover, and prior albums that also "
                "had no lead singles when available."
            ),
        )
        st.divider()
        st.subheader("Historical anchors")
        default_hist_momentum, hist_momentum_source = lookup_historical_single_momentum(
            artist_name,
            album_date,
            no_lead_singles_rollout=no_lead_singles,
        )
        historical_single_momentum = st.number_input(
            "Historical momentum anchor",
            min_value=0.0,
            value=default_hist_momentum,
            step=100.0,
            help=(
                "Prior-era anchor for momentum_growth_ratio. With singles: prior "
                "lead-single Week-1 sum (or prior album first-week AE if none). "
                "Without singles: max first-week AE among prior no-lead-single albums."
            ),
        )
        st.caption(f"Auto-filled from: {hist_momentum_source}")
        st.divider()
        st.subheader("Training gates (P75)")
        for name in BOTTOM_UP_TARGETS:
            th = resolve_dynamic_threshold(model_metas[name])
            label = name.capitalize()
            st.metric(label, f"{th:,.0f}" if th else "n/a")

    st.subheader("Hypothetical lead singles")
    if no_lead_singles:
        st.info(
            "Baseline-only mode: no pre-release singles. The model uses artist "
            "historical peak, catalog carryover, and—when available—prior albums "
            "that also shipped without lead singles."
        )
    else:
        st.caption(
            "Enable each single with the checkbox. If Week 2 has not happened yet, "
            "leave “Week 2 observed” unchecked — we extrapolate W2 from this artist’s "
            "historical singles decay."
        )
    singles: List[HypotheticalSingle] = []
    with st.expander(
        "Configure up to 3 pre-release singles",
        expanded=not no_lead_singles,
    ):
        for i in range(MAX_SINGLES):
            hdr_l, hdr_r = st.columns([0.08, 0.92])
            enabled = hdr_l.checkbox(
                " ",
                value=(i == 0) and not no_lead_singles,
                key=f"enable_{i}",
                label_visibility="collapsed",
                disabled=no_lead_singles,
            )
            hdr_r.markdown(f"**Single {i + 1}**")

            c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.0])
            weeks = c1.slider(
                "Weeks before album",
                min_value=1,
                max_value=20,
                value=max(8 - i * 2, 1),
                key=f"weeks_{i}",
                disabled=no_lead_singles or not enabled,
            )
            w1 = c2.number_input(
                "Week 1 volume (AE)",
                min_value=0.0,
                value=5000.0 * (1.1 - 0.15 * i) if enabled else 0.0,
                step=100.0,
                key=f"w1_{i}",
                disabled=no_lead_singles or not enabled,
            )
            week2_observed = c4.checkbox(
                "Week 2 observed",
                value=(i == 0),
                key=f"w2_obs_{i}",
                disabled=no_lead_singles or not enabled,
                help="Uncheck if only one week of data exists; W2 will be extrapolated.",
            )
            w2 = c3.number_input(
                "Week 2 volume (AE)",
                min_value=0.0,
                value=4500.0 * (1.05 - 0.15 * i) if (enabled and week2_observed) else 0.0,
                step=100.0,
                key=f"w2_{i}",
                disabled=no_lead_singles or not enabled or not week2_observed,
            )
            singles.append(
                HypotheticalSingle(
                    enabled=enabled,
                    weeks_before_album=int(weeks),
                    week1_volume=float(w1),
                    week2_volume=float(w2),
                    week2_observed=week2_observed,
                )
            )

    if st.button("Run Forecast", type="primary", use_container_width=True):
        try:
            feature_rows, synth_meta = synthesize_feature_rows(
                artist_name=artist_name,
                album_date=album_date,
                singles=singles,
                shape_medians=shape_medians,
                training_ref=training_ref,
                sales_stats=sales_stats,
                sales_probs=sales_probs,
                songs_stats=songs_stats,
                songs_probs=songs_probs,
                track_params=track_params,
                model_metas=model_metas,
                historical_single_momentum=historical_single_momentum,
                baseline_only=no_lead_singles,
            )
            preds = run_bottom_up_forecast(
                feature_rows, models, is_debut=bool(synth_meta["is_debut"])
            )
            baseline_max = float(
                feature_rows["streaming"]["max_historical_week1_volume"].iloc[0]
            )
            floored_total, floor_applied = apply_superstar_floor(
                preds["total_ae"],
                baseline_max,
            )
            preds["total_ae"] = floored_total
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.stop()

        st.success("Forecast complete.")

        if synth_meta.get("is_baseline_only"):
            st.info(
                f"Catalog carryover: **{synth_meta['expected_week1_carryover']:,.0f} AE** "
                f"({synth_meta.get('week2_retention_source', 'baseline estimate')}). "
                f"Historical peak (streaming): **{baseline_max:,.0f} AE**."
            )

        if synth_meta.get("momentum_growth_ratio") is not None:
            st.caption(
                f"Relative momentum growth: **{synth_meta['momentum_growth_ratio']:.2f}×** "
                f"(current {synth_meta['composite_peak_momentum']:,.0f} AE vs "
                f"prior-era {synth_meta['historical_single_momentum']:,.0f} AE)"
            )
        if (
            synth_meta.get("week2_retention_source")
            and not synth_meta.get("is_baseline_only")
            and synth_meta.get("week2_retention_ratio", 0) > 0
        ):
            st.caption(
                f"Week-2 extrapolation ratio (W2/W1): **{synth_meta['week2_retention_ratio']:.3f}** "
                f"— {synth_meta['week2_retention_source']}"
            )
        if synth_meta.get("singles_detail"):
            extrap = [d for d in synth_meta["singles_detail"] if d["week2_extrapolated"]]
            if extrap:
                st.info(
                    "Week 2 was extrapolated for "
                    + ", ".join(
                        f"single @ {d['weeks_before_album']}w "
                        f"(W2≈{d['week2']:,.0f})"
                        for d in extrap
                    )
                )

        c1, c2, c3 = st.columns(3)
        c1.metric("Streaming (AE)", f"{preds['streaming']:,.0f}")
        c2.metric("Song sales (AE)", f"{preds['song']:,.0f}")
        product_label = "Product sales (AE)"
        if synth_meta["is_debut"] and preds.get("product", 0) != preds.get("product_capped"):
            product_label += " (debut-capped)"
        c3.metric(product_label, f"{preds['product_capped']:,.0f}")

        if floor_applied:
            st.warning(
                "🌟 Superstar Immunity Floor Applied: Single momentum tracked too low, "
                "projection floored to 65% of historical peak."
            )

        st.metric(
            "Reconciled Total AE",
            f"{preds['total_ae']:,.0f}",
            help="Sum of streaming + song + product (product capped at 5% of streaming for debuts).",
        )

        if synth_meta["is_debut"]:
            st.info(
                "Cold start: debut artist — product prediction capped at "
                f"{PRODUCT_DEBUT_CAP_FRACTION:.0%} of streaming "
                f"({preds['streaming'] * PRODUCT_DEBUT_CAP_FRACTION:,.0f})."
            )

        st.subheader("Synthesized features (streaming model input)")
        display_cols = model_metas["streaming"]["feature_cols"]
        stream_vals = feature_rows["streaming"][display_cols].iloc[0]
        stream_display = pd.DataFrame({
            "Feature": display_cols,
            "Value": stream_vals.values,
        })
        st.dataframe(stream_display, use_container_width=True, hide_index=True)

        with st.expander("All component feature rows"):
            tabs = st.tabs([t.capitalize() for t in BOTTOM_UP_TARGETS])
            for tab, name in zip(tabs, BOTTOM_UP_TARGETS):
                with tab:
                    cols = model_metas[name]["feature_cols"]
                    vals = feature_rows[name][cols].iloc[0]
                    st.dataframe(
                        pd.DataFrame({"Feature": cols, "Value": vals.values}),
                        use_container_width=True,
                        hide_index=True,
                    )


if __name__ == "__main__":
    main()
