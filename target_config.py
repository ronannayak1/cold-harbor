#!/usr/bin/env python3
"""
target_config.py
================
Central registry for multi-target album first-week forecasting pipelines.

Pre-album singles only carry streaming (rollout shape) metrics. Product and song
first-week targets live on albums; target-specific baselines and cluster probs
come from data/sales/ or data/songs/ and are merged at album level only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

TARGET_CHOICES = ("total_ae", "product", "song", "streaming")

# Bottom-up ensemble uses these component targets (summed to reconcile Total AE).
BOTTOM_UP_TARGETS = ("streaming", "song", "product")

# Weekly metric on lead singles for gamma curves / composite rollout (not album sales).
SINGLES_ROLLOUT_METRIC_COL = "STREAMING_EQUIVALENT"

# Shared rollout / shape features (album_meta_features export).
BASE_ROLLOUT_FEATURES = [
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

CLUSTER_FEATURE_COLS = [
    "norm_terminal_velocity",
    "norm_total_auc",
    "hype_concentration",
    "peak_proximity_weeks",
    "cannibalization_ratio",
]

BASE_ABSOLUTE_EXPORT_COLS = [
    "count_pre_release_tracks",
    "max_single_peak_volume",
    "total_pre_release_auc",
    "terminal_velocity",
    "composite_peak_momentum",
    "expected_week1_carryover",
    "catalog_momentum_slope",
    "momentum_baseline_interaction",
    "historical_single_momentum",
    "momentum_growth_ratio",
]

MOMENTUM_BASELINE_INTERACTION_COL = "momentum_baseline_interaction"
HISTORICAL_SINGLE_MOMENTUM_COL = "historical_single_momentum"
MOMENTUM_GROWTH_RATIO_COL = "momentum_growth_ratio"


@dataclass(frozen=True)
class TargetConfig:
    """Paths, columns, and artifact locations for one forecast target."""

    name: str
    target_col: str
    artifacts_dir: Path
    models_dir: Path
    target_data_dir: Optional[Path]
    targets_path: Path
    use_firstweekae_baseline: bool
    baseline_max_col: str
    baseline_median_col: str
    debut_col: str
    baseline_momentum_col: str = HISTORICAL_SINGLE_MOMENTUM_COL

    @property
    def singles_rollout_metric_col(self) -> str:
        """Metric on weekly singles for pre-album rollout features (always streaming)."""
        return SINGLES_ROLLOUT_METRIC_COL

    @property
    def features_path(self) -> Path:
        return self.artifacts_dir / "album_meta_features.parquet"

    @property
    def classifier_path(self) -> Path:
        return self.models_dir / "lgb_hurdle_classifier.txt"

    @property
    def regressor_path(self) -> Path:
        return self.models_dir / "lgb_hightier_regressor.joblib"

    @property
    def inference_output_path(self) -> Path:
        return self.models_dir / "inference_output.csv"

    @property
    def artist_stats_path(self) -> Optional[Path]:
        if self.target_data_dir is None:
            return None
        return self.target_data_dir / "artist_stats.parquet"

    @property
    def artist_cluster_probs_path(self) -> Optional[Path]:
        if self.target_data_dir is None:
            return None
        return self.target_data_dir / "artist_cluster_probs.parquet"


def get_target_config(target: str) -> TargetConfig:
    """Resolve configuration for --target choice."""
    if target not in TARGET_CHOICES:
        raise ValueError(f"Unknown target '{target}'. Choose from {TARGET_CHOICES}.")

    if target == "total_ae":
        return TargetConfig(
            name="total_ae",
            target_col="FIRST_WEEK_TOTAL_AE",
            artifacts_dir=Path("singles_artifacts/total_ae"),
            models_dir=Path("models/album_firstweek"),
            target_data_dir=None,
            targets_path=Path("data/78_pre_album_albums_firstweek.csv"),
            use_firstweekae_baseline=True,
            baseline_max_col="max_historical_week1_volume",
            baseline_median_col="max_historical_week1_volume",
            debut_col="is_debut_studio_album",
        )
    if target == "streaming":
        return TargetConfig(
            name="streaming",
            target_col="FIRST_WEEK_STREAMING_EQUIVALENT",
            artifacts_dir=Path("singles_artifacts/streaming"),
            models_dir=Path("models/streaming_firstweek"),
            target_data_dir=None,
            targets_path=Path("data/78_pre_album_albums_firstweek.csv"),
            use_firstweekae_baseline=True,
            baseline_max_col="max_historical_week1_volume",
            baseline_median_col="max_historical_week1_volume",
            debut_col="is_debut_studio_album",
        )
    if target == "product":
        return TargetConfig(
            name="product",
            target_col="FIRST_WEEK_PRODUCT_SALES",
            artifacts_dir=Path("singles_artifacts/sales"),
            models_dir=Path("models/sales_firstweek"),
            target_data_dir=Path("data/sales"),
            targets_path=Path("data/78_pre_album_albums_firstweek.csv"),
            use_firstweekae_baseline=False,
            baseline_max_col="max_historical_product_sales",
            baseline_median_col="median_historical_product_sales",
            debut_col="is_debut_album",
        )
    # song
    return TargetConfig(
        name="song",
        target_col="FIRST_WEEK_SONG_SALE_EQUIVALENT",
        artifacts_dir=Path("singles_artifacts/songs"),
        models_dir=Path("models/songs_firstweek"),
        target_data_dir=Path("data/songs"),
        targets_path=Path("data/78_pre_album_albums_firstweek.csv"),
        use_firstweekae_baseline=False,
        baseline_max_col="max_historical_song_sales",
        baseline_median_col="median_historical_song_sales",
        debut_col="is_debut_album",
    )


def validate_target_data_dir(cfg: TargetConfig) -> None:
    """Exit if album-level product/song artifact folders or parquet files are missing."""
    if cfg.target_data_dir is None:
        return
    if not cfg.target_data_dir.is_dir():
        print(
            f"ERROR: --target {cfg.name} requires directory {cfg.target_data_dir}, "
            "which does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)
    for path in (cfg.artist_stats_path, cfg.artist_cluster_probs_path):
        if path is None or not path.exists():
            print(
                f"ERROR: Missing required artifact for --target {cfg.name}: {path}",
                file=sys.stderr,
            )
            sys.exit(1)


def _pick_stats_column(stats: pd.DataFrame, kind: str) -> Optional[str]:
    """Find max or median historical column in artist_stats (flexible naming)."""
    kind = kind.lower()
    priority = []
    for col in stats.columns:
        cl = col.lower()
        if kind == "max" and "max" in cl and any(
            k in cl for k in ("historical", "first_week", "week1", "product", "song", "sales", "ae")
        ):
            priority.append(col)
        if kind == "median" and "median" in cl and any(
            k in cl for k in ("historical", "first_week", "week1", "product", "song", "sales", "ae", "peak")
        ):
            priority.append(col)
    if priority:
        return priority[0]
    if kind == "max":
        for cand in ("max_historical_week1_volume", "max_first_week", "max_peak_volume"):
            if cand in stats.columns:
                return cand
    for cand in ("median_historical_week1_volume", "median_first_week", "median_peak_volume"):
        if cand in stats.columns:
            return cand
    return None


def _pick_stats_momentum_column(stats: pd.DataFrame) -> Optional[str]:
    """Find historical single momentum column in artist_stats (flexible naming)."""
    if HISTORICAL_SINGLE_MOMENTUM_COL in stats.columns:
        return HISTORICAL_SINGLE_MOMENTUM_COL
    for col in stats.columns:
        cl = col.lower()
        if "historical" in cl and "momentum" in cl:
            return col
        if "single" in cl and "momentum" in cl:
            return col
    if "median_peak_volume" in stats.columns:
        return "median_peak_volume"
    return None


def compute_momentum_growth_ratio(
    composite_peak_momentum: pd.Series | float,
    historical_single_momentum: pd.Series | float,
) -> pd.Series | float:
    """
    Relative momentum growth: current composite peak / prior-era single W1.

    Denominator uses 1.0 when historical momentum is 0 (debut / no prior singles).
    """
    if isinstance(composite_peak_momentum, pd.Series):
        peak = pd.to_numeric(composite_peak_momentum, errors="coerce").fillna(0.0)
        hist = pd.to_numeric(historical_single_momentum, errors="coerce").fillna(0.0)
        denom = hist.replace(0.0, 1.0)
        return peak / denom
    peak_val = float(composite_peak_momentum)
    hist_val = float(historical_single_momentum)
    denom = hist_val if hist_val > 0.0 else 1.0
    return peak_val / denom


def merge_target_artist_artifacts(
    album_df: pd.DataFrame,
    cfg: TargetConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge album-level artist_stats and cluster probabilities onto album_df.

    Used for product/song targets only (data/sales or data/songs). Singles weekly
    data do not carry product sales; rollout shape comes from streaming elsewhere.

    Returns updated dataframe and list of new column names for export/training.
    """
    validate_target_data_dir(cfg)
    assert cfg.artist_stats_path is not None
    assert cfg.artist_cluster_probs_path is not None

    out = album_df.copy()
    out["DISPLAY_ARTIST"] = out["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()

    stats = pd.read_parquet(cfg.artist_stats_path)
    stats["DISPLAY_ARTIST"] = stats["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()

    max_src = _pick_stats_column(stats, "max")
    med_src = _pick_stats_column(stats, "median")
    if max_src is None and med_src is None:
        print(
            f"ERROR: Could not find max/median baseline columns in {cfg.artist_stats_path}. "
            f"Columns: {list(stats.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    stats_sub = stats[["DISPLAY_ARTIST"]].copy()
    if max_src:
        stats_sub[cfg.baseline_max_col] = pd.to_numeric(stats[max_src], errors="coerce")
    else:
        stats_sub[cfg.baseline_max_col] = 0.0
    if med_src:
        stats_sub[cfg.baseline_median_col] = pd.to_numeric(stats[med_src], errors="coerce")
    else:
        stats_sub[cfg.baseline_median_col] = stats_sub[cfg.baseline_max_col]

    out = out.merge(stats_sub, on="DISPLAY_ARTIST", how="left")
    out[cfg.baseline_max_col] = out[cfg.baseline_max_col].fillna(0.0)
    out[cfg.baseline_median_col] = out[cfg.baseline_median_col].fillna(0.0)
    out[cfg.debut_col] = (
        (out[cfg.baseline_max_col] == 0.0) & (out[cfg.baseline_median_col] == 0.0)
    ).astype(int)

    probs = pd.read_parquet(cfg.artist_cluster_probs_path)
    artist_col = "DISPLAY_ARTIST" if "DISPLAY_ARTIST" in probs.columns else None
    if artist_col is None:
        for c in probs.columns:
            if "artist" in c.lower():
                artist_col = c
                break
    if artist_col is None:
        print(
            f"ERROR: artist_cluster_probs missing DISPLAY_ARTIST column: {list(probs.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    cluster_col = "Archetype_Cluster" if "Archetype_Cluster" in probs.columns else None
    if cluster_col is None:
        for c in probs.columns:
            if "cluster" in c.lower():
                cluster_col = c
                break
    prob_val_col = "prob" if "prob" in probs.columns else None
    if prob_val_col is None:
        for c in probs.columns:
            if "prob" in c.lower():
                prob_val_col = c
                break
    if cluster_col is None or prob_val_col is None:
        print(
            f"ERROR: artist_cluster_probs needs cluster + prob columns: {list(probs.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    probs = probs.copy()
    probs[artist_col] = probs[artist_col].fillna("").astype(str).str.strip()
    prob_wide = (
        probs.pivot_table(
            index=artist_col,
            columns=cluster_col,
            values=prob_val_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(columns={artist_col: "DISPLAY_ARTIST"})
    )
    prob_cols = [c for c in prob_wide.columns if c != "DISPLAY_ARTIST"]
    prob_wide = prob_wide.rename(
        columns={c: f"cluster_prob_{int(c)}" if str(c).isdigit() else f"cluster_prob_{c}" for c in prob_cols}
    )
    new_prob_cols = [c for c in prob_wide.columns if c.startswith("cluster_prob_")]

    out = out.merge(prob_wide, on="DISPLAY_ARTIST", how="left")
    for c in new_prob_cols:
        out[c] = out[c].fillna(0.0)

    extra_cols = [
        cfg.baseline_max_col,
        cfg.baseline_median_col,
        cfg.debut_col,
        *new_prob_cols,
    ]
    return out, extra_cols


def build_training_feature_cols(
    feature_df: pd.DataFrame,
    cfg: TargetConfig,
) -> List[str]:
    """Feature list for LightGBM: rollout + target-specific baselines + cluster probs."""
    cols = list(BASE_ROLLOUT_FEATURES)
    if cfg.use_firstweekae_baseline:
        cols.extend(["max_historical_week1_volume", "is_debut_studio_album"])
    else:
        cols.extend([cfg.baseline_max_col, cfg.baseline_median_col, cfg.debut_col])
        cols.extend([c for c in feature_df.columns if c.startswith("cluster_prob_")])
    cols.extend([
        cfg.baseline_momentum_col,
        MOMENTUM_GROWTH_RATIO_COL,
        MOMENTUM_BASELINE_INTERACTION_COL,
    ])
    # Deduplicate while preserving order; only keep columns present in df
    seen: set[str] = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen and c in feature_df.columns:
            seen.add(c)
            ordered.append(c)
    return ordered


def add_target_cli(parser) -> None:
    """Register --target on an argparse parser."""
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="total_ae",
        help=(
            "Forecast target: total_ae, streaming, product (sales), or song (song sales)"
        ),
    )
