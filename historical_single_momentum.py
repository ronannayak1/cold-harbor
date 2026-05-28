#!/usr/bin/env python3
"""
historical_single_momentum.py
=============================
Construct historical_single_momentum from the singles weekly dataset and
lead-single → album mapping.

For each album, find the artist's immediately prior album in the cohort, take
all lead singles tied to that prior album, and sum each single's Week-1
STREAMING_EQUIVALENT. This parallels composite_peak_momentum (sum of current
campaign single W1 volumes) for momentum_growth_ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from target_config import HISTORICAL_SINGLE_MOMENTUM_COL, SINGLES_ROLLOUT_METRIC_COL

DEFAULT_LEAD_PATH = Path("data/lead_singles_with_album_within_6mo.csv")
DEFAULT_WEEKLY_PATH = Path("data/78_pre_album_singles_weekly.parquet")
DEFAULT_FIRSTWEEKAE_PATH = Path("data/firstweekae.csv")


def load_lead_single_mapping(lead_path: Path = DEFAULT_LEAD_PATH) -> pd.DataFrame:
    """
    Standardize lead-single → album mapping (one row per single).

    Returns columns: MRELG_ID, ALBUM_MRELG_ID, DISPLAY_ARTIST, ALBUM_FIRST_SALE_DATE.
    """
    lead = pd.read_csv(lead_path, encoding="utf-8-sig")
    lead.columns = [c.upper() for c in lead.columns]

    if "MRELG_ID_ALBUM" in lead.columns and "MRELG_ID" in lead.columns:
        lead = lead.rename(
            columns={
                "MRELG_ID": "SINGLE_MRELG_ID",
                "MRELG_ID_ALBUM": "ALBUM_MRELG_ID",
                "FIRST_SALE_DATE_ALBUM": "ALBUM_FIRST_SALE_DATE",
            }
        )

    artist_col = "DISPLAY_ARTIST" if "DISPLAY_ARTIST" in lead.columns else None
    if artist_col is None:
        for c in lead.columns:
            if "artist" in c.lower():
                artist_col = c
                break
    if artist_col is None:
        raise ValueError(f"{lead_path} missing DISPLAY_ARTIST column")

    if "SINGLE_MRELG_ID" in lead.columns:
        single_col = "SINGLE_MRELG_ID"
    elif "MRELG_ID" in lead.columns:
        single_col = "MRELG_ID"
    else:
        raise ValueError(f"{lead_path} missing single MRELG_ID column")

    out = lead[[single_col, "ALBUM_MRELG_ID", artist_col, "ALBUM_FIRST_SALE_DATE"]].copy()
    out = out.rename(
        columns={
            single_col: "MRELG_ID",
            artist_col: "DISPLAY_ARTIST",
        }
    )
    out["MRELG_ID"] = out["MRELG_ID"].astype(str).str.strip()
    out["ALBUM_MRELG_ID"] = out["ALBUM_MRELG_ID"].astype(str).str.strip()
    out["DISPLAY_ARTIST"] = out["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    out["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(
        out["ALBUM_FIRST_SALE_DATE"], errors="coerce"
    )
    out = out.dropna(subset=["MRELG_ID", "ALBUM_MRELG_ID", "ALBUM_FIRST_SALE_DATE"])
    out = out.drop_duplicates(subset=["MRELG_ID"], keep="first")
    return out


def build_single_week1_volumes(
    weekly_path: Path = DEFAULT_WEEKLY_PATH,
    metric_col: str = SINGLES_ROLLOUT_METRIC_COL,
    single_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Week-1 volume per single: metric at the earliest WEEKS_SINCE_RELEASE observed.

    Returns DataFrame with MRELG_ID, week1_volume.
    """
    if not weekly_path.exists():
        return pd.DataFrame(columns=["MRELG_ID", "week1_volume"])

    cols = ["MRELG_ID", "WEEKS_SINCE_RELEASE", metric_col]
    weekly = pd.read_parquet(weekly_path, columns=cols)
    weekly["MRELG_ID"] = weekly["MRELG_ID"].astype(str).str.strip()
    weekly[metric_col] = pd.to_numeric(weekly[metric_col], errors="coerce").fillna(0.0)
    weekly["WEEKS_SINCE_RELEASE"] = pd.to_numeric(
        weekly["WEEKS_SINCE_RELEASE"], errors="coerce"
    )

    if single_ids is not None:
        weekly = weekly[weekly["MRELG_ID"].isin(single_ids)]

    min_week = weekly.groupby("MRELG_ID", sort=False)["WEEKS_SINCE_RELEASE"].transform(
        "min"
    )
    week1_rows = weekly[weekly["WEEKS_SINCE_RELEASE"] == min_week]
    peaks = (
        week1_rows.groupby("MRELG_ID", sort=False)[metric_col]
        .max()
        .reset_index()
        .rename(columns={metric_col: "week1_volume"})
    )
    return peaks


def _prior_album_id(
    artist: str,
    album_date: pd.Timestamp,
    album_timeline: pd.DataFrame,
) -> Optional[str]:
    """Most recent cohort album by this artist strictly before album_date."""
    artist_key = artist.strip().lower()
    sub = album_timeline[
        (album_timeline["artist_key"] == artist_key)
        & (album_timeline["ALBUM_FIRST_SALE_DATE"] < album_date)
    ]
    if sub.empty:
        return None
    idx = sub["ALBUM_FIRST_SALE_DATE"].idxmax()
    return str(sub.loc[idx, "ALBUM_MRELG_ID"])


def compute_historical_single_momentum_table(
    album_keys: pd.DataFrame,
    lead_mapping: pd.DataFrame,
    week1_volumes: pd.DataFrame,
    momentum_col: str = HISTORICAL_SINGLE_MOMENTUM_COL,
) -> pd.DataFrame:
    """
    Per-album historical momentum: sum of Week-1 AE for prior album's lead singles.

    album_keys must include ALBUM_MRELG_ID, DISPLAY_ARTIST, ALBUM_FIRST_SALE_DATE.
    """
    keys = album_keys[
        ["ALBUM_MRELG_ID", "DISPLAY_ARTIST", "ALBUM_FIRST_SALE_DATE"]
    ].drop_duplicates(subset=["ALBUM_MRELG_ID"])
    keys = keys.copy()
    keys["ALBUM_FIRST_SALE_DATE"] = pd.to_datetime(
        keys["ALBUM_FIRST_SALE_DATE"], errors="coerce"
    )
    keys["artist_key"] = keys["DISPLAY_ARTIST"].str.lower()

    album_timeline = keys[["ALBUM_MRELG_ID", "artist_key", "ALBUM_FIRST_SALE_DATE"]].copy()

    lead_by_album = lead_mapping.groupby("ALBUM_MRELG_ID", sort=False)["MRELG_ID"].apply(
        list
    ).to_dict()

    w1 = week1_volumes.set_index("MRELG_ID")["week1_volume"].to_dict()

    rows = []
    for _, row in keys.iterrows():
        album_id = str(row["ALBUM_MRELG_ID"])
        artist = str(row["DISPLAY_ARTIST"])
        album_date = row["ALBUM_FIRST_SALE_DATE"]
        prior_id = _prior_album_id(artist, album_date, album_timeline)
        if prior_id is None:
            momentum = 0.0
        else:
            single_ids = lead_by_album.get(prior_id, [])
            momentum = float(
                sum(w1.get(sid, 0.0) for sid in single_ids)
            )
        rows.append(
            {
                "ALBUM_MRELG_ID": album_id,
                momentum_col: momentum,
                "prior_album_mrelg_id": prior_id,
            }
        )

    return pd.DataFrame(rows)


def attach_historical_single_momentum(
    album_df: pd.DataFrame,
    *,
    lead_path: Path = DEFAULT_LEAD_PATH,
    weekly_path: Path = DEFAULT_WEEKLY_PATH,
    metric_col: str = SINGLES_ROLLOUT_METRIC_COL,
    momentum_col: str = HISTORICAL_SINGLE_MOMENTUM_COL,
) -> pd.DataFrame:
    """Merge per-album historical_single_momentum onto album_df."""
    out = album_df.copy()
    if "ALBUM_MRELG_ID" not in out.columns:
        raise ValueError("album_df must contain ALBUM_MRELG_ID")

    lead = load_lead_single_mapping(lead_path)
    single_ids = set(lead["MRELG_ID"].unique())
    week1 = build_single_week1_volumes(
        weekly_path, metric_col=metric_col, single_ids=single_ids
    )

    if "DISPLAY_ARTIST" not in out.columns or "ALBUM_FIRST_SALE_DATE" not in out.columns:
        raise ValueError(
            "album_df must contain DISPLAY_ARTIST and ALBUM_FIRST_SALE_DATE "
            "to compute historical single momentum from prior albums"
        )

    mom_table = compute_historical_single_momentum_table(
        out[["ALBUM_MRELG_ID", "DISPLAY_ARTIST", "ALBUM_FIRST_SALE_DATE"]],
        lead,
        week1,
        momentum_col=momentum_col,
    )

    out = out.drop(columns=[momentum_col], errors="ignore")
    out = out.merge(
        mom_table[["ALBUM_MRELG_ID", momentum_col]],
        on="ALBUM_MRELG_ID",
        how="left",
    )
    out[momentum_col] = out[momentum_col].fillna(0.0)
    return out


def albums_with_lead_singles(lead_mapping: pd.DataFrame) -> set[str]:
    """Album MRELG IDs that have at least one mapped pre-album lead single."""
    return set(lead_mapping["ALBUM_MRELG_ID"].astype(str).str.strip().unique())


def load_artist_album_history(
    firstweekae_path: Path = DEFAULT_FIRSTWEEKAE_PATH,
) -> pd.DataFrame:
    """
    Artist album timeline from firstweekae (canonical first-week volumes).

    Columns: ALBUM_MRELG_ID, DISPLAY_ARTIST, FIRST_SALE_DATE, FIRST_WEEK_TOTAL_AE, artist_key.
    """
    if not firstweekae_path.exists():
        return pd.DataFrame(
            columns=[
                "ALBUM_MRELG_ID",
                "DISPLAY_ARTIST",
                "FIRST_SALE_DATE",
                "FIRST_WEEK_TOTAL_AE",
                "artist_key",
            ]
        )

    hist = pd.read_csv(
        firstweekae_path,
        encoding="utf-8-sig",
        usecols=["MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"],
    )
    hist = hist.rename(columns={"MRELG_ID": "ALBUM_MRELG_ID"})
    hist["ALBUM_MRELG_ID"] = hist["ALBUM_MRELG_ID"].astype(str).str.strip()
    hist["DISPLAY_ARTIST"] = hist["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    hist["FIRST_SALE_DATE"] = pd.to_datetime(hist["FIRST_SALE_DATE"], errors="coerce")
    hist["FIRST_WEEK_TOTAL_AE"] = pd.to_numeric(
        hist["FIRST_WEEK_TOTAL_AE"], errors="coerce"
    )
    hist = hist.dropna(subset=["FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"])
    hist = hist[(hist["ALBUM_MRELG_ID"] != "") & (hist["DISPLAY_ARTIST"] != "")]
    hist["artist_key"] = hist["DISPLAY_ARTIST"].str.lower()
    return hist


def _prior_album_row(
    artist_name: str,
    album_date: pd.Timestamp,
    album_history: pd.DataFrame,
) -> Optional[pd.Series]:
    """Most recent firstweekae album strictly before album_date."""
    artist_key = artist_name.strip().lower()
    sub = album_history[
        (album_history["artist_key"] == artist_key)
        & (album_history["FIRST_SALE_DATE"] < album_date)
    ]
    if sub.empty:
        return None
    return sub.loc[sub["FIRST_SALE_DATE"].idxmax()]


def lookup_prior_no_lead_single_era_momentum(
    artist_name: str,
    album_date: pd.Timestamp | str,
    *,
    lead_path: Path = DEFAULT_LEAD_PATH,
    firstweekae_path: Path = DEFAULT_FIRSTWEEKAE_PATH,
) -> Tuple[float, str]:
    """
    Max first-week AE among prior albums that had no mapped lead singles.

    Used when the current rollout has no pre-release singles (catalog / baseline path).
    """
    album_ts = pd.Timestamp(album_date)
    lead = load_lead_single_mapping(lead_path)
    with_leads = albums_with_lead_singles(lead)
    history = load_artist_album_history(firstweekae_path)

    artist_key = artist_name.strip().lower()
    prior = history[
        (history["artist_key"] == artist_key)
        & (history["FIRST_SALE_DATE"] < album_ts)
        & (~history["ALBUM_MRELG_ID"].isin(with_leads))
    ]
    if prior.empty:
        return 0.0, "no prior no-lead-single albums in history"

    peak = float(prior["FIRST_WEEK_TOTAL_AE"].max())
    n = int(len(prior))
    return peak, f"max first-week AE from {n} prior album(s) without lead singles"


def lookup_historical_single_momentum(
    artist_name: str,
    album_date: pd.Timestamp | str,
    *,
    lead_path: Path = DEFAULT_LEAD_PATH,
    weekly_path: Path = DEFAULT_WEEKLY_PATH,
    firstweekae_path: Path = DEFAULT_FIRSTWEEKAE_PATH,
    metric_col: str = SINGLES_ROLLOUT_METRIC_COL,
    no_lead_singles_rollout: bool = False,
) -> Tuple[float, Optional[str], str]:
    """
    Historical momentum for one artist + album date (what-if / Streamlit).

    When ``no_lead_singles_rollout`` is True, anchor on prior albums that also had
    no lead singles (first-week AE), not prior single Week-1 volumes.

    Otherwise: sum prior album's lead-single Week-1 volumes; if the prior album had
    no lead singles, fall back to that album's first-week total AE.

    Returns (momentum, prior_album_mrelg_id or None, human-readable source).
    """
    album_ts = pd.Timestamp(album_date)
    lead = load_lead_single_mapping(lead_path)
    with_leads = albums_with_lead_singles(lead)
    history = load_artist_album_history(firstweekae_path)

    if no_lead_singles_rollout:
        momentum, source = lookup_prior_no_lead_single_era_momentum(
            artist_name,
            album_ts,
            lead_path=lead_path,
            firstweekae_path=firstweekae_path,
        )
        prior_row = _prior_album_row(artist_name, album_ts, history)
        prior_id = str(prior_row["ALBUM_MRELG_ID"]) if prior_row is not None else None
        return momentum, prior_id, source

    week1 = build_single_week1_volumes(
        weekly_path,
        metric_col=metric_col,
        single_ids=set(lead["MRELG_ID"].unique()),
    )

    album_timeline = (
        lead[["ALBUM_MRELG_ID", "DISPLAY_ARTIST", "ALBUM_FIRST_SALE_DATE"]]
        .drop_duplicates(subset=["ALBUM_MRELG_ID"])
        .copy()
    )
    album_timeline["artist_key"] = album_timeline["DISPLAY_ARTIST"].str.lower()

    prior_id = _prior_album_id(artist_name, album_ts, album_timeline)
    if prior_id is None:
        prior_row = _prior_album_row(artist_name, album_ts, history)
        if prior_row is None:
            return 0.0, None, "no prior albums"
        fw = float(prior_row["FIRST_WEEK_TOTAL_AE"])
        pid = str(prior_row["ALBUM_MRELG_ID"])
        return fw, pid, "prior album first-week AE (no cohort lead-single timeline)"

    prior_singles = lead.loc[lead["ALBUM_MRELG_ID"] == prior_id, "MRELG_ID"].tolist()
    w1 = week1.set_index("MRELG_ID")["week1_volume"].to_dict()
    momentum = float(sum(w1.get(sid, 0.0) for sid in prior_singles))

    if momentum > 0:
        return momentum, prior_id, "prior album lead-single Week-1 sum"

    if prior_id not in with_leads:
        prior_row = history[history["ALBUM_MRELG_ID"] == prior_id]
        if not prior_row.empty:
            fw = float(prior_row.iloc[0]["FIRST_WEEK_TOTAL_AE"])
            return fw, prior_id, "prior album had no lead singles — using first-week AE"

    prior_row = _prior_album_row(artist_name, album_ts, history)
    if prior_row is not None:
        fw = float(prior_row["FIRST_WEEK_TOTAL_AE"])
        return fw, str(prior_row["ALBUM_MRELG_ID"]), "fallback: prior album first-week AE"

    return 0.0, prior_id, "no prior single or album momentum found"
