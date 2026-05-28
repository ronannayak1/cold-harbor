#!/usr/bin/env python3
"""
Build pre-album datasets for lead-single analysis.

1) Singles weekly streams: filter streams_product_songs_ae_singles.parquet to
   SINGLE_MRELG_ID from lead_singles_with_album_within_6mo.csv, keep selected
   columns, preserve sequential week order per MRELG_ID.

2) Album first-week snapshot: filter firstweekae.csv to ALBUM_MRELG_ID from the
   same lead-singles mapping file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "pyarrow is required for parquet I/O. Install with: pip install pyarrow"
    ) from exc


LEAD_SINGLES_PATH = Path("data/lead_singles_with_album_within_6mo.csv")
STREAMS_SINGLES_PARQUET = Path("data/streams_product_songs_ae_singles.parquet")
FIRSTWEEKAE_PATH = Path("data/firstweekae.csv")

SINGLES_OUTPUT = Path("data/78_pre_album_singles_weekly.parquet")
ALBUMS_OUTPUT = Path("data/78_pre_album_albums_firstweek.csv")

SINGLE_STREAM_COLS = [
    "MRELG_ID",
    "TITLE",
    "DISPLAY_ARTIST",
    "GENRES",
    "FIRST_SALE_DATE",
    "WEEK_END_DATE",
    "WEEKS_SINCE_RELEASE",
    "PRODUCT_SALES",
    "STREAMING_EQUIVALENT",
    "SONG_SALE_EQUIVALENT",
    "TOTAL_ALBUM_EQUIVALENTS",
]

ALBUM_FIRSTWEEK_COLS = [
    "MRELG_ID",
    "TITLE",
    "DISPLAY_ARTIST",
    "GENRES",
    "FIRST_SALE_DATE",
    "FIRST_WEEK_PRODUCT_SALES",
    "FIRST_WEEK_STREAMING_EQUIVALENT",
    "FIRST_WEEK_SONG_SALE_EQUIVALENT",
    "FIRST_WEEK_TOTAL_AE",
]


def load_lead_single_ids() -> tuple[set[str], set[str]]:
    lead = pd.read_csv(LEAD_SINGLES_PATH, encoding="utf-8-sig", usecols=["SINGLE_MRELG_ID", "ALBUM_MRELG_ID"])
    single_ids = set(lead["SINGLE_MRELG_ID"].astype(str).str.strip().dropna().unique())
    album_ids = set(lead["ALBUM_MRELG_ID"].astype(str).str.strip().dropna().unique())
    single_ids.discard("")
    album_ids.discard("")
    return single_ids, album_ids


def build_singles_weekly(single_ids: set[str]) -> pd.DataFrame:
    if not STREAMS_SINGLES_PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet input: {STREAMS_SINGLES_PARQUET}")

    # Read only needed columns; filter to lead singles.
    streams = pd.read_parquet(STREAMS_SINGLES_PARQUET, columns=SINGLE_STREAM_COLS)
    missing = set(SINGLE_STREAM_COLS) - set(streams.columns)
    if missing:
        raise ValueError(
            f"{STREAMS_SINGLES_PARQUET} missing columns: {sorted(missing)}"
        )

    streams["MRELG_ID"] = streams["MRELG_ID"].astype(str).str.strip()
    filtered = streams[streams["MRELG_ID"].isin(single_ids)].copy()

    # Preserve sequential weekly order within each release.
    filtered["WEEKS_SINCE_RELEASE"] = pd.to_numeric(
        filtered["WEEKS_SINCE_RELEASE"], errors="coerce"
    )
    filtered["WEEK_END_DATE"] = pd.to_datetime(filtered["WEEK_END_DATE"], errors="coerce")
    filtered = filtered.sort_values(
        ["MRELG_ID", "WEEKS_SINCE_RELEASE", "WEEK_END_DATE"],
        kind="mergesort",
    ).reset_index(drop=True)

    return filtered[SINGLE_STREAM_COLS]


def build_albums_firstweek(album_ids: set[str]) -> pd.DataFrame:
    if not FIRSTWEEKAE_PATH.exists():
        raise FileNotFoundError(f"Missing csv input: {FIRSTWEEKAE_PATH}")

    albums = pd.read_csv(FIRSTWEEKAE_PATH, encoding="utf-8-sig", usecols=ALBUM_FIRSTWEEK_COLS)
    missing = set(ALBUM_FIRSTWEEK_COLS) - set(albums.columns)
    if missing:
        raise ValueError(f"{FIRSTWEEKAE_PATH} missing columns: {sorted(missing)}")

    albums["MRELG_ID"] = albums["MRELG_ID"].astype(str).str.strip()
    filtered = albums[albums["MRELG_ID"].isin(album_ids)].copy()
    filtered["FIRST_SALE_DATE"] = pd.to_datetime(filtered["FIRST_SALE_DATE"], errors="coerce")
    filtered = filtered.sort_values(
        ["MRELG_ID", "FIRST_SALE_DATE"], kind="mergesort"
    ).reset_index(drop=True)

    return filtered[ALBUM_FIRSTWEEK_COLS]


def main() -> None:
    single_ids, album_ids = load_lead_single_ids()
    print(f"Lead singles to extract: {len(single_ids):,}")
    print(f"Lead albums to extract: {len(album_ids):,}")

    singles_weekly = build_singles_weekly(single_ids)
    albums_firstweek = build_albums_firstweek(album_ids)

    SINGLES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    singles_weekly.to_parquet(SINGLES_OUTPUT, index=False)
    albums_firstweek.to_csv(ALBUMS_OUTPUT, index=False)

    print()
    print("=== Outputs ===")
    print(f"Singles weekly rows: {len(singles_weekly):,}")
    print(f"Distinct single MRELG_ID in output: {singles_weekly['MRELG_ID'].nunique():,}")
    print(f"Wrote: {SINGLES_OUTPUT}")
    print()
    print(f"Album first-week rows: {len(albums_firstweek):,}")
    print(f"Distinct album MRELG_ID in output: {albums_firstweek['MRELG_ID'].nunique():,}")
    print(f"Wrote: {ALBUMS_OUTPUT}")

    missing_singles = len(single_ids - set(singles_weekly["MRELG_ID"].unique()))
    missing_albums = len(album_ids - set(albums_firstweek["MRELG_ID"].unique()))
    if missing_singles:
        print(f"Warning: {missing_singles:,} SINGLE_MRELG_ID not found in parquet.")
    if missing_albums:
        print(f"Warning: {missing_albums:,} ALBUM_MRELG_ID not found in firstweekae.csv.")


if __name__ == "__main__":
    main()
