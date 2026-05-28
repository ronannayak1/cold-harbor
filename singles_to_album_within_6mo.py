#!/usr/bin/env python3
"""
Identify singles whose artist released at least one album within 6 months.

Inputs:
- data/lead_single_hunt.csv
- data/firstweekae.csv

Logic:
- Match on DISPLAY_ARTIST (case-insensitive, trimmed).
- Use FIRST_SALE_DATE for both single and album release date.
- Album must be after the single date and within 6 calendar months.
- Exclude exact same MRELG_ID when checking follow-up albums.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SINGLES_PATH = Path("data/lead_single_hunt.csv")
ALBUMS_PATH = Path("data/firstweekae.csv")
OUTPUT_PATH = Path("data/lead_singles_with_album_within_6mo.csv")


def _load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"MRELG_ID", "DISPLAY_ARTIST", "FIRST_SALE_DATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df[["MRELG_ID", "TITLE", "DISPLAY_ARTIST", "FIRST_SALE_DATE"]].copy()
    df["MRELG_ID"] = df["MRELG_ID"].astype(str).str.strip()
    df["DISPLAY_ARTIST"] = df["DISPLAY_ARTIST"].fillna("").astype(str).str.strip()
    df["ARTIST_KEY"] = df["DISPLAY_ARTIST"].str.lower()
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df = df.dropna(subset=["FIRST_SALE_DATE"])
    df = df[df["ARTIST_KEY"] != ""]
    df = df[df["MRELG_ID"] != ""]
    return df


def main() -> None:
    singles = _load_base(SINGLES_PATH).rename(
        columns={
            "MRELG_ID": "SINGLE_MRELG_ID",
            "TITLE": "SINGLE_TITLE",
            "DISPLAY_ARTIST": "SINGLE_DISPLAY_ARTIST",
            "FIRST_SALE_DATE": "SINGLE_FIRST_SALE_DATE",
        }
    )

    albums = _load_base(ALBUMS_PATH).rename(
        columns={
            "MRELG_ID": "ALBUM_MRELG_ID",
            "TITLE": "ALBUM_TITLE",
            "DISPLAY_ARTIST": "ALBUM_DISPLAY_ARTIST",
            "FIRST_SALE_DATE": "ALBUM_FIRST_SALE_DATE",
        }
    )

    # Artist-level join, then filter by valid 6-month forward window.
    merged = singles.merge(albums, on="ARTIST_KEY", how="left")
    merged = merged[merged["ALBUM_MRELG_ID"] != merged["SINGLE_MRELG_ID"]]
    merged = merged[merged["ALBUM_FIRST_SALE_DATE"] > merged["SINGLE_FIRST_SALE_DATE"]]

    horizon = merged["SINGLE_FIRST_SALE_DATE"] + pd.DateOffset(months=6)
    merged = merged[merged["ALBUM_FIRST_SALE_DATE"] <= horizon]

    # For each single, keep earliest qualifying follow-up album.
    matched = (
        merged.sort_values(
            ["SINGLE_MRELG_ID", "ALBUM_FIRST_SALE_DATE", "ALBUM_MRELG_ID"]
        )
        .groupby("SINGLE_MRELG_ID", as_index=False)
        .first()
    )

    matched["DAYS_TO_NEXT_ALBUM"] = (
        matched["ALBUM_FIRST_SALE_DATE"] - matched["SINGLE_FIRST_SALE_DATE"]
    ).dt.days

    out_cols = [
        "SINGLE_MRELG_ID",
        "SINGLE_TITLE",
        "SINGLE_DISPLAY_ARTIST",
        "SINGLE_FIRST_SALE_DATE",
        "ALBUM_MRELG_ID",
        "ALBUM_TITLE",
        "ALBUM_DISPLAY_ARTIST",
        "ALBUM_FIRST_SALE_DATE",
        "DAYS_TO_NEXT_ALBUM",
    ]
    matched[out_cols].to_csv(OUTPUT_PATH, index=False)

    total_singles = len(singles)
    matched_singles = matched["SINGLE_MRELG_ID"].nunique()
    pct = (matched_singles / total_singles * 100.0) if total_singles else 0.0

    print("=== Singles -> Album within 6 months ===")
    print(f"Singles considered: {total_singles:,}")
    print(f"Singles with >=1 matching album within 6 months: {matched_singles:,}")
    print(f"Match rate: {pct:.2f}%")
    print(f"Output written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
