#!/usr/bin/env python3
"""
Basic lag-1 linear regression for future first-week album units.

Rules:
- Uses only past FIRST_WEEK_TOTAL_AE from the same artist (lag_1 feature).
- Uses artists with at least 3 unique MRELG_ID releases.
- Trains on all lag-eligible releases except each artist's latest one.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


DATA_PATH = Path("data/firstweekae.csv")
MIN_UNIQUE_RELEASES = 3
MIN_TRAIN_SAMPLES_PER_ARTIST = 3
HIGH_VOLUME_THRESHOLD = 15000.0
DATE_FMT = "%Y-%m-%d"
MAX_LOG_PRED = 20.0


@dataclass
class Release:
    artist: str
    mrelg_id: str
    sale_date: datetime
    volume: float


@dataclass
class Sample:
    artist: str
    mrelg_id: str
    sale_date: datetime
    lag_1: float
    target: float


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, DATE_FMT)
    except (TypeError, ValueError):
        return None


def load_releases(path: Path) -> List[Release]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    releases: List[Release] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"DISPLAY_ARTIST", "MRELG_ID", "FIRST_SALE_DATE", "FIRST_WEEK_TOTAL_AE"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            artist = (row.get("DISPLAY_ARTIST") or "").strip()
            mrelg_id = (row.get("MRELG_ID") or "").strip()
            sale_date = parse_date((row.get("FIRST_SALE_DATE") or "").strip())
            volume = parse_float((row.get("FIRST_WEEK_TOTAL_AE") or "").strip())
            if not artist or not mrelg_id or sale_date is None or volume is None:
                continue
            releases.append(Release(artist=artist, mrelg_id=mrelg_id, sale_date=sale_date, volume=volume))
    return releases


def build_samples(releases: List[Release], min_unique_releases: int) -> List[Sample]:
    by_artist: dict[str, List[Release]] = defaultdict(list)
    unique_mrelg_per_artist: dict[str, set[str]] = defaultdict(set)

    for r in releases:
        by_artist[r.artist].append(r)
        unique_mrelg_per_artist[r.artist].add(r.mrelg_id)

    valid_artists = {
        artist
        for artist, mrelg_ids in unique_mrelg_per_artist.items()
        if len(mrelg_ids) >= min_unique_releases
    }

    samples: List[Sample] = []
    for artist in valid_artists:
        ordered = sorted(by_artist[artist], key=lambda r: (r.sale_date, r.mrelg_id))
        for idx in range(1, len(ordered)):
            prev_release = ordered[idx - 1]
            curr_release = ordered[idx]
            samples.append(
                Sample(
                    artist=artist,
                    mrelg_id=curr_release.mrelg_id,
                    sale_date=curr_release.sale_date,
                    lag_1=prev_release.volume,
                    target=curr_release.volume,
                )
            )
    return samples


def split_train_test(samples: List[Sample]) -> tuple[List[Sample], List[Sample]]:
    by_artist: dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_artist[sample.artist].append(sample)

    train: List[Sample] = []
    test: List[Sample] = []
    for artist_samples in by_artist.values():
        ordered = sorted(artist_samples, key=lambda s: (s.sale_date, s.mrelg_id))
        if len(ordered) < (MIN_TRAIN_SAMPLES_PER_ARTIST + 1):
            continue
        test.append(ordered[-1])
        train.extend(ordered[:-1])
    return train, test


def fit_simple_linear_regression(xs: List[float], ys: List[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("xs and ys must be non-empty and same length")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if sxx == 0:
        return mean_y, 0.0

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return intercept, slope


def predict(intercept: float, slope: float, x: float) -> float:
    return intercept + slope * x


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def fit_artist_models(train_samples: List[Sample]) -> dict[str, tuple[float, float]]:
    """Fit a separate lag-1 linear model per artist in log1p space."""
    by_artist_x: dict[str, List[float]] = defaultdict(list)
    by_artist_y: dict[str, List[float]] = defaultdict(list)
    for s in train_samples:
        by_artist_x[s.artist].append(math.log1p(max(s.lag_1, 0.0)))
        by_artist_y[s.artist].append(math.log1p(max(s.target, 0.0)))

    models: dict[str, tuple[float, float]] = {}
    for artist, xs in by_artist_x.items():
        ys = by_artist_y[artist]
        models[artist] = fit_simple_linear_regression(xs, ys)
    return models


def filter_high_volume_artists(
    releases: List[Release], threshold: float
) -> tuple[List[Release], set[str]]:
    """
    Keep artists whose max FIRST_WEEK_TOTAL_AE is strictly above threshold.
    Returns filtered releases and the selected artist set.
    """
    max_by_artist: dict[str, float] = defaultdict(float)
    for r in releases:
        if r.volume > max_by_artist[r.artist]:
            max_by_artist[r.artist] = r.volume

    selected = {artist for artist, max_v in max_by_artist.items() if max_v > threshold}
    filtered = [r for r in releases if r.artist in selected]
    return filtered, selected


def evaluate_run(tag: str, releases: List[Release]) -> None:
    """Train/evaluate artist-specific models and print summary."""
    samples = build_samples(releases, min_unique_releases=MIN_UNIQUE_RELEASES)
    if not samples:
        print(f"{tag}: no lag-based samples after filtering.")
        return

    train_samples, test_samples = split_train_test(samples)
    if not train_samples or not test_samples:
        print(f"{tag}: insufficient train/test samples after split.")
        return

    models = fit_artist_models(train_samples)
    y_test = [s.target for s in test_samples]
    # Predict in log1p space, then transform back with expm1.
    y_pred = []
    for s in test_samples:
        intercept, slope = models[s.artist]
        x_log = math.log1p(max(s.lag_1, 0.0))
        pred_log = predict(intercept, slope, x_log)
        pred_log = min(pred_log, MAX_LOG_PRED)
        pred_units = math.expm1(pred_log)
        # Guardrail for negative values from linear model in log space.
        y_pred.append(max(pred_units, 0.0))

    # Baseline for each artist-specific row: predict last release directly.
    y_pred_last = [s.lag_1 for s in test_samples]

    artists_in_samples = {s.artist for s in samples}
    print(f"=== {tag} ===")
    print(f"Artists with >= {MIN_UNIQUE_RELEASES} distinct MRELG_ID: {len(artists_in_samples):,}")
    print(f"Lagged samples: {len(samples):,}")
    print(f"Train rows: {len(train_samples):,}")
    print(f"Test rows: {len(test_samples):,}")
    print("Metrics on held-out latest release per artist (units scale):")
    print(f"Artist-specific regression MAE:  {mae(y_test, y_pred):,.2f}")
    print(f"Artist-specific regression RMSE: {rmse(y_test, y_pred):,.2f}")
    print(f"Last-release baseline MAE:       {mae(y_test, y_pred_last):,.2f}")
    print(f"Last-release baseline RMSE:      {rmse(y_test, y_pred_last):,.2f}")
    print()
    print("Example predictions (first 10 test rows):")
    for row, pred in list(zip(test_samples, y_pred))[:10]:
        print(
            f"{row.artist} | {row.mrelg_id} | {row.sale_date.date()} "
            f"| actual={row.target:,.2f} | pred={pred:,.2f} | lag_1={row.lag_1:,.2f}"
        )
    print()


def main() -> None:
    releases = load_releases(DATA_PATH)
    artists_used = {r.artist for r in releases}
    print(f"Artists in raw file: {len(artists_used):,}")
    print()

    evaluate_run("All eligible artists (artist-specific model)", releases)

    high_vol_releases, high_vol_artists = filter_high_volume_artists(
        releases, threshold=HIGH_VOLUME_THRESHOLD
    )
    print(
        f"High-volume artist filter: max FIRST_WEEK_TOTAL_AE > {HIGH_VOLUME_THRESHOLD:,.0f} "
        f"(artists selected: {len(high_vol_artists):,})"
    )
    print()
    evaluate_run("High-volume artists only (artist-specific model)", high_vol_releases)


if __name__ == "__main__":
    main()
