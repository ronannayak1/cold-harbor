#!/usr/bin/env python3
"""
run_bottom_up_ensemble.py
=========================
Bottom-up reconciliation of Total AE from component hurdle models.

Runs Streaming, Song, and Product two-stage hurdle inference, applies a cold-start
cap on physical (product) sales for debut artists, and sums components:

    Final_Total_AE = pred_streaming + pred_song + pred_product
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from run_inference_pipeline import (
    CLASSIFIER_THRESHOLD,
    HurdleModel,
    load_feature_cols,
    load_features,
    resolve_dynamic_threshold,
)
from target_config import BOTTOM_UP_TARGETS, TargetConfig, get_target_config

JOIN_KEY = "ALBUM_MRELG_ID"
DEFAULT_OUTPUT = Path("models/bottom_up_ensemble/reconciled_total_ae.csv")
PRODUCT_DEBUT_CAP_FRACTION = 0.05
SUPERSTAR_THRESHOLD = 150_000.0
SUPERSTAR_FLOOR_RATIO = 0.65


def load_model_meta(cfg: TargetConfig) -> Dict:
    """Load hurdle_model_meta.json for a component target."""
    from run_inference_pipeline import load_hurdle_meta

    meta = load_hurdle_meta(cfg.models_dir)
    if not meta:
        raise FileNotFoundError(
            f"Missing model metadata for --target {cfg.name}: "
            f"{cfg.models_dir / 'hurdle_model_meta.json'}. "
            f"Run train_hurdle_model.py --target {cfg.name} first."
        )
    return meta


def load_hurdle_bundle(target_name: str) -> Tuple[HurdleModel, TargetConfig, pd.DataFrame, float | None]:
    """Load features, models, and training dynamic threshold for one component."""
    cfg = get_target_config(target_name)
    features = load_features(cfg.features_path)
    meta = load_model_meta(cfg)
    dynamic_threshold = resolve_dynamic_threshold(meta)
    feature_cols = load_feature_cols(cfg, features)

    model = HurdleModel(
        classifier_path=cfg.classifier_path,
        regressor_path=cfg.regressor_path,
        feature_cols=feature_cols,
        threshold=CLASSIFIER_THRESHOLD,
        dynamic_threshold=dynamic_threshold,
    )
    return model, cfg, features, dynamic_threshold


def run_component_inference(
    target_name: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Run hurdle inference for one bottom-up component; return scores + P75 gate."""
    model, cfg, features, dynamic_threshold = load_hurdle_bundle(target_name)
    if verbose:
        print(f"\n--- Component: {target_name} ({cfg.target_col}) ---")
        print(f"  Features: {cfg.features_path} ({len(features):,} albums)")
        if dynamic_threshold is not None:
            print(f"  Volume gate (P75): {dynamic_threshold:,.2f}")
        print(f"  Classifier prob cutoff: {CLASSIFIER_THRESHOLD}")

    scored = model.predict_albums(features)
    scored = scored.rename(
        columns={
            "Final_Album_Prediction": f"pred_{target_name}",
            "MRELG_ID_ALBUM": JOIN_KEY,
        }
    )
    keep = [JOIN_KEY, "DISPLAY_ARTIST", f"pred_{target_name}", "Classifier_Probability"]
    return scored[keep], dynamic_threshold


def debut_mask(df: pd.DataFrame) -> np.ndarray:
    """True where album is a debut (product or studio flag)."""
    debut = np.zeros(len(df), dtype=bool)
    if "is_debut_album" in df.columns:
        debut |= df["is_debut_album"].fillna(0).astype(int).to_numpy() == 1
    if "is_debut_studio_album" in df.columns:
        debut |= df["is_debut_studio_album"].fillna(0).astype(int).to_numpy() == 1
    return debut


def reconcile_bottom_up(
    streaming_df: pd.DataFrame,
    song_df: pd.DataFrame,
    product_df: pd.DataFrame,
    product_features: pd.DataFrame,
    streaming_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join component predictions and apply debut product cap.

    pred_product = min(pred_product_raw, pred_streaming * 0.05) for debuts.
    """
    merged = streaming_df.merge(song_df, on=[JOIN_KEY, "DISPLAY_ARTIST"], how="inner")
    merged = merged.merge(product_df, on=[JOIN_KEY, "DISPLAY_ARTIST"], how="inner")

    feat_cols = [JOIN_KEY]
    for c in ("is_debut_album", "is_debut_studio_album"):
        if c in product_features.columns:
            feat_cols.append(c)
    if len(feat_cols) > 1:
        merged = merged.merge(product_features[feat_cols].drop_duplicates(JOIN_KEY), on=JOIN_KEY, how="left")

    if "max_historical_week1_volume" in streaming_features.columns:
        baseline_df = streaming_features[[JOIN_KEY, "max_historical_week1_volume"]].drop_duplicates(
            JOIN_KEY
        )
        merged = merged.merge(baseline_df, on=JOIN_KEY, how="left")
    else:
        merged["max_historical_week1_volume"] = 0.0

    pred_streaming = merged["pred_streaming"].to_numpy(dtype=float)
    pred_song = merged["pred_song"].to_numpy(dtype=float)
    pred_product_raw = merged["pred_product"].to_numpy(dtype=float)

    is_debut = debut_mask(merged)
    cap = pred_streaming * PRODUCT_DEBUT_CAP_FRACTION
    pred_product = np.where(is_debut, np.minimum(pred_product_raw, cap), pred_product_raw)
    pred_product = np.clip(pred_product, 0.0, None)

    baseline_max = pd.to_numeric(
        merged["max_historical_week1_volume"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    final_total_raw = pred_streaming + pred_song + pred_product
    superstar_floor = baseline_max * SUPERSTAR_FLOOR_RATIO
    floor_mask = (baseline_max >= SUPERSTAR_THRESHOLD) & (final_total_raw < superstar_floor)
    final_total = np.where(floor_mask, superstar_floor, final_total_raw)

    out = pd.DataFrame({
        JOIN_KEY: merged[JOIN_KEY],
        "DISPLAY_ARTIST": merged["DISPLAY_ARTIST"],
        "pred_streaming": np.round(pred_streaming).astype(int),
        "pred_song": np.round(pred_song).astype(int),
        "pred_product_raw": np.round(pred_product_raw).astype(int),
        "pred_product": np.round(pred_product).astype(int),
        "Is_Debut_Capped": is_debut.astype(int),
        "max_historical_week1_volume": np.round(baseline_max, 1),
        "SUPERSTAR_FLOOR_APPLIED": floor_mask.astype(int),
        "Final_Total_AE": np.round(final_total).astype(int),
    })
    return out.sort_values("Final_Total_AE", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bottom-up Total AE = Streaming + Song + Product (with debut cap)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for reconciled output CSV",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-component logs")
    args = parser.parse_args()

    verbose = not args.quiet
    if verbose:
        print("=" * 65)
        print("Bottom-Up Ensemble — reconciled Total AE")
        print(f"  Components: {', '.join(BOTTOM_UP_TARGETS)}")
        print(f"  Debut product cap: {PRODUCT_DEBUT_CAP_FRACTION:.0%} of pred_streaming")
        print(f"  Stage 2: expm1 applied in hurdle inference")
        print("=" * 65)

    thresholds: Dict[str, float | None] = {}
    component_frames: Dict[str, pd.DataFrame] = {}

    for name in BOTTOM_UP_TARGETS:
        scored, p75 = run_component_inference(name, verbose=verbose)
        thresholds[name] = p75
        component_frames[name] = scored

    product_cfg = get_target_config("product")
    product_features = load_features(product_cfg.features_path)
    streaming_cfg = get_target_config("streaming")
    streaming_features = load_features(streaming_cfg.features_path)

    reconciled = reconcile_bottom_up(
        component_frames["streaming"],
        component_frames["song"],
        component_frames["product"],
        product_features,
        streaming_features,
    )

    if verbose:
        print("\n--- Reconciliation ---")
        print(f"  Albums with all three components: {len(reconciled):,}")
        print(f"  Debut-capped product rows: {int(reconciled['Is_Debut_Capped'].sum()):,}")
        print(
            f"  Superstar floor rows: {int(reconciled['SUPERSTAR_FLOOR_APPLIED'].sum()):,}"
        )
        print(f"  Final_Total_AE range: {reconciled['Final_Total_AE'].min():,} – "
              f"{reconciled['Final_Total_AE'].max():,}")
        gate_str = ", ".join(
            f"{k}={v:,.0f}" if v is not None else f"{k}=n/a"
            for k, v in thresholds.items()
        )
        print(f"\n  Training P75 gates: {gate_str}")
        print("\n  Top 15 reconciled albums:")
        print("-" * 90)
        for rank, row in reconciled.head(15).iterrows():
            artist = str(row["DISPLAY_ARTIST"])[:28]
            cap = " [capped]" if row["Is_Debut_Capped"] else ""
            print(
                f"  {rank + 1:>2}. {artist:<28} "
                f"S={row['pred_streaming']:>7,} "
                f"So={row['pred_song']:>6,} "
                f"P={row['pred_product']:>6,}{cap} "
                f"→ {row['Final_Total_AE']:>8,}"
            )
        print("-" * 90)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(args.output, index=False)

    meta_out = {
        "components": list(BOTTOM_UP_TARGETS),
        "dynamic_thresholds": thresholds,
        "classifier_probability_cutoff": CLASSIFIER_THRESHOLD,
        "product_debut_cap_fraction": PRODUCT_DEBUT_CAP_FRACTION,
        "n_albums": len(reconciled),
        "output_csv": str(args.output),
    }
    meta_path = args.output.parent / "bottom_up_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nSaved: {args.output} ({len(reconciled):,} rows)")
    print(f"Meta:  {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
