#!/usr/bin/env python3
"""
run_inference_pipeline.py
=========================
Production inference for the Two-Stage Hurdle Architecture (album-level).

Hard cutoff strategy (when dynamic_threshold is in model meta):
  - Stage 1: classifier probability > prob_threshold (trained on actual >= P75).
  - Stage 2: expm1(regressor) >= dynamic_threshold (regressor trained only above P75).
  - Both must pass to use the regressor; otherwise carryover + fallback (capped at P75).

Use --target to switch total_ae | streaming | product | song.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hurdle_regressor import load_regressor, predict_log_scale, resolve_regressor_path
from target_config import (
    TargetConfig,
    add_target_cli,
    build_training_feature_cols,
    get_target_config,
)

# Stage 1 probability gate (classifier predicts P(actual >= training P75)).
CLASSIFIER_THRESHOLD = 0.50
FALLBACK_BASE = 1500.0


def load_hurdle_meta(models_dir: Path) -> dict:
    """Load hurdle_model_meta.json; empty dict if missing."""
    meta_path = models_dir / "hurdle_model_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_dynamic_threshold(meta: dict) -> float | None:
    """Training P75 gatekeeper cutoff from saved metadata."""
    raw = meta.get("dynamic_threshold", meta.get("major_threshold"))
    if raw is None:
        return None
    value = float(raw)
    return value if value > 0 else None


def apply_hurdle_cutoff(
    prob_major: np.ndarray,
    high_tier_preds: np.ndarray,
    carryover: np.ndarray,
    *,
    prob_threshold: float,
    fallback_base: float,
    dynamic_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine classifier + regressor paths with optional P75 volume gate.

    Returns (final_prediction, use_regressor, passes_classifier, passes_volume).
    """
    fallback = carryover + fallback_base
    passes_classifier = prob_major > prob_threshold

    if dynamic_threshold is not None:
        fallback = np.minimum(fallback, dynamic_threshold)
        passes_volume = high_tier_preds >= dynamic_threshold
        use_regressor = passes_classifier & passes_volume
    else:
        passes_volume = np.ones(len(prob_major), dtype=bool)
        use_regressor = passes_classifier

    final = np.where(use_regressor, high_tier_preds, fallback)
    final = np.clip(final, 0.0, None)
    return final, use_regressor, passes_classifier, passes_volume


class HurdleModel:
    """Two-stage hurdle model: classifier gate + high-tier regressor."""

    def __init__(
        self,
        classifier_path: Path,
        regressor_path: Path,
        feature_cols: list[str],
        threshold: float = CLASSIFIER_THRESHOLD,
        fallback_base: float = FALLBACK_BASE,
        dynamic_threshold: float | None = None,
    ) -> None:
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        reg_path = resolve_regressor_path(regressor_path.parent)
        if not reg_path.exists():
            raise FileNotFoundError(f"Regressor model not found: {reg_path}")
        self.classifier = lgb.Booster(model_file=str(classifier_path))
        self.regressor = load_regressor(reg_path)
        self._regressor_is_legacy = reg_path.suffix == ".txt"
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.fallback_base = fallback_base
        self.dynamic_threshold = dynamic_threshold

    def predict_values(
        self,
        feature_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (final_prediction, regressor_expm1, classifier_probability) per row.
        """
        X = feature_df[self.feature_cols].copy()
        X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")
        carryover = feature_df["expected_week1_carryover"].to_numpy(dtype=float)
        prob_major = self.classifier.predict(X)
        if self._regressor_is_legacy:
            high_tier_preds = np.expm1(self.regressor.predict(X))
        else:
            high_tier_preds = np.expm1(predict_log_scale(self.regressor, X))
        final, _, _, _ = apply_hurdle_cutoff(
            prob_major,
            high_tier_preds,
            carryover,
            prob_threshold=self.threshold,
            fallback_base=self.fallback_base,
            dynamic_threshold=self.dynamic_threshold,
        )
        return final, high_tier_preds, prob_major

    def predict_albums(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run two-stage inference with hard binary cutoff.

        Input must be album-level (keyed by ALBUM_MRELG_ID).
        Returns a scored DataFrame sorted descending by Final_Album_Prediction.
        """
        if "ALBUM_MRELG_ID" not in feature_df.columns:
            raise ValueError(
                "Input data must be aggregated at the album level. "
                "Column 'ALBUM_MRELG_ID' not found — ensure you are loading "
                "the Phase 2 album_meta_features output, not the track-level panel."
            )

        missing = [c for c in self.feature_cols if c not in feature_df.columns]
        if missing:
            raise ValueError(
                f"Feature alignment error: missing columns {missing}. "
                f"Ensure input matches album_meta_features for this --target."
            )

        X = feature_df[self.feature_cols].copy()
        X["Archetype_Cluster"] = X["Archetype_Cluster"].astype("category")

        carryover = feature_df["expected_week1_carryover"].to_numpy(dtype=float)
        final_prediction, high_tier_preds, prob_major = self.predict_values(feature_df)
        final_prediction = final_prediction.round().astype(int)

        _, use_regressor, passes_clf, passes_vol = apply_hurdle_cutoff(
            prob_major,
            high_tier_preds,
            carryover,
            prob_threshold=self.threshold,
            fallback_base=self.fallback_base,
            dynamic_threshold=self.dynamic_threshold,
        )

        n_total = len(feature_df)
        n_clf = int(passes_clf.sum())
        n_reg = int(use_regressor.sum())
        print(f"  Albums scored: {n_total:,}")
        print(f"  Pass classifier (prob > {self.threshold:.2f}): "
              f"{n_clf:,} ({100*n_clf/max(n_total,1):.1f}%)")
        if self.dynamic_threshold is not None:
            n_vol = int(passes_vol.sum())
            print(f"  Pass volume gate (>= P75 {self.dynamic_threshold:,.2f}): "
                  f"{n_vol:,} ({100*n_vol/max(n_total,1):.1f}%)")
            print(f"  Using regressor (both gates): "
                  f"{n_reg:,} ({100*n_reg/max(n_total,1):.1f}%)")
        else:
            print(f"  Using regressor: {n_reg:,} ({100*n_reg/max(n_total,1):.1f}%)")

        hist_col = (
            "max_historical_week1_volume"
            if "max_historical_week1_volume" in feature_df.columns
            else next(
                (c for c in feature_df.columns if c.startswith("max_historical")),
                "max_historical_week1_volume",
            )
        )
        debut_col = (
            "is_debut_studio_album"
            if "is_debut_studio_album" in feature_df.columns
            else "is_debut_album"
        )

        result = pd.DataFrame({
            "MRELG_ID_ALBUM": feature_df["ALBUM_MRELG_ID"].values,
            "DISPLAY_ARTIST": feature_df["DISPLAY_ARTIST"].values,
            "Historical_Max": feature_df[hist_col].to_numpy(dtype=float).round(1)
            if hist_col in feature_df.columns
            else 0.0,
            "Is_Debut": feature_df[debut_col].to_numpy(dtype=int)
            if debut_col in feature_df.columns
            else 0,
            "Expected_Carryover_From_Singles": carryover.round(1),
            "Classifier_Probability": prob_major.round(6),
            "Passes_Classifier": passes_clf.astype(int),
            "Regressor_Prediction": high_tier_preds.round(1),
            "Passes_Volume_Gate": passes_vol.astype(int),
            "Dynamic_Threshold": self.dynamic_threshold
            if self.dynamic_threshold is not None
            else np.nan,
            "Is_Priority_Rollout": use_regressor.astype(int),
            "Final_Album_Prediction": final_prediction,
        })

        result = result.sort_values(
            "Final_Album_Prediction", ascending=False
        ).reset_index(drop=True)
        return result


def load_features(path: Path) -> pd.DataFrame:
    """Load album-level rollout features with validation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. "
            "Run singles_archetypes.py --phase 2 for this --target first."
        )
    df = pd.read_parquet(path)

    if "ALBUM_MRELG_ID" not in df.columns:
        raise ValueError(
            f"Input file {path} does not contain 'ALBUM_MRELG_ID'. "
            "This does not appear to be album-level data."
        )

    df["Archetype_Cluster"] = df["Archetype_Cluster"].astype("category")
    df["ALBUM_MRELG_ID"] = df["ALBUM_MRELG_ID"].astype(str).str.strip()
    return df


def load_feature_cols(cfg: TargetConfig, feature_df: pd.DataFrame) -> list[str]:
    """Prefer feature list from saved model meta; else build from config."""
    meta_path = cfg.models_dir / "hurdle_model_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if "feature_cols" in meta:
            cols = meta["feature_cols"]
            missing = [c for c in cols if c not in feature_df.columns]
            if not missing:
                return cols
    return build_training_feature_cols(feature_df, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Two-Stage Hurdle inference on album-level features"
    )
    add_target_cli(parser)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--classifier-path", type=Path, default=None)
    parser.add_argument("--regressor-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_target_config(args.target)
    feature_path = args.feature_path or cfg.features_path
    output_path = args.output or cfg.inference_output_path
    classifier_path = args.classifier_path or cfg.classifier_path
    regressor_path = args.regressor_path or cfg.regressor_path

    meta = load_hurdle_meta(cfg.models_dir)
    dynamic_threshold = resolve_dynamic_threshold(meta)

    print("=" * 65)
    print(f"Two-Stage Hurdle Inference — target={cfg.name}")
    if dynamic_threshold is not None:
        print(f"  Volume gate (training P75): {dynamic_threshold:,.2f}")
        print(f"  Regressor used when: prob > {args.threshold} AND expm1(pred) >= P75")
        print(f"  Fallback: min(carryover + {FALLBACK_BASE:,.0f}, P75)")
    else:
        print("  Volume gate: not in meta (classifier prob only)")
        print(f"  Regressor used when: prob > {args.threshold}")
        print(f"  Fallback: carryover + {FALLBACK_BASE:,.0f}")
    print(f"  Stage 2 magnitude: expm1(log prediction)")
    print(f"  Features: {feature_path}")
    print(f"  Models: {cfg.models_dir}")
    print("=" * 65)

    print("\n[1/3] Loading models and features...")
    features = load_features(feature_path)
    feature_cols = load_feature_cols(cfg, features)
    print(f"  Feature columns ({len(feature_cols)})")

    model = HurdleModel(
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        feature_cols=feature_cols,
        threshold=args.threshold,
        dynamic_threshold=dynamic_threshold,
    )
    print("  Classifier loaded.")
    print("  Regressor loaded.")

    print("\n[2/3] Running inference...")
    print(f"  Input albums: {len(features):,}")
    results = model.predict_albums(features)

    print("\n[3/3] Output summary:")
    priority = results[results["Is_Priority_Rollout"] == 1]
    non_priority = results[results["Is_Priority_Rollout"] == 0]

    if not priority.empty:
        print(f"  Priority rollouts: {len(priority):,}")
        print(f"  Priority prediction range: "
              f"{priority['Final_Album_Prediction'].min():,} – "
              f"{priority['Final_Album_Prediction'].max():,}")
        print(f"  Priority mean prediction:  "
              f"{priority['Final_Album_Prediction'].mean():,.0f}")
    else:
        print("  Priority rollouts: 0 (no albums exceeded threshold)")

    print(f"  Non-priority albums: {len(non_priority):,}")
    if not non_priority.empty:
        fb_desc = (
            f"min(carryover + {FALLBACK_BASE:,.0f}, P75)"
            if dynamic_threshold is not None
            else f"carryover + {FALLBACK_BASE:,.0f}"
        )
        print(f"  Non-priority mean prediction: "
              f"{non_priority['Final_Album_Prediction'].mean():,.0f} ({fb_desc})")

    print(f"\n  Top 20 predicted albums:")
    print("-" * 90)
    print(f"  {'#':<4} {'Artist':<28} {'Carryover':>10} {'Prob':>6} "
          f"{'Priority':>8} {'Prediction':>12}")
    print("-" * 90)
    for rank, (_, row) in enumerate(results.head(20).iterrows(), start=1):
        artist = str(row["DISPLAY_ARTIST"])[:26]
        pri = "YES" if row["Is_Priority_Rollout"] == 1 else "no"
        print(
            f"  {rank:<4} {artist:<28} "
            f"{row['Expected_Carryover_From_Singles']:>10,.1f} "
            f"{row['Classifier_Probability']:>6.3f} "
            f"{pri:>8} "
            f"{row['Final_Album_Prediction']:>12,}"
        )
    print("-" * 90)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path} ({len(results):,} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
