#!/usr/bin/env python3
"""
streamlit_whatif_app.py
=======================
Streamlit what-if tool for projecting album first-week volume.

Run:
  streamlit run streamlit_whatif_app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from album_projection import ManualSingle, synthesize_album_features
from run_inference_pipeline import HurdleModel, EXPECTED_FEATURES

st.set_page_config(page_title="Album First-Week Projection", layout="wide")

st.title("Album First-Week Projection (What-If)")
st.caption(
    "Simulate a pre-release rollout and score it with the Two-Stage Hurdle model."
)

with st.sidebar:
    st.header("Release setup")
    artist = st.text_input("Artist name", value="Taylor Swift")
    album_date = st.date_input("Expected album release date", value=date.today())
    threshold = st.slider("Classifier threshold", 0.05, 0.95, 0.50, 0.05)
    n_singles = st.number_input("Number of pre-release singles", 1, 5, 1)

st.subheader("Hypothetical singles")
singles: list[ManualSingle] = []
cols_header = st.columns([2, 2, 2, 2])
cols_header[0].markdown("**Weeks before album**")
cols_header[1].markdown("**Week 1 volume (AE)**")
cols_header[2].markdown("**Week 2 volume (AE)**")

for i in range(int(n_singles)):
    c1, c2, c3 = st.columns([2, 2, 2])
    weeks_before = c1.number_input(
        f"Single {i + 1} lead time",
        min_value=1,
        max_value=26,
        value=8 - i * 2,
        key=f"weeks_{i}",
        label_visibility="collapsed",
    )
    w1 = c2.number_input(
        f"W1_{i}",
        min_value=0.0,
        value=5000.0 * (1.2 - 0.1 * i),
        step=100.0,
        key=f"w1_{i}",
        label_visibility="collapsed",
    )
    w2 = c3.number_input(
        f"W2_{i}",
        min_value=0.0,
        value=4500.0 * (1.15 - 0.1 * i),
        step=100.0,
        key=f"w2_{i}",
        label_visibility="collapsed",
    )
    singles.append(
        ManualSingle(
            weeks_before_album=int(weeks_before),
            week1_volume=float(w1),
            week2_volume=float(w2),
        )
    )

if st.button("Run projection", type="primary"):
    try:
        feature_row = synthesize_album_features(artist, album_date, singles)
        model = HurdleModel(threshold=threshold)
        result = model.predict_albums(feature_row)

        st.success("Projection complete.")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Classifier probability", f"{result['Classifier_Probability'].iloc[0]:.3f}")
        col_b.metric(
            "Priority rollout?",
            "Yes" if result["Is_Priority_Rollout"].iloc[0] else "No",
        )
        col_c.metric(
            "Final album prediction",
            f"{result['Final_Album_Prediction'].iloc[0]:,}",
        )
        col_d.metric(
            "Historical max (artist)",
            f"{result['Historical_Max'].iloc[0]:,.0f}",
        )

        st.subheader("Synthesized features (model inputs)")
        display_cols = [
            "expected_week1_carryover",
            "composite_peak_momentum",
            "max_single_peak_volume",
            "total_pre_release_auc",
            "catalog_momentum_slope",
            "max_historical_week1_volume",
            "is_debut_studio_album",
            "Archetype_Cluster",
            "cannibalization_ratio",
        ]
        st.dataframe(feature_row[display_cols].T.rename(columns={0: "Value"}))

        st.subheader("Full feature vector (LightGBM order)")
        st.code(
            "\n".join(
                f"{c}: {feature_row[c].iloc[0]}"
                for c in EXPECTED_FEATURES
                if c in feature_row.columns
            ),
            language="text",
        )

        st.subheader("Scoring output")
        st.dataframe(result, use_container_width=True)

    except FileNotFoundError as exc:
        st.error(
            f"Missing model or artifact file: {exc}. "
            "Train models with `python train_hurdle_model.py` first."
        )
    except Exception as exc:
        st.exception(exc)

with st.expander("How this works"):
    st.markdown(
        """
        1. **Artist lookup** — `max_historical_week1_volume` and debut flag from
           `data/firstweekae.csv` (albums strictly before your release date).
           `catalog_momentum_slope` and default archetype from training artifacts.
        2. **Single synthesis** — Each single's W1/W2 volumes fit a gamma decay;
           curves are stacked into composite momentum (Phase 2 logic).
        3. **Hurdle scoring** — Classifier gates at the threshold; above gate uses
           the high-tier Tweedie regressor, below uses carryover + 1,500 fallback.
        """
    )
