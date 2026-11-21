"""Saved-model inference helpers. This module does not import Streamlit."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cybersift.config import (  # noqa: E402
    DEMO_SAMPLE_PATH,
    RF_MODEL_PATH,
    RISK_HIGH_MIN,
    RISK_LOW_MAX,
)
from cybersift.data import prepare_features  # noqa: E402


def load_artifact():
    """Load the saved main random-forest artifact."""
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {RF_MODEL_PATH}. Train the model first."
        )
    return joblib.load(RF_MODEL_PATH)


def score_flows(df, artifact, threshold=None):
    """Score flows and return the rows plus data-quality warning counts."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input is empty. Provide at least one flow row to score.")

    features = artifact.get("features")
    if not features:
        raise ValueError("The saved model does not contain a feature list.")

    cleaned_input = df.copy()
    cleaned_input.columns = [str(column).strip() for column in cleaned_input.columns]
    missing = [feature for feature in features if feature not in cleaned_input.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    raw_features = cleaned_input.loc[:, features]
    numeric_features = raw_features.apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric_features.astype("float64").to_numpy()
    raw_nan_mask = raw_features.isna().to_numpy()
    nan_mask = np.isnan(numeric_values)
    inf_mask = np.isinf(numeric_values)
    non_numeric_mask = (~raw_nan_mask) & nan_mask
    finite_mask = np.isfinite(numeric_values)

    if not finite_mask.any():
        raise ValueError(
            "No numeric feature data was found. Include at least one numeric flow value."
        )

    prepared = prepare_features(df, features)
    row_has_nan = raw_nan_mask.any(axis=1)
    row_has_inf = inf_mask.any(axis=1)
    row_has_non_numeric = non_numeric_mask.any(axis=1)
    row_repaired = row_has_nan | row_has_inf | row_has_non_numeric
    negative_counts = np.sum(finite_mask & (numeric_values < 0), axis=0)
    warnings = {
        "rows_with_nan": int(row_has_nan.sum()),
        "rows_with_inf": int(row_has_inf.sum()),
        "rows_with_non_numeric_values": int(row_has_non_numeric.sum()),
        "rows_with_nan_or_inf_repaired": int(row_repaired.sum()),
        "cells_repaired": int(
            np.sum(raw_nan_mask | inf_mask | non_numeric_mask)
        ),
        "negative_value_counts": {
            feature: int(count)
            for feature, count in zip(features, negative_counts)
            if count > 0
        },
    }

    pipeline = artifact.get("pipeline")
    if pipeline is None:
        raise ValueError("The saved model artifact does not contain a pipeline.")
    classes = pipeline.named_steps["classifier"].classes_
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) == 0:
        raise ValueError("The saved model has no attack class with label 1.")
    probabilities = pipeline.predict_proba(prepared)[:, int(positive_columns[0])]

    if threshold is None:
        threshold = artifact.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("The alert threshold must be a number between 0 and 1.") from error
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("The alert threshold must be between 0 and 1.")

    scored = df.copy()
    scored["risk_score"] = probabilities.astype(float)
    scored["flagged"] = probabilities >= threshold
    scored["risk_level"] = np.where(
        probabilities < RISK_LOW_MAX,
        "Low",
        np.where(probabilities < RISK_HIGH_MIN, "Medium", "High"),
    )
    return scored, warnings


if __name__ == "__main__":
    if not DEMO_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Demo sample not found at {DEMO_SAMPLE_PATH}")

    demo_rows = pd.read_csv(DEMO_SAMPLE_PATH, nrows=5)
    demo_artifact = load_artifact()
    demo_scores, _ = score_flows(demo_rows, demo_artifact)
    for row_number, risk_score in enumerate(demo_scores["risk_score"], start=1):
        print(f"Demo row {row_number}: risk_score={risk_score:.4f}")
