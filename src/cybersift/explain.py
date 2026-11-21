"""Small, human-readable explanations for scored flow rows."""

import numpy as np
import pandas as pd

from cybersift.config import BENIGN_LABEL, LABEL_COL, PRESET_LABELS


FRIENDLY_NAMES = {
    "Flow Packets/s": "packet rate",
    "Flow Duration": "flow duration",
    "Total Fwd Packets": "forward packet count",
    "Total Backward Packets": "backward packet count",
    "Total Length of Fwd Packets": "forward data volume",
    "Total Length of Bwd Packets": "backward data volume",
    "Flow Bytes/s": "data transfer rate",
    "Flow IAT Mean": "average gap between packets",
    "Bwd Packet Length Std": "variation in backward packet sizes",
    "Bwd Packet Length Mean": "average backward packet size",
    "Fwd Packet Length Mean": "average forward packet size",
    "Packet Length Std": "variation in packet sizes",
    "Packet Length Variance": "packet-size variation",
    "SYN Flag Count": "connection-start (SYN) signals",
    "RST Flag Count": "connection-reset (RST) signals",
    "Destination Port": "destination service port",
    "Bwd Packets/s": "backward packet rate",
    "min_seg_size_forward": "minimum forward segment size",
    "Init_Win_bytes_forward": "initial forward window size",
    "Init_Win_bytes_backward": "initial backward window size",
    "Active Mean": "average active time",
    "Idle Mean": "average idle time",
}


def explain_flow(row, artifact, top_k=5):
    """Return the largest robust feature deviations. This is an approximation, not SHAP."""
    if top_k <= 0:
        return []

    row_values = {str(name).strip(): value for name, value in row.items()}
    medians = artifact.get("benign_median", {})
    mads = artifact.get("benign_mad", {})
    importances = artifact.get("feature_importances", {})
    explanations = []

    for feature in artifact.get("features", []):
        if feature not in row_values:
            continue
        value = pd.to_numeric(pd.Series([row_values[feature]]), errors="coerce").iloc[0]
        median = medians.get(feature)
        mad = mads.get(feature)
        if (
            pd.isna(value)
            or not np.isfinite(value)
            or median is None
            or pd.isna(median)
            or not np.isfinite(float(median))
        ):
            continue

        difference = float(value) - float(median)
        if mad is None or pd.isna(mad) or float(mad) <= 0:
            robust_z = 0.0 if difference == 0 else float(np.sign(difference) * 10)
        else:
            robust_z = difference / float(mad)

        contribution = float(importances.get(feature, 0.0)) * min(
            abs(robust_z), 10.0
        )
        if contribution <= 0:
            continue

        direction = "higher than normal" if robust_z > 0 else "lower than normal"
        friendly_name = FRIENDLY_NAMES.get(feature, feature.replace("_", " ").lower())
        sentence = (
            f"The value for {friendly_name} is {direction} compared with benign traffic."
        )
        explanations.append(
            {
                "feature": feature,
                "friendly_name": friendly_name,
                "value": float(value),
                "typical_benign_value": float(median),
                "robust_z": float(robust_z),
                "direction": direction,
                "contribution": float(contribution),
                "explanation": sentence,
            }
        )

    explanations.sort(key=lambda item: item["contribution"], reverse=True)
    return explanations[:top_k]


def get_preset_flow(sample_df, kind):
    """Return the real sample flow closest to the requested class median."""
    if kind not in PRESET_LABELS:
        choices = ", ".join(PRESET_LABELS)
        raise ValueError(f"Unknown preset '{kind}'. Choose one of: {choices}.")
    if sample_df.empty:
        raise ValueError("The demo sample is empty.")
    if LABEL_COL not in sample_df.columns:
        raise ValueError(f"The demo sample is missing '{LABEL_COL}'.")

    labels = sample_df[LABEL_COL].astype("string").str.strip()
    selected = labels.isin(PRESET_LABELS[kind]).to_numpy()
    if not selected.any():
        wanted = ", ".join(PRESET_LABELS[kind])
        raise ValueError(f"No sample flows found for '{kind}' (expected Label: {wanted}).")

    excluded = {LABEL_COL, "is_attack", "day", "split", "dayshift"}
    feature_columns = []
    for column in sample_df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(sample_df[column], errors="coerce")
        if values.notna().any():
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError("The demo sample has no numeric flow features.")

    matching_positions = np.flatnonzero(selected)
    values = sample_df.iloc[matching_positions][feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    values = values.replace([np.inf, -np.inf], np.nan)
    medians = values.median()
    medians = medians[medians.notna()]
    if medians.empty:
        raise ValueError(f"No numeric flow values are available for '{kind}'.")
    values = values.loc[:, medians.index]
    deviations = (values - medians).abs().median()
    spreads = values.quantile(0.75) - values.quantile(0.25)
    scales = deviations.where(deviations > 0, spreads).fillna(1.0)
    scales = scales.where(scales > 0, 1.0)
    normalized = (values.fillna(medians) - medians) / scales
    distances = np.sqrt(np.square(normalized).sum(axis=1).to_numpy())
    closest_position = int(np.argmin(distances))
    return sample_df.iloc[int(matching_positions[closest_position])].copy()
