"""A small, editable rules baseline for flagging suspicious flows."""

import json

import numpy as np
import pandas as pd

from cybersift.config import (
    BENIGN_LABEL,
    LABEL_COL,
    RANDOM_STATE,
    RULE_FLOOD_PERCENTILE,
    RULE_THRESHOLDS_PATH,
)


def _finite_numeric(series):
    """Convert a feature to numbers and treat infinities as missing."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.astype("float64")


def build_rule_thresholds(df):
    """Build thresholds from benign training rows and save them as JSON."""
    required = {"split", LABEL_COL, "Flow Packets/s", "Flow Duration"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns needed for rule thresholds: {missing}")

    benign_train = df.loc[(df["split"] == "train") & (df[LABEL_COL] == BENIGN_LABEL)]
    if benign_train.empty:
        raise ValueError("No benign training rows are available for rule thresholds.")

    packet_rate = _finite_numeric(benign_train["Flow Packets/s"]).dropna()
    duration = _finite_numeric(benign_train["Flow Duration"]).dropna()
    if packet_rate.empty or duration.empty:
        raise ValueError("Training benign rows do not contain usable threshold values.")

    thresholds = {
        "flow_packets_s_p99": float(packet_rate.quantile(RULE_FLOOD_PERCENTILE)),
        "flow_duration_median": float(duration.median()),
    }
    RULE_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RULE_THRESHOLDS_PATH.open("w", encoding="utf-8") as threshold_file:
        json.dump(thresholds, threshold_file, indent=2)
        threshold_file.write("\n")
    return thresholds


def rule_based_predict(df):
    """Return 1 when any of the three simple attack rules fires."""
    required = [
        "Flow Packets/s",
        "Total Fwd Packets",
        "SYN Flag Count",
        "RST Flag Count",
        "Flow Duration",
        "Destination Port",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed for rules: {missing}")
    if not RULE_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Rule thresholds not found at {RULE_THRESHOLDS_PATH}; build them first."
        )
    with RULE_THRESHOLDS_PATH.open("r", encoding="utf-8") as threshold_file:
        thresholds = json.load(threshold_file)

    needed_thresholds = {"flow_packets_s_p99", "flow_duration_median"}
    missing_thresholds = sorted(needed_thresholds - set(thresholds))
    if missing_thresholds:
        raise ValueError(f"Threshold file is missing values: {missing_thresholds}")

    packet_rate = _finite_numeric(df["Flow Packets/s"])
    fwd_packets = _finite_numeric(df["Total Fwd Packets"])
    syn_flags = _finite_numeric(df["SYN Flag Count"])
    rst_flags = _finite_numeric(df["RST Flag Count"])
    duration = _finite_numeric(df["Flow Duration"])
    destination_port = _finite_numeric(df["Destination Port"])

    flood_like = packet_rate > thresholds["flow_packets_s_p99"]
    scan_like = (
        (fwd_packets <= 2)
        & ((syn_flags >= 1) | (rst_flags >= 1))
        & (duration < thresholds["flow_duration_median"])
    )
    password_guessing_like = (
        destination_port.isin([21, 22])
        & (duration < thresholds["flow_duration_median"])
    )

    predictions = flood_like | scan_like | password_guessing_like
    return predictions.to_numpy(dtype=np.int8)
