"""Four fixed, explainable rules for flagging suspicious network flows."""

import json

import numpy as np
import pandas as pd

from cybersift.config import RULE_THRESHOLDS_PATH


def _load_thresholds():
    """Load the fixed rule values saved in the project models folder."""
    if not RULE_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Rule thresholds not found: {RULE_THRESHOLDS_PATH}")
    with RULE_THRESHOLDS_PATH.open("r", encoding="utf-8") as threshold_file:
        return json.load(threshold_file)


def _numeric_column(df, column):
    """Convert a rule feature to numbers and treat infinities as missing."""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).astype("float64")


def rule_hits(df):
    """Return one boolean column for each rule."""
    thresholds = _load_thresholds()
    use_r3_fallback = thresholds.get("R3_uses_fallback", False)
    required = [
        "Bwd Packet Length Std",
        "Bwd Packet Length Mean",
        "Destination Port",
    ]
    if use_r3_fallback:
        required.append("min_seg_size_forward")
    else:
        required.extend(["Total Length of Fwd Packets", "Bwd Packets/s"])

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed for rules: {missing}")

    r1 = _numeric_column(df, "Bwd Packet Length Std") > thresholds[
        "R1_bwd_packet_length_std_gt"
    ]
    r2 = _numeric_column(df, "Bwd Packet Length Mean") > thresholds[
        "R2_bwd_packet_length_mean_gt"
    ]

    if use_r3_fallback:
        r3 = _numeric_column(df, "min_seg_size_forward") > thresholds[
            "R3_fallback_min_seg_size_forward_gt"
        ]
    else:
        r3 = (
            _numeric_column(df, "Total Length of Fwd Packets")
            <= thresholds["R3_total_length_fwd_packets_lte"]
        ) & (
            _numeric_column(df, "Bwd Packets/s")
            > thresholds["R3_bwd_packets_s_gt"]
        )

    r4 = _numeric_column(df, "Destination Port") <= thresholds[
        "R4_destination_port_lte"
    ]
    return pd.DataFrame(
        {
            "R1": r1.fillna(False).astype(bool),
            "R2": r2.fillna(False).astype(bool),
            "R3": r3.fillna(False).astype(bool),
            "R4": r4.fillna(False).astype(bool),
        },
        index=df.index,
    )


def rule_based_predict(df):
    """Return 1 when any rule fires and 0 otherwise."""
    return rule_hits(df).any(axis=1).to_numpy(dtype=np.int8)


def describe_rules():
    """Return short descriptions of the four active rules."""
    thresholds = _load_thresholds()
    r3_description = (
        "R3 targets PortScan: flag flows with min_seg_size_forward above 38."
        if thresholds.get("R3_uses_fallback", False)
        else "R3 targets PortScan: flag tiny forward probes with a fast backward reply."
    )
    return [
        "R1 targets DoS Hulk and DDoS: flag unusually variable backward packet lengths.",
        "R2 targets DDoS: flag consistently large backward packet lengths.",
        r3_description,
        "R4 targets FTP and SSH password guessing: flag destination ports 21 and 22.",
    ]
