"""Loading, cleaning, and preparing CIC-IDS2017 flow data."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from cybersift.config import DROP_COLUMNS, LABEL_COL


def _day_from_filename(filename):
    """Return the weekday named in a CIC-IDS2017 filename."""
    match = re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday)", filename, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not find a weekday in filename: {filename}")
    return match.group(1).capitalize()


def load_raw(raw_dir):
    """Read raw CSVs one at a time, using float32 for numeric flow columns."""
    raw_path = Path(raw_dir)
    csv_paths = sorted(raw_path.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    frames = []
    for csv_path in csv_paths:
        header = pd.read_csv(csv_path, nrows=0).columns
        dtypes = {
            column: ("string" if column.strip() == LABEL_COL else np.float32)
            for column in header
        }
        frame = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["day"] = _day_from_filename(csv_path.name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, copy=False)


def get_drop_columns(df):
    """List configured duplicate columns and columns constant in this dataset."""
    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]
    configured_and_detected = list(dict.fromkeys([*DROP_COLUMNS, *constant_columns]))
    return [column for column in configured_and_detected if column in df.columns]


def clean_raw(df):
    """Normalize labels and invalid values, then remove duplicate/constant data."""
    if df.empty:
        raise ValueError("The raw dataset is empty.")

    clean = df.copy()
    clean.columns = [str(column).strip() for column in clean.columns]

    if LABEL_COL not in clean.columns:
        raise ValueError(f"Required label column '{LABEL_COL}' is missing.")

    labels = clean[LABEL_COL].astype("string").str.strip()
    for bad_character in ("�", "ï¿½", "â€“", "â€”", "–"):
        labels = labels.str.replace(bad_character, "-", regex=False)
    clean[LABEL_COL] = labels

    numeric_columns = clean.select_dtypes(include=[np.number]).columns
    clean.loc[:, numeric_columns] = clean.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    clean = clean.drop_duplicates().reset_index(drop=True)

    clean = clean.drop(columns=get_drop_columns(clean))

    clean["is_attack"] = clean[LABEL_COL].ne("BENIGN").astype(np.int8)
    return clean


def prepare_features(df, features):
    """Select and clean model features in the requested order without imputing."""
    prepared = df.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    ordered_features = [str(feature).strip() for feature in features]
    missing = [feature for feature in ordered_features if feature not in prepared.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    result = prepared.loc[:, ordered_features].copy()
    for column in ordered_features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
