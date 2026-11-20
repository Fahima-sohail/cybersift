"""Loading, cleaning, and preparing CIC-IDS2017 flow data."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cybersift.config import (
    DAY_ORDER,
    DROP_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    FEATURES_PATH,
    LABEL_COL,
    MODELS_DIR,
    RANDOM_STATE,
    SPLITS_PATH,
)


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


def get_feature_list(df):
    """Save and return the cleaned dataset's model feature names."""
    features = [
        column for column in df.columns if column not in {LABEL_COL, "is_attack", "day"}
    ]
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(features)}."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with FEATURES_PATH.open("w", encoding="utf-8") as feature_file:
        json.dump(features, feature_file, indent=2)
        feature_file.write("\n")
    return features


def make_splits(df):
    """Add random and weekday split assignments, then save the full rows."""
    required = [LABEL_COL, "day"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required split columns: {missing}")
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    data = df.reset_index(drop=True).copy()
    row_numbers = np.arange(len(data))

    train_val_rows, test_rows = train_test_split(
        row_numbers,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data[LABEL_COL],
    )
    train_rows, val_rows = train_test_split(
        train_val_rows,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=data.iloc[train_val_rows][LABEL_COL],
    )

    random_split = np.empty(len(data), dtype=object)
    random_split[train_rows] = "train"
    random_split[val_rows] = "val"
    random_split[test_rows] = "test"

    unknown_days = sorted(set(data["day"].dropna().unique()) - set(DAY_ORDER))
    if unknown_days or data["day"].isna().any():
        raise ValueError(f"Unknown or missing day values: {unknown_days}")
    dayshift_split = np.where(data["day"].to_numpy() == "Friday", "test", "train")

    data["split"] = random_split
    data["dayshift"] = dayshift_split

    # Each row must be assigned to exactly one group in each split protocol.
    for assignment, allowed_groups in (
        (random_split, {"train", "val", "test"}),
        (dayshift_split, {"train", "test"}),
    ):
        if pd.isna(assignment).any() or not set(assignment).issubset(allowed_groups):
            raise AssertionError("Some rows have missing or invalid split assignments.")
        memberships = np.zeros(len(data), dtype=np.int8)
        for group in allowed_groups:
            memberships += assignment == group
        if not np.all(memberships == 1):
            raise AssertionError("A row is assigned to more than one split group.")

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(SPLITS_PATH, index=False)
    return data
