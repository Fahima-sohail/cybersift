"""Create the cleaned CIC-IDS2017 dataset."""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cybersift.config import (  # noqa: E402
    CLEAN_DATA_PATH,
    DEV_SAMPLE_FRAC,
    LABEL_COL,
    RANDOM_STATE,
    RAW_DIR,
)
from cybersift.data import clean_raw, get_drop_columns, load_raw  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Clean the raw CIC-IDS2017 CSV files.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Save a stratified 10%% sample for development.",
    )
    args = parser.parse_args()

    raw = load_raw(RAW_DIR)
    rows_before = len(raw)
    numeric = raw.select_dtypes(include=[np.number])
    inf_cells = int(np.isinf(numeric.to_numpy()).sum())
    original_nan_cells = int(raw.isna().to_numpy().sum())
    columns_to_drop = get_drop_columns(raw)

    cleaned = clean_raw(raw)
    duplicates_removed = rows_before - len(cleaned)
    clean_rows = len(cleaned)
    cleaned_nan_cells = int(cleaned.isna().to_numpy().sum())

    if args.dev:
        _, cleaned = train_test_split(
            cleaned,
            test_size=DEV_SAMPLE_FRAC,
            random_state=RANDOM_STATE,
            stratify=cleaned["is_attack"],
        )
        cleaned = cleaned.sort_index().reset_index(drop=True)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(CLEAN_DATA_PATH, index=False)

    day_attack_share = cleaned.groupby("day", observed=True)["is_attack"].mean()
    shares = ", ".join(f"{day}={share:.1%}" for day, share in day_attack_share.items())
    labels = cleaned[LABEL_COL].value_counts(dropna=False)
    label_summary = ", ".join(f"{label}: {count}" for label, count in labels.items())

    print(
        f"Rows: {rows_before:,} raw -> {clean_rows:,} clean; "
        f"{len(cleaned):,} saved ({'dev 10%' if args.dev else 'full'})"
    )
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(
        f"Invalid cells: inf={inf_cells:,} converted to NaN; "
        f"NaN raw={original_nan_cells:,}, clean={cleaned_nan_cells:,}"
    )
    print(f"Constant/duplicate columns dropped ({len(columns_to_drop)}): {', '.join(columns_to_drop)}")
    print(f"Label counts: {label_summary}")
    print(f"Attack share by day: {shares}")


if __name__ == "__main__":
    main()
