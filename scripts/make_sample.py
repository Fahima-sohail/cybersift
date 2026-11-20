"""Create the deterministic train/test splits and demo sample."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cybersift.config import (  # noqa: E402
    BENIGN_LABEL,
    CLEAN_DATA_PATH,
    DEMO_SAMPLE_PATH,
    DEMO_SAMPLE_ROWS,
    LABEL_COL,
    RARE_ATTACK_MAX_TEST_ROWS,
    RANDOM_STATE,
)
from cybersift.data import get_feature_list, make_splits  # noqa: E402
from cybersift.rules import build_rule_thresholds  # noqa: E402


def make_demo_sample(test_rows, features):
    """Keep every rare attack row and stratify the rest by attack type."""
    label_counts = test_rows[LABEL_COL].value_counts()
    rare_labels = label_counts[
        (label_counts.index != BENIGN_LABEL)
        & (label_counts < RARE_ATTACK_MAX_TEST_ROWS)
    ].index
    rare_rows = test_rows.loc[test_rows[LABEL_COL].isin(rare_labels)]
    common_rows = test_rows.loc[~test_rows[LABEL_COL].isin(rare_labels)]
    target_rows = min(DEMO_SAMPLE_ROWS, len(test_rows))

    if len(rare_rows) > target_rows:
        raise ValueError("Rare attack rows exceed the requested demo sample size.")

    common_sample_size = target_rows - len(rare_rows)
    if common_sample_size == len(common_rows):
        common_sample = common_rows
    elif common_sample_size == 0:
        common_sample = common_rows.iloc[0:0]
    else:
        common_sample, _ = train_test_split(
            common_rows,
            train_size=common_sample_size,
            random_state=RANDOM_STATE,
            stratify=common_rows[LABEL_COL],
        )

    sample = pd.concat([rare_rows, common_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return sample[features + [LABEL_COL]]


def main():
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEAN_DATA_PATH}")

    clean = pd.read_parquet(CLEAN_DATA_PATH)
    features = get_feature_list(clean)
    splits = make_splits(clean)
    thresholds = build_rule_thresholds(splits)

    random_counts = splits["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    day_counts = splits["dayshift"].value_counts().reindex(["train", "test"], fill_value=0)
    test_rows = splits.loc[splits["split"] == "test"]
    sample = make_demo_sample(test_rows, features)

    DEMO_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(DEMO_SAMPLE_PATH, index=False)
    size_mb = DEMO_SAMPLE_PATH.stat().st_size / (1024 * 1024)

    print(
        "random split rows: "
        f"train={random_counts['train']:,}, val={random_counts['val']:,}, "
        f"test={random_counts['test']:,}"
    )
    print(
        "dayshift split rows: "
        f"train={day_counts['train']:,}, val=0, test={day_counts['test']:,}"
    )
    print(f"demo sample: {len(sample):,} rows, {size_mb:.2f} MB")
    print(f"saved thresholds: {json.dumps(thresholds, sort_keys=True)}")


if __name__ == "__main__":
    main()
