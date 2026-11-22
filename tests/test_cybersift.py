"""Behavior checks for the saved CyberSift pipeline and demo sample."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cybersift.config import (  # noqa: E402
    BENIGN_LABEL,
    DEMO_SAMPLE_PATH,
    EXPECTED_FEATURE_COUNT,
    LABEL_COL,
    RANDOM_STATE,
)
from cybersift.predict import load_artifact, score_flows  # noqa: E402


@pytest.fixture(scope="session")
def artifact():
    return load_artifact()


@pytest.fixture(scope="session")
def demo():
    return pd.read_csv(DEMO_SAMPLE_PATH)


def _bounded(scores):
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_artifact_has_expected_feature_count(artifact):
    # Proves the committed artifact matches the project's 69-feature contract.
    assert len(artifact["features"]) == EXPECTED_FEATURE_COUNT


def test_benign_flows_are_mostly_low_risk(artifact, demo):
    # Proves representative benign demo flows usually stay below low risk.
    rows = demo.loc[demo[LABEL_COL].eq(BENIGN_LABEL)].sample(50, random_state=RANDOM_STATE)
    scored, _ = score_flows(rows, artifact)
    low_share = float(scored["risk_score"].lt(0.3).mean())
    print(f"BENIGN: {low_share:.1%} low risk across {len(rows)} flows")
    assert low_share >= 0.8
    _bounded(scored["risk_score"].to_numpy())


def test_attack_detection_rates_by_label(artifact, demo):
    # Proves each sample attack type is scored and reports its observed detection rate.
    attacks = demo.loc[demo[LABEL_COL].ne(BENIGN_LABEL)]
    for label, group in attacks.groupby(LABEL_COL):
        rows = group.sample(n=20, replace=len(group) < 20, random_state=RANDOM_STATE)
        scored, _ = score_flows(rows, artifact)
        rate = float(scored["flagged"].mean())
        print(f"{label}: {rate:.1%} detected (20 rows, {rows.index.nunique()} unique)")
        assert 0.0 <= rate <= 1.0
        if label in {"DDoS", "PortScan"}:
            assert rate >= 0.9
        _bounded(scored["risk_score"].to_numpy())


def test_scores_are_probabilities(artifact, demo):
    # Proves model output always stays within the probability range.
    scored, _ = score_flows(demo.sample(50, random_state=RANDOM_STATE), artifact)
    _bounded(scored["risk_score"].to_numpy())


def test_empty_dataframe_is_rejected(artifact):
    # Proves empty input gets a clear validation error.
    with pytest.raises(ValueError, match="Input is empty"):
        score_flows(pd.DataFrame(), artifact)


def test_missing_feature_names_are_reported(artifact, demo):
    # Proves missing model columns are named in the error.
    missing = artifact["features"][0]
    with pytest.raises(ValueError, match=missing):
        score_flows(demo.iloc[[0]].drop(columns=missing), artifact)


def test_non_numeric_input_is_rejected(artifact, demo):
    # Proves all-text feature rows fail with the no-numeric-data message.
    row = demo.iloc[[0]].copy()
    for feature in artifact["features"]:
        row[feature] = "not numeric"
    with pytest.raises(ValueError, match="No numeric feature data"):
        score_flows(row, artifact)


def test_all_nan_row_is_rejected(artifact):
    # Proves an all-missing feature row cannot produce a fabricated score.
    row = pd.DataFrame([{feature: np.nan for feature in artifact["features"]}])
    with pytest.raises(ValueError, match="No numeric feature data"):
        score_flows(row, artifact)


def test_infinity_is_repaired(artifact, demo):
    # Proves infinity is reported and imputed before a finite score is returned.
    row = demo.iloc[[0]].copy()
    row.loc[:, artifact["features"][0]] = np.inf
    scored, warnings = score_flows(row, artifact)
    assert warnings["rows_with_inf"] == 1
    _bounded(scored["risk_score"].to_numpy())


def test_extra_unknown_columns_are_ignored(artifact, demo):
    # Proves unrelated columns do not alter the model probability.
    row = demo.iloc[[0]].copy()
    expected, _ = score_flows(row, artifact)
    row["examiner note"] = "ignore this"
    actual, _ = score_flows(row, artifact)
    np.testing.assert_allclose(actual["risk_score"], expected["risk_score"])


def test_shuffled_columns_keep_identical_scores(artifact, demo):
    # Proves feature order is normalized to the saved artifact's order.
    row = demo.iloc[[0]]
    expected, _ = score_flows(row, artifact)
    actual, _ = score_flows(row.loc[:, list(reversed(row.columns))], artifact)
    np.testing.assert_allclose(actual["risk_score"], expected["risk_score"])


def test_leading_spaces_in_column_names_are_cleaned(artifact, demo):
    # Proves leading spaces do not make otherwise valid feature names fail.
    row = demo.iloc[[0]].copy()
    expected, _ = score_flows(row, artifact)
    spaced = row.rename(columns={name: f" {name}" for name in row.columns})
    actual, _ = score_flows(spaced, artifact)
    np.testing.assert_allclose(actual["risk_score"], expected["risk_score"])


def test_all_zero_features_produce_a_bounded_score(artifact):
    # Proves the all-zero boundary row is handled by the saved pipeline.
    row = pd.DataFrame([{feature: 0.0 for feature in artifact["features"]}])
    scored, _ = score_flows(row, artifact)
    _bounded(scored["risk_score"].to_numpy())


def test_negative_duration_produces_a_bounded_score(artifact, demo):
    # Proves a negative duration is accepted without an invalid probability.
    row = demo.iloc[[0]].copy()
    row["Flow Duration"] = -1.0
    scored, _ = score_flows(row, artifact)
    _bounded(scored["risk_score"].to_numpy())


def test_huge_values_produce_a_bounded_score(artifact):
    # Proves very large finite feature values do not break inference.
    row = pd.DataFrame([{feature: 1e12 for feature in artifact["features"]}])
    scored, _ = score_flows(row, artifact)
    _bounded(scored["risk_score"].to_numpy())


def test_threshold_zero_flags_all_and_one_flags_none(artifact):
    # Proves threshold endpoints behave correctly on a known sub-one score.
    row = pd.DataFrame([{feature: 0.0 for feature in artifact["features"]}])
    at_zero, _ = score_flows(row, artifact, threshold=0.0)
    at_one, _ = score_flows(row, artifact, threshold=1.0)
    assert at_zero["flagged"].all()
    assert not at_one["flagged"].any()


def test_repeated_input_is_deterministic(artifact, demo):
    # Proves repeated inference returns the same saved-model output.
    row = demo.iloc[:5]
    first, _ = score_flows(row, artifact)
    second, _ = score_flows(row, artifact)
    np.testing.assert_allclose(first["risk_score"], second["risk_score"], rtol=0, atol=1e-12)


def test_single_row_matches_batch_score(artifact, demo):
    # Proves scoring one flow alone matches its score inside a batch.
    batch = demo.iloc[:3]
    single, _ = score_flows(batch.iloc[[0]], artifact)
    together, _ = score_flows(batch, artifact)
    np.testing.assert_allclose(single["risk_score"].iloc[0], together["risk_score"].iloc[0])
