"""Run the saved-model pytest checks and write a compact evidence record."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
import xml.etree.ElementTree as ET

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cybersift.config import REPORTS_DIR, TEST_CASES_CSV_PATH, TEST_LOG_PATH  # noqa: E402


CASE_DETAILS = {
    "test_artifact_has_expected_feature_count": (
        "Consistency", "Load the saved RF artifact", "The artifact has 69 features"
    ),
    "test_benign_flows_are_mostly_low_risk": (
        "Representative", "50 labeled BENIGN demo flows", "At least 80% score below 0.3"
    ),
    "test_attack_detection_rates_by_label": (
        "Representative", "20 demo samples per attack label; rare labels repeat rows", "Print detection rate; DDoS and PortScan are at least 90%"
    ),
    "test_scores_are_probabilities": (
        "Representative", "50 mixed demo flows", "Every score is finite and in [0, 1]"
    ),
    "test_empty_dataframe_is_rejected": (
        "Invalid input", "Empty DataFrame", "Raise ValueError with an empty-input message"
    ),
    "test_missing_feature_names_are_reported": (
        "Invalid input", "One required feature removed", "Raise ValueError naming the missing feature"
    ),
    "test_non_numeric_input_is_rejected": (
        "Invalid input", "All feature values are text", "Raise ValueError when no numeric data remains"
    ),
    "test_all_nan_row_is_rejected": (
        "Invalid input", "One row with all features NaN", "Raise ValueError when no numeric data remains"
    ),
    "test_infinity_is_repaired": (
        "Invalid input", "One infinite feature and otherwise valid values", "Report the infinity and return a bounded score"
    ),
    "test_extra_unknown_columns_are_ignored": (
        "Feature handling", "Valid row plus an unknown text column", "Score matches the row without the extra column"
    ),
    "test_shuffled_columns_keep_identical_scores": (
        "Feature handling", "Valid row with reversed column order", "Score matches the original column order"
    ),
    "test_leading_spaces_in_column_names_are_cleaned": (
        "Feature handling", "Valid row with leading spaces in column names", "Score matches the normalized names"
    ),
    "test_all_zero_features_produce_a_bounded_score": (
        "Boundary", "One row with all feature values zero", "Inference returns a probability in [0, 1]"
    ),
    "test_negative_duration_produces_a_bounded_score": (
        "Boundary", "Valid demo row with Flow Duration = -1", "Inference returns a probability in [0, 1]"
    ),
    "test_huge_values_produce_a_bounded_score": (
        "Boundary", "One row with every feature = 1e12", "Inference returns a probability in [0, 1]"
    ),
    "test_threshold_zero_flags_all_and_one_flags_none": (
        "Boundary", "All-zero row at thresholds 0 and 1", "Zero flags all; one flags none"
    ),
    "test_repeated_input_is_deterministic": (
        "Consistency", "The same five demo flows scored twice", "Risk scores are identical"
    ),
    "test_single_row_matches_batch_score": (
        "Consistency", "The same flow scored alone and in a three-row batch", "Risk scores are identical"
    ),
}


def _pytest_prefix():
    """Use the active interpreter if it has pytest, otherwise the system one."""
    active = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode == 0:
        return [sys.executable], os.environ.copy()

    system_python = shutil.which("python")
    venv_site = PROJECT_DIR / ".venv" / "Lib" / "site-packages"
    if system_python is None or not venv_site.is_dir():
        raise RuntimeError("pytest is not installed for the active or system Python.")
    system = subprocess.run(
        [system_python, "-m", "pytest", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if system.returncode != 0:
        raise RuntimeError("pytest is not installed for the active or system Python.")

    environment = os.environ.copy()
    extra_paths = [str(venv_site), str(SRC_DIR)]
    if environment.get("PYTHONPATH"):
        extra_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return [system_python], environment


def _read_results(xml_path):
    """Convert the temporary JUnit file into simple test outcomes."""
    root = ET.parse(xml_path).getroot()
    cases = root.findall(".//testcase")
    results = {}
    for case in cases:
        name = case.attrib.get("name", "unknown")
        failed = case.find("failure")
        if failed is None:
            failed = case.find("error")
        skipped = case.find("skipped")
        if failed is not None:
            results[name] = f"FAIL: {failed.attrib.get('message', 'see test log')}"
        elif skipped is not None:
            results[name] = "SKIPPED"
        else:
            results[name] = "PASS"
    return results


def _write_evidence(results):
    """Save one evidence row for every pytest case."""
    TEST_CASES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_CASES_CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["test_name", "category", "input_description", "expected", "result"])
        for name, (category, input_description, expected) in CASE_DETAILS.items():
            writer.writerow(
                [name, category, input_description, expected, results.get(name, "NOT RUN")]
            )


def main():
    """Run pytest verbosely, save its log, and record case outcomes."""
    TEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix, environment = _pytest_prefix()
    xml_path = REPORTS_DIR / f".pytest-results-{uuid4().hex}.xml"
    command = prefix + [
        "-m",
        "pytest",
        "-v",
        "-s",
        "tests/test_cybersift.py",
        f"--junitxml={xml_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_text = f"$ {' '.join(command)}\n{completed.stdout}"
    TEST_LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")
    try:
        results = _read_results(xml_path) if xml_path.exists() else {}
    finally:
        xml_path.unlink(missing_ok=True)
    _write_evidence(results)
    print(f"Evidence log: {TEST_LOG_PATH}")
    print(f"Evidence table: {TEST_CASES_CSV_PATH}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
