"""Train the CyberSift baselines and random forest."""

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cybersift.config import (  # noqa: E402
    BASELINES_MODEL_PATH,
    BENIGN_LABEL,
    CLASS_WEIGHT_BALANCED,
    DAYSHIFT_RF_MODEL_PATH,
    DECISION_TREE_MAX_DEPTH,
    DEV_TRAIN_FRAC,
    FEATURES_PATH,
    LABEL_COL,
    LOGREG_MAX_ITER,
    MODELS_DIR,
    RANDOM_STATE,
    RF_MAX_ARTIFACT_BYTES,
    RF_MAX_DEPTH,
    RF_MAX_DEPTH_GRID,
    RF_MAX_FEATURES,
    RF_MIN_MAX_DEPTH,
    RF_MIN_N_ESTIMATORS,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_LEAF_GRID,
    RF_MODEL_PATH,
    RF_N_ESTIMATORS,
    RF_SMALL_MAX_DEPTH,
    RF_SMALL_N_ESTIMATORS,
    RF_TINY_MAX_DEPTH,
    RF_TINY_N_ESTIMATORS,
    RF_TUNING_N_ESTIMATORS,
    SPLITS_PATH,
    TABLES_DIR,
    TARGET_COL,
    TUNING_CSV_PATH,
    TRAIN_CAP,
    TUNE_CAP,
)
from cybersift.data import prepare_features  # noqa: E402


def stage_start(name):
    """Print a timestamp when a stage starts."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"{timestamp} | START | {name}", flush=True)


def stage_end(name):
    """Print a timestamp when a stage finishes."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"{timestamp} | END   | {name}", flush=True)


def read_split(filter_column, split_name, features):
    """Read one assigned split and only the columns needed for modeling."""
    columns = [*features, LABEL_COL, TARGET_COL, filter_column]
    rows = pd.read_parquet(
        SPLITS_PATH,
        columns=columns,
        filters=[(filter_column, "==", split_name)],
    )
    return rows.drop(columns=[filter_column])


def take_stratified_rows(rows, sample_size, stratify_values):
    """Return a deterministic stratified subset of the requested size."""
    if sample_size >= len(rows):
        return rows.reset_index(drop=True)
    sample_indices, _ = train_test_split(
        np.arange(len(rows)),
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=stratify_values,
    )
    return rows.iloc[sample_indices].reset_index(drop=True)


def make_xy(rows, features):
    """Use the shared feature cleaner and return labels beside the features."""
    x = prepare_features(rows, features)
    y = rows[TARGET_COL].to_numpy(dtype=np.int8)
    labels = rows[LABEL_COL].astype("string").fillna("UNKNOWN").to_numpy(dtype=str)
    return x, y, labels


def make_pipeline(classifier, scale=False):
    """Start every model pipeline with the training-only median imputer."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def make_random_forest(max_depth, min_samples_leaf, n_estimators):
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=RF_MAX_FEATURES,
        class_weight=CLASS_WEIGHT_BALANCED,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return make_pipeline(classifier)


def positive_probabilities(pipeline, x):
    """Return probabilities for the attack class."""
    classes = pipeline.named_steps["classifier"].classes_
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return pipeline.predict_proba(x)[:, positive_column]


def threshold_for_best_f1(y_true, probabilities):
    """Choose the validation probability threshold with the highest F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.argmax(f1_values))])


def validation_metrics(y_true, probabilities, threshold, labels):
    """Summarize scores, attack-type detection, and classification errors."""
    predictions = (probabilities >= threshold).astype(np.int8)
    benign_mask = labels == BENIGN_LABEL
    attack_mask = ~benign_mask
    false_positive_mask = benign_mask & (predictions == 1)
    false_negative_mask = attack_mask & (predictions == 0)

    per_attack_detection = {}
    false_negative_counts = {}
    for label in sorted(set(labels) - {BENIGN_LABEL}):
        label_mask = labels == label
        per_attack_detection[label] = float(predictions[label_mask].mean())
        false_negative_counts[label] = int(np.sum(label_mask & (predictions == 0)))

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "per_attack_type_detection": per_attack_detection,
        "error_analysis": {
            "false_positives": int(false_positive_mask.sum()),
            "false_negatives": int(false_negative_mask.sum()),
            "false_positive_rate": float(false_positive_mask.sum() / benign_mask.sum()),
            "false_negative_counts_by_attack": false_negative_counts,
        },
    }


def benign_statistics(x, labels, features):
    """Get benign training medians and median absolute deviations."""
    benign_x = x.iloc[np.flatnonzero(labels == BENIGN_LABEL)]
    medians = benign_x.median()
    mads = (benign_x - medians).abs().median()
    median_values = {
        feature: float(medians[feature]) if pd.notna(medians[feature]) else None
        for feature in features
    }
    mad_values = {
        feature: float(mads[feature]) if pd.notna(mads[feature]) else None
        for feature in features
    }
    return median_values, mad_values


def save_baselines(artifacts):
    """Save the baselines as soon as each one has finished."""
    stage_start("saving baselines.joblib")
    joblib.dump(artifacts, BASELINES_MODEL_PATH, compress=3)
    stage_end("saving baselines.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train CyberSift baseline models.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use a stratified 10%% subset of each training split.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip the RF grid and use max_depth=20, min_samples_leaf=2.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(
        f"{timestamp} | INFO  | TRAIN_CAP={TRAIN_CAP} "
        f"TUNE_CAP={TUNE_CAP} dev={args.dev} skip_tuning={args.skip_tuning}",
        flush=True,
    )

    stage_start("load data")
    with FEATURES_PATH.open("r", encoding="utf-8") as feature_file:
        features = json.load(feature_file)
    if len(features) != 69:
        raise ValueError(f"Expected 69 saved features, found {len(features)}.")

    train_rows = read_split("split", "train", features)
    validation_rows = read_split("split", "val", features)
    if args.dev:
        dev_size = max(1, int(len(train_rows) * DEV_TRAIN_FRAC))
        train_rows = take_stratified_rows(
            train_rows, dev_size, train_rows[LABEL_COL]
        )

    train_sample_rows = take_stratified_rows(
        train_rows, min(TRAIN_CAP, len(train_rows)), train_rows[LABEL_COL]
    )
    tuning_sample_rows = None
    if not args.skip_tuning:
        tuning_sample_rows = take_stratified_rows(
            train_rows, min(TUNE_CAP, len(train_rows)), train_rows[LABEL_COL]
        )

    x_train, y_train, train_labels = make_xy(train_sample_rows, features)
    del train_sample_rows
    if tuning_sample_rows is not None:
        x_tuning, y_tuning, _ = make_xy(tuning_sample_rows, features)
        del tuning_sample_rows
    del train_rows
    x_validation, y_validation, validation_labels = make_xy(validation_rows, features)
    del validation_rows
    stage_end("load data")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_artifacts = {"features": features}

    stage_start("LogisticRegression baseline")
    logistic = make_pipeline(
        LogisticRegression(
            max_iter=LOGREG_MAX_ITER,
            class_weight=CLASS_WEIGHT_BALANCED,
        ),
        scale=True,
    )
    logistic.fit(x_train, y_train)
    logistic_probabilities = positive_probabilities(logistic, x_validation)
    logistic_threshold = threshold_for_best_f1(y_validation, logistic_probabilities)
    logistic_metrics = validation_metrics(
        y_validation,
        logistic_probabilities,
        logistic_threshold,
        validation_labels,
    )
    baseline_artifacts["logistic_regression"] = {
        "pipeline": logistic,
        "threshold": logistic_threshold,
        "validation_metrics": logistic_metrics,
    }
    stage_end("LogisticRegression baseline")
    save_baselines(baseline_artifacts)

    stage_start("DecisionTree baseline")
    decision_tree = make_pipeline(
        DecisionTreeClassifier(
            max_depth=DECISION_TREE_MAX_DEPTH,
            class_weight=CLASS_WEIGHT_BALANCED,
        )
    )
    decision_tree.fit(x_train, y_train)
    tree_probabilities = positive_probabilities(decision_tree, x_validation)
    tree_threshold = threshold_for_best_f1(y_validation, tree_probabilities)
    tree_metrics = validation_metrics(
        y_validation,
        tree_probabilities,
        tree_threshold,
        validation_labels,
    )
    baseline_artifacts["decision_tree"] = {
        "pipeline": decision_tree,
        "threshold": tree_threshold,
        "validation_metrics": tree_metrics,
    }
    stage_end("DecisionTree baseline")
    save_baselines(baseline_artifacts)

    if args.skip_tuning:
        stage_start("RF tuning skipped")
        best_depth = RF_MAX_DEPTH
        best_leaf = RF_MIN_SAMPLES_LEAF
        stage_end("RF tuning skipped")
    else:
        stage_start("RF tuning")
        tuning_results = []
        for max_depth in RF_MAX_DEPTH_GRID:
            for min_samples_leaf in RF_MIN_SAMPLES_LEAF_GRID:
                config_name = (
                    f"RF tuning max_depth={max_depth} "
                    f"min_samples_leaf={min_samples_leaf}"
                )
                stage_start(config_name)
                pipeline = make_random_forest(
                    max_depth,
                    min_samples_leaf,
                    RF_TUNING_N_ESTIMATORS,
                )
                pipeline.fit(x_tuning, y_tuning)
                predictions = pipeline.predict(x_validation)
                tuning_results.append(
                    {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "n_estimators": RF_TUNING_N_ESTIMATORS,
                        "tune_rows": int(len(y_tuning)),
                        "validation_f1": float(
                            f1_score(y_validation, predictions, zero_division=0)
                        ),
                    }
                )

                stage_start("saving tuning.csv")
                pd.DataFrame(tuning_results).to_csv(TUNING_CSV_PATH, index=False)
                stage_end("saving tuning.csv")
                stage_end(config_name)
        stage_end("RF tuning")

        best_result = max(tuning_results, key=lambda row: row["validation_f1"])
        best_depth = int(best_result["max_depth"])
        best_leaf = int(best_result["min_samples_leaf"])
        del x_tuning, y_tuning, pipeline, predictions

    final_params = {
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": best_depth,
        "min_samples_leaf": best_leaf,
        "max_features": RF_MAX_FEATURES,
        "class_weight": CLASS_WEIGHT_BALANCED,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    train_benign_median, train_benign_mad = benign_statistics(
        x_train, train_labels, features
    )

    def fit_main_forest(params, stage_name="final RF fit"):
        stage_start(stage_name)
        model = make_random_forest(
            params["max_depth"],
            params["min_samples_leaf"],
            params["n_estimators"],
        )
        model.fit(x_train, y_train)
        probabilities = positive_probabilities(model, x_validation)
        threshold = threshold_for_best_f1(y_validation, probabilities)
        metrics = validation_metrics(
            y_validation, probabilities, threshold, validation_labels
        )
        stage_end(stage_name)
        return model, probabilities, threshold, metrics

    def save_main_forest(model, params, threshold, metrics):
        importances = model.named_steps["classifier"].feature_importances_
        artifact = {
            "pipeline": model,
            "features": features,
            "threshold": float(threshold),
            "benign_median": train_benign_median,
            "benign_mad": train_benign_mad,
            "feature_importances": {
                feature: float(value) for feature, value in zip(features, importances)
            },
            "n_train_rows": int(len(y_train)),
            "validation_metrics": metrics,
            "best_params": params,
        }
        stage_start("saving cybersift_rf.joblib")
        joblib.dump(artifact, RF_MODEL_PATH, compress=3)
        size_bytes = RF_MODEL_PATH.stat().st_size
        stage_end("saving cybersift_rf.joblib")
        return size_bytes

    random_forest, rf_probabilities, rf_threshold, rf_metrics = fit_main_forest(
        final_params
    )
    rf_size = save_main_forest(
        random_forest, final_params, rf_threshold, rf_metrics
    )
    fallback_used = False
    if rf_size > RF_MAX_ARTIFACT_BYTES:
        fallback_used = True
        fallback_options = [
            (
                RF_SMALL_N_ESTIMATORS,
                min(best_depth or RF_SMALL_MAX_DEPTH, RF_SMALL_MAX_DEPTH),
            ),
            (RF_TINY_N_ESTIMATORS, RF_TINY_MAX_DEPTH),
            (RF_MIN_N_ESTIMATORS, RF_MIN_MAX_DEPTH),
        ]
        for n_estimators, max_depth in fallback_options:
            final_params = {
                **final_params,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            }
            random_forest, rf_probabilities, rf_threshold, rf_metrics = fit_main_forest(
                final_params,
                stage_name="RF size-reduction fit",
            )
            rf_size = save_main_forest(
                random_forest, final_params, rf_threshold, rf_metrics
            )
            if rf_size <= RF_MAX_ARTIFACT_BYTES:
                break

    del fit_main_forest, save_main_forest, random_forest, rf_probabilities
    del x_train, y_train, train_labels, x_validation, y_validation, validation_labels

    stage_start("load dayshift train data")
    dayshift_rows = read_split("dayshift", "train", features)
    if args.dev:
        dayshift_rows = take_stratified_rows(
            dayshift_rows,
            max(1, int(len(dayshift_rows) * DEV_TRAIN_FRAC)),
            dayshift_rows[LABEL_COL],
        )
    dayshift_rows = take_stratified_rows(
        dayshift_rows,
        min(TRAIN_CAP, len(dayshift_rows)),
        dayshift_rows[LABEL_COL],
    )
    x_dayshift, y_dayshift, _ = make_xy(dayshift_rows, features)
    del dayshift_rows
    stage_end("load dayshift train data")

    stage_start("dayshift RF fit")
    dayshift_forest = make_random_forest(
        final_params["max_depth"],
        final_params["min_samples_leaf"],
        final_params["n_estimators"],
    )
    dayshift_forest.fit(x_dayshift, y_dayshift)
    stage_end("dayshift RF fit")

    stage_start("saving rf_dayshift.joblib")
    joblib.dump(
        {
            "pipeline": dayshift_forest,
            "features": features,
            "n_train_rows": int(len(y_dayshift)),
            "best_params": final_params,
            "evaluation_only": True,
        },
        DAYSHIFT_RF_MODEL_PATH,
        compress=3,
    )
    stage_end("saving rf_dayshift.joblib")

    stage_start("reporting validation scores")
    print("| Model | Threshold | Precision | Recall | F1 | PR-AUC |", flush=True)
    print("|---|---:|---:|---:|---:|---:|", flush=True)
    for model_name, metrics in (
        ("LogisticRegression", logistic_metrics),
        ("DecisionTree", tree_metrics),
        ("RandomForest", rf_metrics),
    ):
        print(
            f"| {model_name} | {metrics['threshold']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['pr_auc']:.4f} |",
            flush=True,
        )
    print(f"Main RF file size: {rf_size / (1024 * 1024):.2f} MB", flush=True)
    if fallback_used:
        print(
            "Main RF exceeded 50 MB; retrained with fewer trees/smaller depth: "
            f"n_estimators={final_params['n_estimators']}, "
            f"max_depth={final_params['max_depth']}",
            flush=True,
        )
    if rf_size > RF_MAX_ARTIFACT_BYTES:
        print("Main RF is still above 50 MB after retries.", flush=True)
    stage_end("reporting validation scores")


if __name__ == "__main__":
    main()
