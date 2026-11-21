"""Evaluate saved CyberSift models on the untouched test splits."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cybersift.config import (  # noqa: E402
    ABLATION_CSV_PATH,
    ABLATION_TRAIN_ROWS,
    BENIGN_LABEL,
    BASELINES_MODEL_PATH,
    CONFUSION_MATRIX_FIGURE_PATH,
    DAYSHIFT_DETECTION_CSV_PATH,
    DAYSHIFT_DETECTION_FIGURE_PATH,
    DAYSHIFT_RF_MODEL_PATH,
    DAYSHIFT_COL,
    DESTINATION_PORT_FEATURE,
    DETECTION_BY_ATTACK_CSV_PATH,
    DETECTION_BY_ATTACK_FIGURE_PATH,
    ERROR_SAMPLE_ROWS,
    FALSE_NEGATIVES_CSV_PATH,
    FALSE_POSITIVES_CSV_PATH,
    FEATURE_IMPORTANCE_CSV_PATH,
    FEATURE_IMPORTANCE_FIGURE_PATH,
    FEATURES_PATH,
    FIGURES_DIR,
    FIGURE_DPI,
    LABEL_COL,
    METRICS_JSON_PATH,
    MODEL_COMPARISON_CSV_PATH,
    PR_CURVE_FIGURE_PATH,
    RARE_LABEL_ROWS,
    RANDOM_STATE,
    RF_MODEL_PATH,
    ROC_CURVE_FIGURE_PATH,
    SPLIT_COL,
    SPLITS_PATH,
    TABLES_DIR,
    TARGET_COL,
    TEST_SPLIT,
    TRAIN_SPLIT,
    THRESHOLD_TRADEOFF_FIGURE_PATH,
    VAL_SPLIT,
    WORKLOAD_REDUCTION_CSV_PATH,
    WORKLOAD_TOP_K,
)
from cybersift.data import prepare_features  # noqa: E402
from cybersift.rules import rule_based_predict  # noqa: E402


MODEL_COLORS = {
    "Rules": "#E45756",
    "LogisticRegression": "#4C78A8",
    "DecisionTree": "#54A24B",
    "RandomForest": "#F58518",
    "RandomForest (dayshift)": "#B279A2",
}


def set_plot_style():
    """Use one clear style for report figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#344054",
            "axes.labelcolor": "#263238",
            "text.color": "#263238",
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.size": 9,
            "grid.color": "#D9E0E6",
            "grid.alpha": 0.7,
        }
    )


def read_split(column, value, features):
    """Read one split and only the columns needed for evaluation."""
    columns = list(dict.fromkeys([*features, LABEL_COL, TARGET_COL, column]))
    rows = pd.read_parquet(
        SPLITS_PATH,
        columns=columns,
        filters=[(column, "==", value)],
    )
    return rows.drop(columns=[column])


def positive_probabilities(pipeline, x):
    """Return probabilities for the attack class."""
    classes = pipeline.named_steps["classifier"].classes_
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return pipeline.predict_proba(x)[:, positive_column]


def score_pipeline(pipeline, x):
    """Score a batch and measure model prediction speed."""
    start = time.perf_counter()
    probabilities = positive_probabilities(pipeline, x)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return probabilities, float(len(x) / elapsed)


def metric_summary(y_true, scores, threshold, scoring_speed=None):
    """Calculate binary metrics from scores and a saved or validation threshold."""
    predictions = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    negatives = int(np.sum(y_true == 0))
    return {
        "threshold": float(threshold),
        "n_flows": int(len(y_true)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(matrix[0, 1] / negatives) if negatives else 0.0,
        "scoring_flows_per_second": (
            float(scoring_speed) if scoring_speed is not None else None
        ),
        "confusion_matrix": matrix.tolist(),
    }


def recall_by_label(labels, predictions):
    """Return attack recall by original Label, excluding benign flows."""
    recalls = {}
    for label in sorted(set(labels) - {BENIGN_LABEL}):
        mask = labels == label
        recalls[label] = {
            "rows": int(mask.sum()),
            "recall": float(predictions[mask].mean()),
            "too_rare": bool(mask.sum() < RARE_LABEL_ROWS),
        }
    return recalls


def draw_confusion_matrix(matrix):
    total = matrix.sum()
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            percent = 100 * matrix[row, column] / total if total else 0.0
            annotations[row, column] = f"{matrix[row, column]:,}\n{percent:.1f}%"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax,
    )
    ax.set_title("Random-test RF confusion matrix\n(count and % of all flows)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.savefig(CONFUSION_MATRIX_FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_curves(curve_models):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, (y_true, scores) in curve_models.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS[name],
            label=f"{name} (AP={average_precision_score(y_true, scores):.3f})",
            linewidth=1.8,
        )
    ax.set_title("Precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(PR_CURVE_FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, (y_true, scores) in curve_models.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=MODEL_COLORS[name],
            label=f"{name} (AUC={roc_auc_score(y_true, scores):.3f})",
            linewidth=1.8,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#98A2B3", linewidth=1)
    ax.set_title("ROC curves")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(ROC_CURVE_FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_threshold_tradeoff(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return
    sample_indices = np.linspace(
        0, len(thresholds) - 1, min(500, len(thresholds)), dtype=int
    )
    selected_thresholds = thresholds[sample_indices]
    sorted_scores = np.sort(scores)
    flagged = len(scores) - np.searchsorted(
        sorted_scores, selected_thresholds, side="left"
    )
    flagged_percent = 100 * flagged / len(scores)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        selected_thresholds,
        precision[:-1][sample_indices] * 100,
        label="Precision",
        color="#4C78A8",
    )
    ax.plot(
        selected_thresholds,
        recall[:-1][sample_indices] * 100,
        label="Recall",
        color="#E45756",
    )
    ax.set_xlabel("RF probability threshold")
    ax.set_ylabel("Precision / recall (%)")
    ax.set_ylim(0, 102)
    ax.grid(True)
    ax.set_title("RF threshold trade-off on random TEST")

    right_ax = ax.twinx()
    right_ax.plot(
        selected_thresholds,
        flagged_percent,
        label="Flows flagged",
        color="#54A24B",
        linestyle="--",
    )
    right_ax.set_ylabel("Flows flagged (%)")
    right_ax.set_ylim(0, 102)
    lines, labels = ax.get_legend_handles_labels()
    right_lines, right_labels = right_ax.get_legend_handles_labels()
    ax.legend(lines + right_lines, labels + right_labels, loc="best")
    fig.savefig(
        THRESHOLD_TRADEOFF_FIGURE_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def draw_detection_figure(summary, path, title):
    labels = summary["Label"].tolist()
    positions = np.arange(len(labels))
    width = 0.38
    plot_labels = [
        f"{label}\n(too rare)" if rare else label
        for label, rare in zip(labels, summary["too_rare"])
    ]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.72), 5.5))
    ax.bar(
        positions - width / 2,
        summary["RF recall"] * 100,
        width,
        label="RandomForest",
        color=MODEL_COLORS["RandomForest"],
    )
    if "Rules recall" in summary.columns:
        ax.bar(
            positions + width / 2,
            summary["Rules recall"] * 100,
            width,
            label="Rules",
            color=MODEL_COLORS["Rules"],
        )
    ax.set_xticks(positions, plot_labels, rotation=35, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_feature_importance(top_features):
    ordered = top_features.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(ordered["feature"], ordered["importance"], color="#4C78A8")
    ax.set_title("Top 20 RF feature importances")
    ax.set_xlabel("Importance")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def top_deviations(row, features, benign_median, benign_mad):
    """Find the five largest robust deviations from benign training values."""
    deviations = []
    for feature in features:
        value = row[feature]
        median = benign_median.get(feature)
        mad = benign_mad.get(feature)
        if pd.isna(value) or median is None or pd.isna(median):
            continue
        difference = abs(float(value) - float(median))
        scale = float(mad) if mad is not None and pd.notna(mad) and float(mad) > 0 else 1.0
        deviations.append((feature, difference / scale))
    deviations.sort(key=lambda item: item[1], reverse=True)
    return deviations[:5]


def error_rows(indices, error_type, test_rows, x_test, scores, features, rf_bundle):
    """Create a small error table with labels and the most unusual features."""
    output = []
    benign_median = rf_bundle["benign_median"]
    benign_mad = rf_bundle["benign_mad"]
    for index in indices:
        values = x_test.iloc[index]
        record = {
            "Label": str(test_rows.iloc[index][LABEL_COL]),
            "error_type": error_type,
            "attack_probability": float(scores[index]),
        }
        for rank, (feature, deviation) in enumerate(
            top_deviations(values, features, benign_median, benign_mad), start=1
        ):
            record[f"feature_{rank}"] = feature
            record[f"feature_{rank}_deviation"] = float(deviation)
        for rank in range(1, 6):
            record.setdefault(f"feature_{rank}", "")
            record.setdefault(f"feature_{rank}_deviation", None)
        output.append(record)
    return pd.DataFrame(output)


def choose_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.argmax(f1_values))])


def make_ablation_pipeline(params):
    """Build an RF pipeline with the saved main-model settings."""
    classifier = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 20),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def run_ablation(features, rf_bundle, test_rows, y_test, x_test):
    """Compare the same RF settings with and without Destination Port."""
    train_rows = read_split(SPLIT_COL, TRAIN_SPLIT, features)
    sample_size = min(ABLATION_TRAIN_ROWS, len(train_rows))
    if sample_size < len(train_rows):
        selected, _ = train_test_split(
            np.arange(len(train_rows)),
            train_size=sample_size,
            random_state=RANDOM_STATE,
            stratify=train_rows[LABEL_COL],
        )
        train_rows = train_rows.iloc[selected].reset_index(drop=True)

    x_train = prepare_features(train_rows, features)
    y_train = train_rows[TARGET_COL].to_numpy(dtype=np.int8)
    del train_rows

    validation_rows = read_split(SPLIT_COL, VAL_SPLIT, features)
    x_validation = prepare_features(validation_rows, features)
    y_validation = validation_rows[TARGET_COL].to_numpy(dtype=np.int8)
    del validation_rows

    features_without_port = [
        feature for feature in features if feature != DESTINATION_PORT_FEATURE
    ]
    variants = [
        ("RF with Destination Port", features),
        ("RF without Destination Port", features_without_port),
    ]
    rows = []
    params = rf_bundle.get("best_params", {})
    for model_name, model_features in variants:
        pipeline = make_ablation_pipeline(params)
        pipeline.fit(x_train.loc[:, model_features], y_train)
        validation_scores = positive_probabilities(
            pipeline, x_validation.loc[:, model_features]
        )
        threshold = choose_f1_threshold(y_validation, validation_scores)
        test_scores = positive_probabilities(pipeline, x_test.loc[:, model_features])
        metrics = metric_summary(y_test, test_scores, threshold)
        rows.append(
            {
                "model": model_name,
                "train_rows": int(len(y_train)),
                "feature_count": len(model_features),
                **{key: metrics[key] for key in (
                    "threshold", "precision", "recall", "f1", "roc_auc",
                    "pr_auc", "false_positive_rate",
                )},
            }
        )
    pd.DataFrame(rows).to_csv(ABLATION_CSV_PATH, index=False)
    return rows


def main():
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    with FEATURES_PATH.open("r", encoding="utf-8") as feature_file:
        features = json.load(feature_file)
    rf_bundle = joblib.load(RF_MODEL_PATH)
    baseline_bundle = joblib.load(BASELINES_MODEL_PATH)
    dayshift_bundle = joblib.load(DAYSHIFT_RF_MODEL_PATH)
    if rf_bundle.get("features") != features:
        raise ValueError("Saved RF features do not match models/features.json.")

    test_rows = read_split(SPLIT_COL, TEST_SPLIT, features).reset_index(drop=True)
    y_test = test_rows[TARGET_COL].to_numpy(dtype=np.int8)
    test_labels = test_rows[LABEL_COL].astype("string").fillna("UNKNOWN").to_numpy(dtype=str)
    x_test = prepare_features(test_rows, features)

    rf_pipeline = rf_bundle["pipeline"]
    rf_threshold = float(rf_bundle["threshold"])
    rf_scores, rf_speed = score_pipeline(rf_pipeline, x_test)
    rf_metrics = metric_summary(y_test, rf_scores, rf_threshold, rf_speed)
    rf_predictions = (rf_scores >= rf_threshold).astype(np.int8)

    logistic_artifact = baseline_bundle["logistic_regression"]
    logistic_scores, logistic_speed = score_pipeline(
        logistic_artifact["pipeline"], x_test
    )
    logistic_threshold = float(logistic_artifact.get("threshold", 0.5))
    logistic_metrics = metric_summary(
        y_test, logistic_scores, logistic_threshold, logistic_speed
    )

    tree_artifact = baseline_bundle["decision_tree"]
    tree_scores, tree_speed = score_pipeline(tree_artifact["pipeline"], x_test)
    tree_threshold = float(tree_artifact.get("threshold", 0.5))
    tree_metrics = metric_summary(y_test, tree_scores, tree_threshold, tree_speed)

    rule_start = time.perf_counter()
    rule_predictions = rule_based_predict(test_rows).astype(np.int8)
    rule_elapsed = max(time.perf_counter() - rule_start, 1e-9)
    rule_scores = rule_predictions.astype(np.float64)
    rule_metrics = metric_summary(
        y_test,
        rule_scores,
        0.5,
        len(y_test) / rule_elapsed,
    )

    random_models = {
        "Rules": (rule_scores, 0.5, rule_metrics),
        "LogisticRegression": (
            logistic_scores,
            logistic_threshold,
            logistic_metrics,
        ),
        "DecisionTree": (tree_scores, tree_threshold, tree_metrics),
        "RandomForest": (rf_scores, rf_threshold, rf_metrics),
    }
    comparison_rows = []
    for model_name, (_, _, metrics) in random_models.items():
        comparison_rows.append(
            {"test_split": "random_test", "model": model_name, **metrics}
        )

    draw_confusion_matrix(np.asarray(rf_metrics["confusion_matrix"]))
    curve_models = {
        name: (y_test, scores) for name, (scores, _, _) in random_models.items()
    }

    dayshift_features = dayshift_bundle.get("features", features)
    friday_rows = read_split(DAYSHIFT_COL, TEST_SPLIT, dayshift_features).reset_index(
        drop=True
    )
    y_friday = friday_rows[TARGET_COL].to_numpy(dtype=np.int8)
    friday_labels = (
        friday_rows[LABEL_COL].astype("string").fillna("UNKNOWN").to_numpy(dtype=str)
    )
    x_friday = prepare_features(friday_rows, dayshift_features)
    dayshift_pipeline = dayshift_bundle["pipeline"]
    dayshift_threshold = float(dayshift_bundle.get("threshold", 0.5))
    dayshift_scores, dayshift_speed = score_pipeline(dayshift_pipeline, x_friday)
    dayshift_metrics = metric_summary(
        y_friday, dayshift_scores, dayshift_threshold, dayshift_speed
    )
    dayshift_predictions = (dayshift_scores >= dayshift_threshold).astype(np.int8)
    comparison_rows.append(
        {
            "test_split": "Friday",
            "model": "RandomForest (dayshift)",
            **dayshift_metrics,
        }
    )
    curve_models["RandomForest (dayshift)"] = (y_friday, dayshift_scores)
    draw_curves(curve_models)
    draw_threshold_tradeoff(y_test, rf_scores)

    detection_rows = []
    for label in sorted(set(test_labels) - {BENIGN_LABEL}):
        mask = test_labels == label
        detection_rows.append(
            {
                "Label": label,
                "test_rows": int(mask.sum()),
                "RF recall": float(rf_predictions[mask].mean()),
                "Rules recall": float(rule_predictions[mask].mean()),
                "too_rare": bool(mask.sum() < RARE_LABEL_ROWS),
                "status": "too rare" if mask.sum() < RARE_LABEL_ROWS else "enough rows",
            }
        )
    detection_summary = pd.DataFrame(detection_rows)
    detection_summary.to_csv(DETECTION_BY_ATTACK_CSV_PATH, index=False)
    draw_detection_figure(
        detection_summary,
        DETECTION_BY_ATTACK_FIGURE_PATH,
        "Attack-type recall on random TEST",
    )

    friday_detection_rows = []
    for label in sorted(set(friday_labels) - {BENIGN_LABEL}):
        mask = friday_labels == label
        friday_detection_rows.append(
            {
                "Label": label,
                "friday_rows": int(mask.sum()),
                "RF recall": float(dayshift_predictions[mask].mean()),
                "too_rare": bool(mask.sum() < RARE_LABEL_ROWS),
                "status": "too rare" if mask.sum() < RARE_LABEL_ROWS else "enough rows",
            }
        )
    friday_detection = pd.DataFrame(friday_detection_rows)
    friday_detection.to_csv(DAYSHIFT_DETECTION_CSV_PATH, index=False)
    draw_detection_figure(
        friday_detection,
        DAYSHIFT_DETECTION_FIGURE_PATH,
        "Detection of attack types not seen in training",
    )

    importance_source = rf_bundle["feature_importances"]
    if isinstance(importance_source, dict):
        importance_rows = [
            {"feature": name, "importance": float(value)}
            for name, value in importance_source.items()
        ]
    else:
        importance_rows = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(features, importance_source)
        ]
    top_features = pd.DataFrame(importance_rows).nlargest(20, "importance")
    top_features.to_csv(FEATURE_IMPORTANCE_CSV_PATH, index=False)
    draw_feature_importance(top_features)

    workload_flagged = rf_scores >= rf_threshold
    attack_total = int(y_test.sum())
    flagged_total = int(workload_flagged.sum())
    workload = {
        "threshold": rf_threshold,
        "flows_flagged": flagged_total,
        "percent_of_all_flows_reviewed": float(100 * flagged_total / len(y_test)),
        "attacks_caught": int(np.sum(workload_flagged & (y_test == 1))),
        "percent_of_attacks_caught": float(
            100 * np.sum(workload_flagged & (y_test == 1)) / attack_total
        ),
        "precision_among_flagged": float(y_test[workload_flagged].mean())
        if flagged_total
        else 0.0,
    }
    risk_order = np.argsort(rf_scores)[::-1]
    for top_k in WORKLOAD_TOP_K:
        selected = risk_order[: min(top_k, len(risk_order))]
        workload[f"precision_top_{top_k}"] = float(y_test[selected].mean())
    pd.DataFrame(
        [
            {"measure": name, "value": value}
            for name, value in workload.items()
        ]
    ).to_csv(WORKLOAD_REDUCTION_CSV_PATH, index=False)

    rf_test_predictions = (rf_scores >= rf_threshold).astype(np.int8)
    all_false_negative_indices = np.flatnonzero(
        (y_test == 1) & (rf_test_predictions == 0)
    )
    false_negative_counts = Counter(
        str(test_rows.iloc[index][LABEL_COL])
        for index in all_false_negative_indices
    )
    false_negative_indices = all_false_negative_indices[
        np.argsort(rf_scores[all_false_negative_indices])[:ERROR_SAMPLE_ROWS]
    ]
    false_positive_indices = np.flatnonzero((y_test == 0) & (rf_test_predictions == 1))
    false_positive_indices = false_positive_indices[
        np.argsort(rf_scores[false_positive_indices])[::-1][:ERROR_SAMPLE_ROWS]
    ]
    false_negative_table = error_rows(
        false_negative_indices,
        "false_negative",
        test_rows,
        x_test,
        rf_scores,
        features,
        rf_bundle,
    )
    false_positive_table = error_rows(
        false_positive_indices,
        "false_positive",
        test_rows,
        x_test,
        rf_scores,
        features,
        rf_bundle,
    )
    false_negative_table.to_csv(FALSE_NEGATIVES_CSV_PATH, index=False)
    false_positive_table.to_csv(FALSE_POSITIVES_CSV_PATH, index=False)
    ablation_rows = run_ablation(features, rf_bundle, test_rows, y_test, x_test)
    ablation_metrics = {row["model"]: row for row in ablation_rows}

    metrics_output = {
        "random_test": {
            "models": {
                name: metrics
                for name, (_, _, metrics) in random_models.items()
            },
            "detection_by_attack_type": detection_rows,
            "workload_reduction": workload,
            "false_negative_counts_by_attack": dict(false_negative_counts),
        },
        "dayshift_friday": {
            "model": "RandomForest (dayshift)",
            "metrics": dayshift_metrics,
            "recall_by_label": friday_detection_rows,
        },
        "ablation": ablation_metrics,
    }
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_output, metrics_file, indent=2)
        metrics_file.write("\n")
    pd.DataFrame(comparison_rows).to_csv(MODEL_COMPARISON_CSV_PATH, index=False)

    summary_rows = []
    for name, (_, _, metrics) in random_models.items():
        summary_rows.append(
            {
                "Evaluation": f"Random TEST - {name}",
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "PR-AUC": metrics["pr_auc"],
                "Notes": "",
            }
        )
    summary_rows.append(
        {
            "Evaluation": "Friday - RF dayshift",
            "Precision": dayshift_metrics["precision"],
            "Recall": dayshift_metrics["recall"],
            "F1": dayshift_metrics["f1"],
            "PR-AUC": dayshift_metrics["pr_auc"],
            "Notes": "",
        }
    )
    summary_rows.append(
        {
            "Evaluation": "Review above RF threshold",
            "Precision": workload["precision_among_flagged"],
            "Recall": workload["percent_of_attacks_caught"] / 100,
            "F1": None,
            "PR-AUC": None,
            "Notes": f"{workload['percent_of_all_flows_reviewed']:.2f}% of flows flagged",
        }
    )
    for top_k in WORKLOAD_TOP_K:
        summary_rows.append(
            {
                "Evaluation": f"Top {top_k:,} highest-risk flows",
                "Precision": workload[f"precision_top_{top_k}"],
                "Recall": None,
                "F1": None,
                "PR-AUC": None,
                "Notes": "precision at K",
            }
        )
    false_negative_summary = "; ".join(
        f"{label}={count}"
        for label, count in false_negative_counts.most_common()
    ) or "none"
    summary_rows.append(
        {
            "Evaluation": "RF false negatives by Label",
            "Precision": None,
            "Recall": None,
            "F1": None,
            "PR-AUC": None,
            "Notes": false_negative_summary,
        }
    )

    print("| Evaluation | Precision | Recall | F1 | PR-AUC | Notes |")
    print("|---|---:|---:|---:|---:|---|")
    for row in summary_rows:
        cells = [
            f"{row[key]:.4f}" if row[key] is not None else "-"
            for key in ("Precision", "Recall", "F1", "PR-AUC")
        ]
        print(
            f"| {row['Evaluation']} | "
            + " | ".join(cells)
            + f" | {row['Notes']} |"
        )


if __name__ == "__main__":
    main()
