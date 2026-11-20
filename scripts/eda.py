"""Create report figures and summary tables from the cleaned dataset."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cybersift.config import (  # noqa: E402
    CLEAN_DATA_PATH,
    DAY_ORDER,
    EDA_ATTACK_LABELS,
    EDA_BOXPLOT_FEATURES,
    EDA_BOXPLOT_SAMPLE_PER_LABEL,
    EDA_CORRELATION_SAMPLE_ROWS,
    EDA_MEDIAN_FEATURES,
    FIGURES_DIR,
    FIGURE_DPI,
    LABEL_COL,
    RANDOM_STATE,
    TABLES_DIR,
)


CLASS_COLORS = {
    "Benign": "#4C78A8",
    "DDoS": "#F58518",
    "PortScan": "#E45756",
    "DoS Hulk": "#72B7B2",
    "FTP-Patator": "#54A24B",
    "Bot": "#B279A2",
}


def set_report_style():
    """Use one clean, readable style across all report figures."""
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
    sns.set_palette(list(CLASS_COLORS.values()))


def save_figure(fig, path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def validate_columns(df, required):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"The cleaned dataset is missing required columns: {missing}")


def plot_class_balance(df):
    counts = df["is_attack"].value_counts().reindex([0, 1], fill_value=0)
    summary = pd.DataFrame(
        {
            "class": ["Benign", "Attack"],
            "count": [int(counts.loc[0]), int(counts.loc[1])],
        }
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    summary.to_csv(TABLES_DIR / "eda_class_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        summary["class"],
        summary["count"],
        color=[CLASS_COLORS["Benign"], "#E45756"],
        width=0.62,
    )
    ax.set_title("Class balance")
    ax.set_ylabel("Flows")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count, share in zip(bars, summary["count"], summary["share"]):
        ax.annotate(
            f"{count:,}\n{share:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, FIGURES_DIR / "eda_class_balance.png")


def plot_attack_type_counts(df):
    counts = (
        df.loc[df[LABEL_COL] != "BENIGN", LABEL_COL]
        .value_counts()
        .rename_axis(LABEL_COL)
        .reset_index(name="count")
    )
    counts.to_csv(TABLES_DIR / "eda_attack_type_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(counts[LABEL_COL], counts["count"], color="#F58518", width=0.72)
    ax.set_yscale("log")
    ax.set_title("Attack flows by type")
    ax.set_ylabel("Flows (log scale)")
    ax.set_xlabel("Attack type")
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, FIGURES_DIR / "eda_attack_type_counts.png")


def plot_attack_share_by_day(df):
    day_summary = (
        df.groupby("day", observed=True)
        .agg(flow_count=("is_attack", "size"), attack_count=("is_attack", "sum"))
        .reindex(DAY_ORDER)
        .reset_index()
        .rename(columns={"index": "day"})
    )
    day_summary["attack_share"] = day_summary["attack_count"] / day_summary["flow_count"]
    day_summary.to_csv(TABLES_DIR / "eda_attack_share_by_day.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        day_summary["day"],
        day_summary["attack_share"] * 100,
        color="#E45756",
        width=0.65,
    )
    ax.set_title("Attack share by day")
    ax.set_ylabel("Attack flows (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, share in zip(bars, day_summary["attack_share"]):
        ax.annotate(
            f"{share:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    save_figure(fig, FIGURES_DIR / "eda_attack_share_by_day.png")


def plot_feature_boxplots(df):
    labels = ["BENIGN", *EDA_ATTACK_LABELS]
    display_labels = ["Benign", *EDA_ATTACK_LABELS]
    samples = []
    for label in labels:
        group = df.loc[df[LABEL_COL] == label]
        sample_size = min(EDA_BOXPLOT_SAMPLE_PER_LABEL, len(group))
        samples.append(group.sample(n=sample_size, random_state=RANDOM_STATE))

    sample_counts = []
    for feature in EDA_BOXPLOT_FEATURES:
        box_data = []
        tick_labels = []
        for label, display_label, sample in zip(labels, display_labels, samples):
            values = pd.to_numeric(sample[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            box_data.append(values)
            tick_labels.append(f"{display_label}\nn={len(values):,}")
            sample_counts.append(
                {
                    "feature": feature,
                    "Label": label,
                    "sampled_rows": len(sample),
                    "positive_rows_plotted": len(values),
                }
            )

        fig, ax = plt.subplots(figsize=(9, 5))
        boxes = ax.boxplot(
            box_data,
            tick_labels=tick_labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#263238", "linewidth": 1.5},
            whiskerprops={"color": "#667085"},
            capprops={"color": "#667085"},
        )
        for box, display_label in zip(boxes["boxes"], display_labels):
            box.set_facecolor(CLASS_COLORS[display_label])
            box.set_alpha(0.82)
            box.set_edgecolor("#344054")
        ax.set_yscale("log")
        ax.set_title(f"{feature} by class")
        ax.set_ylabel(f"{feature} (log scale)")
        ax.grid(axis="y", which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.5,
            0.01,
            "Up to 20,000 flows sampled per label; zero and negative values omitted on log scale.",
            ha="center",
            color="#667085",
            fontsize=8,
        )
        filename = feature.lower().replace("/", "_").replace(" ", "_")
        save_figure(fig, FIGURES_DIR / f"eda_boxplot_{filename}.png")

    pd.DataFrame(sample_counts).to_csv(
        TABLES_DIR / "eda_boxplot_sample_counts.csv", index=False
    )


def save_medians_by_label(df):
    medians = df.groupby(LABEL_COL, sort=True)[EDA_MEDIAN_FEATURES].median().reset_index()
    medians.to_csv(TABLES_DIR / "eda_medians_by_label.csv", index=False)


def plot_correlation_heatmap(df):
    feature_columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column != "is_attack"
    ]
    sample = df[feature_columns].sample(
        n=min(EDA_CORRELATION_SAMPLE_ROWS, len(df)),
        random_state=RANDOM_STATE,
    )
    variances = sample.var().dropna().sort_values(ascending=False)
    top_features = variances.head(20).index.tolist()
    correlation = sample[top_features].corr()
    correlation.rename_axis("Feature").reset_index().to_csv(
        TABLES_DIR / "eda_correlation_top20.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation of the 20 highest-variance flow features")
    ax.tick_params(axis="x", labelrotation=55, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    save_figure(fig, FIGURES_DIR / "eda_correlation_heatmap.png")


def main():
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEAN_DATA_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    set_report_style()

    df = pd.read_parquet(CLEAN_DATA_PATH)
    required_columns = [
        LABEL_COL,
        "day",
        "is_attack",
        *EDA_BOXPLOT_FEATURES,
        *EDA_MEDIAN_FEATURES,
    ]
    validate_columns(df, required_columns)

    plot_class_balance(df)
    plot_attack_type_counts(df)
    plot_attack_share_by_day(df)
    plot_feature_boxplots(df)
    save_medians_by_label(df)
    plot_correlation_heatmap(df)

    created = sorted(FIGURES_DIR.glob("eda_*.png")) + sorted(TABLES_DIR.glob("eda_*.csv"))
    figure_names = ", ".join(path.relative_to(PROJECT_DIR).as_posix() for path in created if path.suffix == ".png")
    table_names = ", ".join(path.relative_to(PROJECT_DIR).as_posix() for path in created if path.suffix == ".csv")
    print("Created files:")
    print(f"Figures: {figure_names}")
    print(f"Tables: {table_names}")


if __name__ == "__main__":
    main()
