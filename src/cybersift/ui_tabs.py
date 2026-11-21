"""Streamlit tab and sidebar functions for CyberSift."""

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from cybersift.config import (
    ALERT_THRESHOLD_MAX,
    ALERT_THRESHOLD_MIN,
    ALERT_THRESHOLD_STEP,
    APP_HISTOGRAM_COLOR,
    APP_THRESHOLD_LINE_COLOR,
    BENIGN_LABEL,
    DATA_SOURCE_OPTIONS,
    LABEL_COL,
    MAX_UPLOAD_ROWS,
    THREAT_DESK_FEATURE_COLUMNS,
)
from cybersift.predict import score_flows


def render_sidebar(artifact):
    """Show app controls and return the chosen source and threshold."""
    default_threshold = float(artifact.get("threshold", 0.5))
    default_threshold = min(max(default_threshold, ALERT_THRESHOLD_MIN), ALERT_THRESHOLD_MAX)

    with st.sidebar:
        st.subheader("Threat desk controls")
        threshold = st.slider(
            "Alert threshold",
            min_value=ALERT_THRESHOLD_MIN,
            max_value=ALERT_THRESHOLD_MAX,
            value=default_threshold,
            step=ALERT_THRESHOLD_STEP,
            key="alert_threshold",
        )
        st.caption(
            "Lower thresholds catch more attacks but can raise more false alarms; "
            "higher thresholds reduce false alarms but may miss attacks."
        )

        data_source = st.radio(
            "Data source",
            DATA_SOURCE_OPTIONS,
            key="data_source",
        )
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                key="traffic_upload",
                help="CSV files only. Files are capped at 200,000 rows for scoring.",
            )

        with st.container(border=True):
            st.markdown("**Model info**")
            st.caption("Random Forest · saved training pipeline")
            st.write(f"Features: {len(artifact.get('features', []))}")
            train_rows = artifact.get("n_train_rows")
            train_rows_text = f"{int(train_rows):,}" if train_rows is not None else "unknown"
            st.write(f"Training rows: {train_rows_text}")
            st.write(f"Default threshold: {float(artifact.get('threshold', 0.5)):.3f}")

    return threshold, data_source, uploaded_file


def _load_uploaded_csv(uploaded_file):
    """Read and normalize an uploaded CSV, enforcing the row limit."""
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Please choose a .csv file.")
    try:
        rows = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception as error:
        raise ValueError(
            f"Could not read this CSV. Check that it is a valid, comma-separated file. ({error})"
        ) from error

    rows.columns = [str(column).strip() for column in rows.columns]
    duplicate_names = rows.columns[rows.columns.duplicated()].tolist()
    if duplicate_names:
        raise ValueError(
            f"The CSV has duplicate column names after trimming spaces: {duplicate_names}"
        )

    was_capped = len(rows) > MAX_UPLOAD_ROWS
    if was_capped:
        rows = rows.iloc[:MAX_UPLOAD_ROWS].copy()
    return rows, was_capped


def _risk_histogram(scored, threshold):
    """Build a Plotly histogram from the model's scored probabilities."""
    figure = px.histogram(
        scored,
        x="risk_score",
        nbins=40,
        color_discrete_sequence=[APP_HISTOGRAM_COLOR],
        labels={"risk_score": "Attack probability", "count": "Flows"},
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=APP_THRESHOLD_LINE_COLOR,
        annotation_text=f"Alert threshold {threshold:.2f}",
        annotation_position="top",
    )
    figure.update_layout(
        title="Risk score distribution",
        xaxis_title="Attack probability",
        yaxis_title="Flows",
        bargap=0.06,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return figure


def _render_ground_truth(scored):
    """Show a confusion matrix and caught/missed counts when labels exist."""
    reveal = st.toggle("Reveal ground truth", key="reveal_ground_truth")
    if not reveal:
        return

    labels = scored[LABEL_COL].astype("string").fillna("UNKNOWN").str.strip()
    actual = labels.ne(BENIGN_LABEL).astype(np.int8)
    predicted = scored["flagged"].astype(np.int8)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    confusion_figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted benign", "Predicted attack"],
            y=["Actual benign", "Actual attack"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Greens",
            showscale=True,
            colorbar=dict(title="Flows"),
        )
    )
    confusion_figure.update_layout(
        title="Confusion matrix",
        xaxis_title="Model decision",
        yaxis_title="Ground truth",
        margin=dict(l=10, r=10, t=55, b=10),
    )

    attacks = scored.loc[labels.ne(BENIGN_LABEL), [LABEL_COL, "flagged"]].copy()
    if attacks.empty:
        attack_figure = None
    else:
        counts = (
            attacks.groupby(LABEL_COL, dropna=False)["flagged"]
            .agg(caught="sum", total="size")
            .sort_values("total", ascending=False)
        )
        counts["missed"] = counts["total"] - counts["caught"]
        attack_figure = go.Figure(
            data=[
                go.Bar(x=counts.index.astype(str), y=counts["caught"], name="Caught"),
                go.Bar(x=counts.index.astype(str), y=counts["missed"], name="Missed"),
            ]
        )
        attack_figure.update_layout(
            title="Caught and missed attacks by type",
            xaxis_title="Attack type",
            yaxis_title="Flows",
            barmode="stack",
            margin=dict(l=10, r=10, t=55, b=10),
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            confusion_figure,
            width="stretch",
            alt="Confusion matrix comparing predicted and actual benign or attack flows",
        )
    with right:
        if attack_figure is None:
            st.info("No attack labels are present in these rows.")
        else:
            st.plotly_chart(
                attack_figure,
                width="stretch",
                alt="Caught and missed attack flows by original label",
            )


def render_threat_desk(artifact, threshold, data_source, uploaded_file, demo_loader):
    """Load traffic, score it, and render the operational Threat Desk."""
    st.subheader("Threat Desk")
    st.caption("Rank traffic by attack probability and focus review on the highest-risk flows.")

    try:
        if data_source == "Demo traffic":
            input_rows = demo_loader().copy()
            source_key = "demo"
            st.caption(f"Source: bundled demo traffic · {len(input_rows):,} rows")
        elif uploaded_file is None:
            st.info("Upload a CSV from the sidebar to score your traffic.")
            return
        else:
            input_rows, was_capped = _load_uploaded_csv(uploaded_file)
            source_key = f"upload:{uploaded_file.name}:{uploaded_file.size}"
            if was_capped:
                st.warning(
                    f"The upload has more than {MAX_UPLOAD_ROWS:,} rows. "
                    f"Only the first {MAX_UPLOAD_ROWS:,} rows will be scored."
                )
            st.caption(f"Source: {uploaded_file.name} · {len(input_rows):,} rows")

        if st.session_state.get("selected_flow_source") != source_key:
            st.session_state["selected_flow"] = None
            st.session_state["selected_flow_source"] = source_key

        scored, warnings = score_flows(input_rows, artifact, threshold=threshold)
    except Exception as error:
        st.error(str(error))
        return

    total_flows = len(scored)
    flagged = scored.loc[scored["flagged"]].sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    flagged_count = len(flagged)
    review_share = 100.0 * flagged_count / total_flows
    workload_reduced = 100.0 - review_share
    highest_risk = float(scored["risk_score"].max())

    kpi_columns = st.columns(5, gap="small")
    metrics = [
        ("Flows scanned", f"{total_flows:,}"),
        ("Flagged", f"{flagged_count:,}"),
        ("Traffic needing review", f"{review_share:.1f}%"),
        ("Highest risk", f"{highest_risk:.1%}"),
        ("Analyst workload reduced", f"{workload_reduced:.1f}%"),
    ]
    for column, (label, value) in zip(kpi_columns, metrics):
        with column:
            st.metric(label, value, border=True)
    st.caption("Workload reduction estimates review effort versus manually checking every flow.")

    st.plotly_chart(
        _risk_histogram(scored, threshold),
        width="stretch",
        alt="Distribution of model attack probabilities with the alert threshold marked",
    )

    st.subheader("Flagged flows, ranked by risk")
    view_columns = ["risk_score", "risk_level"]
    if LABEL_COL in flagged.columns:
        view_columns.append(LABEL_COL)
    view_columns.extend(
        feature
        for feature in THREAT_DESK_FEATURE_COLUMNS
        if feature in flagged.columns and feature not in view_columns
    )
    ranked_view = flagged.loc[:, view_columns]
    selection = st.dataframe(
        ranked_view,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            )
        },
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="flagged_flows_table",
        alt="Flagged flows sorted from highest to lowest attack probability",
    )
    selected_rows = selection.selection.rows
    if selected_rows:
        st.session_state["selected_flow"] = flagged.iloc[int(selected_rows[0])].to_dict()

    selected_flow = st.session_state.get("selected_flow")
    if selected_flow is not None:
        with st.expander("Selected flow details"):
            st.dataframe(
                pd.DataFrame([selected_flow]),
                hide_index=True,
                width="stretch",
                alt="All values for the selected flagged flow",
            )

    with st.expander("View all columns for flagged flows"):
        st.dataframe(
            flagged,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk score", format="%.3f", min_value=0.0, max_value=1.0
                )
            },
            hide_index=True,
            width="stretch",
            alt="All columns for flagged flows ranked by risk",
        )

    st.download_button(
        "Download flagged flows as CSV",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="cybersift_flagged_flows.csv",
        mime="text/csv",
        width="stretch",
    )

    if LABEL_COL in scored.columns:
        _render_ground_truth(scored)

    with st.expander("Data-quality warnings"):
        st.write(f"Rows with NaN values: {warnings['rows_with_nan']:,}")
        st.write(f"Rows with infinite values: {warnings['rows_with_inf']:,}")
        st.write(
            "Rows with NaN, infinity, or non-numeric values repaired: "
            f"{warnings['rows_with_nan_or_inf_repaired']:,}"
        )
        st.write(f"Cells repaired: {warnings['cells_repaired']:,}")
        negative_counts = warnings["negative_value_counts"]
        if negative_counts:
            st.write("Negative-value counts by feature:")
            st.json(negative_counts)
        else:
            st.write("Negative values: none detected in model features.")


def render_coming_soon():
    """Show the placeholder for tabs scheduled for a later project phase."""
    st.info("coming next")
