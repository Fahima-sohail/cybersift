"""Streamlit tab and sidebar functions for CyberSift."""

import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from cybersift.config import (
    ALERT_THRESHOLD_MAX,
    ALERT_THRESHOLD_MIN,
    ALERT_THRESHOLD_STEP,
    BENIGN_LABEL,
    DATA_SOURCE_OPTIONS,
    DATASET_SOURCE,
    DAYSHIFT_DETECTION_FIGURE_PATH,
    DETECTION_BY_ATTACK_FIGURE_PATH,
    EDA_ATTACK_TYPE_COUNTS_CSV_PATH,
    EDA_CLASS_BALANCE_CSV_PATH,
    EXPECTED_FEATURE_COUNT,
    FEATURE_IMPORTANCE_FIGURE_PATH,
    GLOSSARY_ENTRIES,
    LABEL_COL,
    LIVE_MONITOR_BATCH_SIZE,
    LIVE_MONITOR_PHASES,
    LIVE_MONITOR_ROWS_PER_PHASE,
    LIVE_MONITOR_SPEED_SECONDS,
    MAX_UPLOAD_ROWS,
    METRICS_JSON_PATH,
    MODEL_COMPARISON_CSV_PATH,
    PRESET_LABELS,
    PREPROCESSING_STEPS,
    RANDOM_STATE,
    REPORT_LIMITATIONS,
    CONFUSION_MATRIX_FIGURE_PATH,
    SCORED_INPUT_CACHE_ENTRIES,
    INSPECT_EXPLANATION_FEATURE_COUNT,
    INSPECT_IMPORTANT_FEATURE_COUNT,
    THREAT_ELEVATED_MAX_PCT,
    THREAT_NORMAL_MAX_PCT,
    THREAT_DESK_FEATURE_COLUMNS,
    UPLOAD_INPUT_CACHE_ENTRIES,
)
from cybersift.predict import score_flows
from cybersift.theme import (
    BG,
    BORDER,
    CARD,
    DEEP_RED,
    MUTED,
    PURPLE,
    PURPLE_GLOW,
    PURPLE_WASH,
    RISK_HISTOGRAM_BINS,
    SURFACE,
    THREAT_RED,
    TEXT,
)
from cybersift.explain import FRIENDLY_NAMES, explain_flow, get_preset_flow
from cybersift.ui_components import (
    callout,
    kpi_card,
    render_sidebar_logo,
    render_system_status,
    section_header,
    style_figure,
    threat_banner,
    verdict_badge,
)
from cybersift.threat_panels import (
    render_alert_log,
    render_priority_queue,
    render_targeted_services,
    render_threat_map,
)


@st.cache_data(max_entries=UPLOAD_INPUT_CACHE_ENTRIES)
def _load_uploaded_csv_once(source_key, _uploaded_file):
    """Keep parsed upload rows so threshold changes do not reread a large file."""
    return _load_uploaded_csv(_uploaded_file)


@st.cache_data(max_entries=SCORED_INPUT_CACHE_ENTRIES)
def _score_input_once(source_key, _input_rows, _artifact):
    """Keep model probabilities so a threshold change only updates alert flags."""
    return score_flows(_input_rows, _artifact, threshold=_artifact.get("threshold", 0.5))


def render_sidebar(artifact):
    """Show app controls and return the chosen source and threshold."""
    default_threshold = float(artifact.get("threshold", 0.5))
    default_threshold = min(max(default_threshold, ALERT_THRESHOLD_MIN), ALERT_THRESHOLD_MAX)

    with st.sidebar:
        render_sidebar_logo()
        st.markdown('<div class="cyber-group-label">SENSITIVITY</div>', unsafe_allow_html=True)
        threshold = st.slider(
            "Alert threshold",
            min_value=ALERT_THRESHOLD_MIN,
            max_value=ALERT_THRESHOLD_MAX,
            value=default_threshold,
            step=ALERT_THRESHOLD_STEP,
            key="alert_threshold",
        )
        sensitivity_left, sensitivity_right = st.columns(2)
        with sensitivity_left:
            st.caption("More alerts, fewer misses")
        with sensitivity_right:
            st.caption("Fewer false alarms")

        st.markdown('<div class="cyber-group-label">DATA SOURCE</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Traffic source",
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

        render_system_status(artifact)

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
    scores = scored["risk_score"].to_numpy(dtype=float)
    edges = np.unique(
        np.concatenate(
            [np.linspace(0.0, 1.0, RISK_HISTOGRAM_BINS + 1), [float(threshold)]]
        )
    )
    counts, edges = np.histogram(scores, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    colors = [PURPLE if right <= threshold else THREAT_RED for right in edges[1:]]

    figure = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker_color=colors,
            marker_line_color=BG,
            marker_line_width=1,
            hovertemplate="Risk score: %{x:.2f}<br>Flows: %{y:.0f}<extra></extra>",
            name="Flows",
        )
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=THREAT_RED,
        line_width=2,
        annotation_text=f"ALERT LINE · {threshold:.3f}",
        annotation_position="top",
    )
    figure.update_layout(
        title="How suspicious is each flow? (0 = clearly normal, 1 = almost certainly an attack)",
        xaxis_title="Attack probability",
        yaxis_title="Flows",
        bargap=0,
        showlegend=False,
    )
    figure.update_xaxes(range=[0, 1])
    return style_figure(figure)


def _render_ground_truth(scored):
    """Show a confusion matrix and caught/missed counts when labels exist."""
    section_header(
        "Verification (analyst view)",
        "Compare model alerts with the labels supplied in this file.",
        help_text="Label is the recorded ground truth. It is not used to produce the risk score.",
    )
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
            colorscale=[[0, BG], [0.55, PURPLE], [1, PURPLE_GLOW]],
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
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts["caught"],
                    name="Caught",
                    marker_color=THREAT_RED,
                ),
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts["missed"],
                    name="Missed",
                    marker_color=DEEP_RED,
                ),
            ]
        )
        attack_figure.update_layout(
            title="Caught and missed attacks by type",
            xaxis_title="Attack type",
            yaxis_title="Flows",
            barmode="stack",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        style_figure(attack_figure)
    style_figure(confusion_figure)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            confusion_figure,
            width="stretch",
            alt="Confusion matrix comparing predicted and actual benign or attack flows",
        )
    with right:
        if attack_figure is None:
            callout("info", "No attack labels are present in these rows.")
        else:
            st.plotly_chart(
                attack_figure,
                width="stretch",
                alt="Caught and missed attack flows by original label",
            )


def render_threat_desk(artifact, threshold, data_source, uploaded_file, demo_loader):
    """Load traffic, score it, and render the operational Threat Desk."""
    try:
        if data_source == "Demo traffic":
            input_rows = demo_loader()
            source_key = "demo"
            st.caption(f"Source: bundled demo traffic · {len(input_rows):,} rows")
        elif uploaded_file is None:
            callout("info", "Upload a CSV from the sidebar to score your traffic.")
            return
        else:
            upload_bytes = uploaded_file.getvalue()
            content_hash = hashlib.sha256(upload_bytes).hexdigest()
            source_key = f"upload:{uploaded_file.name}:{content_hash}"
            input_rows, was_capped = _load_uploaded_csv_once(source_key, uploaded_file)
            if was_capped:
                callout(
                    "warning",
                    f"The upload has more than {MAX_UPLOAD_ROWS:,} rows. "
                    f"Only the first {MAX_UPLOAD_ROWS:,} rows will be scored.",
                )
            st.caption(f"Source: {uploaded_file.name} · {len(input_rows):,} rows")

        if st.session_state.get("selected_flow_source") != source_key:
            st.session_state["selected_flow"] = None
            st.session_state["selected_flow_source"] = source_key

        with st.spinner("Scanning flows…"):
            scored, warnings = _score_input_once(source_key, input_rows, artifact)
    except Exception as error:
        callout("error", str(error))
        return

    # Reapply the slider to this cached-data copy without running the model again.
    scored["flagged"] = scored["risk_score"].ge(float(threshold))
    scored["flow_id"] = np.arange(1, len(scored) + 1, dtype=np.int64)
    total_flows = len(scored)
    flagged = scored.loc[scored["flagged"]].sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    flagged_count = len(flagged)
    review_share = 100.0 * flagged_count / total_flows
    workload_reduced = 100.0 - review_share
    highest_risk = float(scored["risk_score"].max())

    threat_level = (
        "NORMAL"
        if review_share < THREAT_NORMAL_MAX_PCT
        else "ELEVATED"
        if review_share <= THREAT_ELEVATED_MAX_PCT
        else "CRITICAL"
    )
    threat_banner(threat_level, review_share, flagged_count)

    kpi_columns = st.columns(5, gap="small")
    cards = [
        ("Flows scanned", f"{total_flows:,}", "total flows scored", "normal"),
        (
            "Alerts raised",
            f"{flagged_count:,}",
            "above alert threshold",
            "threat" if flagged_count > 0 else "normal",
        ),
        ("Review load", f"{review_share:.1f}%", "needs analyst review", "normal"),
        ("Peak risk", f"{highest_risk:.1%}", "highest risk score", "normal"),
        ("Workload saved", f"{workload_reduced:.1f}%", "vs. full manual review", "normal"),
    ]
    for column, (label, value, caption, tone) in zip(kpi_columns, cards):
        with column:
            kpi_card(label, value, caption, tone=tone)

    histogram_column, services_column = st.columns(2, gap="large")
    with histogram_column:
        section_header(
            "Risk profile",
            "See how many flows fall below or above the alert line.",
            help_text="Risk score is the saved model's estimated probability that a flow is an attack.",
        )
        st.plotly_chart(
            _risk_histogram(scored, threshold),
            width="stretch",
            alt="Distribution of model attack probabilities with the alert threshold marked",
        )
    with services_column:
        render_targeted_services(flagged)

    render_priority_queue(flagged, artifact)

    map_column, log_column = st.columns(2, gap="large")
    with map_column:
        try:
            with st.spinner("Preparing flow behaviour map…"):
                render_threat_map(scored, artifact["features"], demo_loader())
        except Exception as error:
            callout("warning", f"The flow behaviour map is unavailable: {error}")
    with log_column:
        render_alert_log(flagged, artifact)

    section_header(
        "Alert review",
        "Open the table to review alerts ranked from highest to lowest risk.",
        help_text="Flow = one conversation between two computers. Select a row to save it for inspection.",
    )
    all_flagged_expander = st.expander("All flagged flows", on_change="rerun")
    if all_flagged_expander.open:
        with all_flagged_expander:
            if flagged.empty:
                callout("success", "No flows cross the current alert line. The queue is clear.")
            view_columns = ["flow_id", "risk_score", "risk_level"]
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
                    "flow_id": st.column_config.NumberColumn("Flow ID", format="%d"),
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk score",
                        format="%.3f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
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

            full_table_expander = st.expander("View all columns", on_change="rerun")
            if full_table_expander.open:
                with full_table_expander:
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

    if LABEL_COL in scored.columns:
        _render_ground_truth(scored)


@st.cache_data(max_entries=5)
def _cached_preset_flow(kind, _sample_rows):
    """Reuse the real sample flow chosen for each preset."""
    return get_preset_flow(_sample_rows, kind)


def _top_important_features(artifact):
    """Return the ten saved-model features with the largest importance."""
    features = list(artifact["features"])
    importances = artifact.get("feature_importances", {})
    if not isinstance(importances, dict):
        importances = dict(zip(features, importances))
    return sorted(
        features,
        key=lambda name: float(importances.get(name, 0.0)),
        reverse=True,
    )[:INSPECT_IMPORTANT_FEATURE_COUNT]


def _format_flow_value(value):
    """Show a real feature value compactly in a table or explanation."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(number):
        return "Missing"
    return f"{float(number):,.4g}"


def _risk_gauge(risk_score, threshold, flagged):
    """Build a dark gauge with a visible alert-threshold marker."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(risk_score),
            number={"valueformat": ".1%", "font": {"size": 32}},
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickmode": "array",
                    "tickvals": [0, float(threshold), 1],
                    "ticktext": ["0", f"Alert {threshold:.2f}", "1"],
                },
                "bar": {"color": THREAT_RED if flagged else PURPLE},
                "bgcolor": BG,
                "bordercolor": PURPLE,
                "borderwidth": 1,
                "steps": [
                    {"range": [0, float(threshold)], "color": SURFACE},
                    {"range": [float(threshold), 1], "color": DEEP_RED},
                ],
                "threshold": {
                    "line": {"color": THREAT_RED, "width": 4},
                    "thickness": 0.78,
                    "value": float(threshold),
                },
            },
        )
    )
    figure.update_layout(title="Attack probability", height=260)
    return style_figure(figure)


def _render_flow_explanation(scored_row, artifact, threshold):
    """Show the flow verdict, risk score, and strongest benign deviations."""
    risk_score = float(scored_row["risk_score"])
    flagged = risk_score >= float(threshold)
    gauge_column, verdict_column = st.columns([1.5, 1], gap="large")
    with gauge_column:
        st.plotly_chart(
            _risk_gauge(risk_score, threshold, flagged),
            width="stretch",
            alt="Risk gauge with the current alert threshold marked",
        )
    with verdict_column:
        st.markdown("#### Verdict")
        verdict_badge(flagged)
        st.caption(
            "This flow crosses the alert line."
            if flagged
            else "This flow is below the alert line."
        )
        st.caption(f"Alert threshold: {float(threshold):.3f}")

    explanations = explain_flow(
        scored_row,
        artifact,
        top_k=INSPECT_EXPLANATION_FEATURE_COUNT,
    )
    section_header(
        "Why was this flagged?",
        "Compare the strongest differences in this flow with typical benign traffic.",
        help_text="These contribution bars are an explanation approximation, not SHAP values.",
    )
    if not explanations:
        callout("info", "No finite feature differences are available to explain this flow.")
        return

    feature_names = [item["friendly_name"] for item in explanations]
    contribution_values = [float(item["contribution"]) for item in explanations]
    bar_color = THREAT_RED if flagged else PURPLE_GLOW
    figure = go.Figure(
        go.Bar(
            x=contribution_values[::-1],
            y=feature_names[::-1],
            orientation="h",
            marker_color=bar_color,
            hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Largest feature contributions",
        xaxis_title="Importance × robust deviation",
        yaxis_title=None,
        height=300,
        showlegend=False,
    )
    style_figure(figure)
    st.plotly_chart(
        figure,
        width="stretch",
        alt="Top five feature contributions for this flow",
    )
    st.write(explanations[0]["explanation"])

    comparison = pd.DataFrame(
        [
            {
                "Feature": item["friendly_name"],
                "Flow value": _format_flow_value(item["value"]),
                "Typical normal value": _format_flow_value(
                    item["typical_benign_value"]
                ),
                "Direction": item["direction"],
            }
            for item in explanations
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        height=220,
        alt="This flow's feature values compared with typical benign values",
    )


def _render_single_flow(row, artifact, threshold):
    """Score one row through the same saved model used by Threat Desk."""
    try:
        with st.spinner("Scoring this flow…"):
            result, _ = score_flows(pd.DataFrame([row]), artifact, threshold=threshold)
        _render_flow_explanation(result.iloc[0], artifact, threshold)
    except Exception as error:
        callout("error", str(error))


def _render_build_flow(artifact, threshold, sample_rows):
    """Build a real flow from a preset and edit the ten strongest features."""
    preset_kind = st.selectbox(
        "Start from a real sample flow",
        options=list(PRESET_LABELS),
        key="inspect_preset_kind",
    )
    try:
        preset = _cached_preset_flow(preset_kind, sample_rows)
    except ValueError as error:
        callout("error", str(error))
        return

    important_features = _top_important_features(artifact)
    edited_values = {}
    with st.form("build_flow_form", border=False):
        st.caption(
            "Adjust these important features. Every other feature stays from the selected real flow."
        )
        columns = st.columns(2, gap="medium")
        for index, feature in enumerate(important_features):
            value = pd.to_numeric(pd.Series([preset.get(feature)]), errors="coerce").iloc[0]
            if pd.isna(value) or not np.isfinite(value) or value < 0:
                value = 0.0
            step = 1.0 if float(value).is_integer() else 0.1
            maximum = 65535.0 if feature == "Destination Port" else None
            with columns[index % len(columns)]:
                edited_values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=maximum,
                    value=float(value),
                    step=step,
                    key=f"inspect_{preset_kind}_{feature}",
                    help=FRIENDLY_NAMES.get(feature, feature),
                )
        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            icon=":material/search:",
        )

    if submitted:
        non_finite = [
            feature
            for feature, value in edited_values.items()
            if not np.isfinite(float(value))
        ]
        negative = [
            feature for feature, value in edited_values.items() if float(value) < 0
        ]
        if non_finite:
            callout("error", f"Enter a finite number for: {', '.join(non_finite)}.")
            return
        if negative:
            callout("error", f"Negative values are not allowed for: {', '.join(negative)}.")
            return

        built_flow = preset.copy()
        for feature, value in edited_values.items():
            built_flow[feature] = float(value)
        st.session_state["last_built_flow"] = {
            "preset": preset_kind,
            "row": built_flow.to_dict(),
        }

    saved_flow = st.session_state.get("last_built_flow")
    if saved_flow and saved_flow.get("preset") == preset_kind:
        section_header(
            "Prediction",
            "The edited flow is scored with the saved Random Forest pipeline.",
        )
        _render_single_flow(saved_flow["row"], artifact, threshold)


def render_inspect_flow(artifact, threshold, demo_loader):
    """Let an examiner inspect a saved flow or build one from a real preset."""
    section_header(
        "Inspect a Flow",
        "Examine one connection, see its saved-model score, and understand the strongest differences.",
        help_text="Flow = one conversation between two computers.",
    )
    sample_rows = demo_loader()
    if sample_rows.empty:
        callout("error", "The demo sample is empty, so there are no flows to inspect.")
        return

    mode = st.segmented_control(
        "Inspection mode",
        options=["Inspect a flow", "Build a flow"],
        default="Inspect a flow",
        key="inspect_mode",
    )
    if mode == "Build a flow":
        _render_build_flow(artifact, threshold, sample_rows)
        return

    selected_flow = st.session_state.get("selected_flow")
    sources = ["Choose a demo row"]
    if selected_flow is not None:
        sources.insert(0, "Selected in Threat Desk")
    source = st.radio("Flow source", sources, horizontal=True, key="inspect_flow_source")
    if source == "Selected in Threat Desk":
        row = selected_flow
        flow_id = row.get("flow_id", "selected")
        st.caption(f"Inspecting Threat Desk row #{flow_id}.")
    else:
        row_number = st.number_input(
            "Demo row number",
            min_value=1,
            max_value=len(sample_rows),
            value=1,
            step=1,
            key="inspect_demo_row_number",
        )
        row = sample_rows.iloc[int(row_number) - 1]
        st.caption(f"Inspecting demo row {int(row_number):,} of {len(sample_rows):,}.")
    _render_single_flow(row, artifact, threshold)


def _build_attack_scenario(sample_rows):
    """Create four equal, repeatable phases from labeled demo rows."""
    if LABEL_COL not in sample_rows.columns:
        raise ValueError("The demo sample needs a Label column for the replay.")
    labels = sample_rows[LABEL_COL].astype("string").str.strip()
    available = [int(labels.eq(label).sum()) for _, label in LIVE_MONITOR_PHASES]
    phase_rows = min(LIVE_MONITOR_ROWS_PER_PHASE, *available)
    phase_rows = (phase_rows // LIVE_MONITOR_BATCH_SIZE) * LIVE_MONITOR_BATCH_SIZE
    if phase_rows < LIVE_MONITOR_BATCH_SIZE:
        raise ValueError(
            "The demo sample needs at least 100 BENIGN, DDoS, and PortScan rows for the replay."
        )

    phases = []
    phase_names = []
    for index, (phase_name, label) in enumerate(LIVE_MONITOR_PHASES):
        class_rows = sample_rows.loc[labels.eq(label)]
        phase = class_rows.sample(
            n=phase_rows,
            random_state=RANDOM_STATE + index,
            replace=False,
        )
        phases.append(phase)
        phase_names.extend([phase_name] * phase_rows)
    return pd.concat(phases, ignore_index=True), phase_names, phase_rows


def _live_history_figure(history):
    """Plot the flagged share recorded for each scored batch."""
    figure = go.Figure(
        go.Scatter(
            x=[row["batch"] for row in history],
            y=[row["flagged_pct"] for row in history],
            mode="lines+markers",
            line={"color": PURPLE_GLOW, "width": 2},
            marker={
                "color": [
                    THREAT_RED if row["flagged_pct"] > 50 else PURPLE
                    for row in history
                ],
                "size": 9,
            },
            hovertemplate="Batch %{x}<br>Flagged: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title="Flagged traffic by batch",
        xaxis_title="Batch",
        yaxis_title="Flows flagged (%)",
        yaxis={"range": [0, 100]},
        height=330,
        showlegend=False,
    )
    return style_figure(figure)


def _show_live_snapshot(result):
    """Show saved replay counters, batch chart, and recent alerts."""
    counters = st.columns(3, gap="small")
    counter_cards = [
        ("Flows seen", f"{result['flows_seen']:,}", "scored in 100-flow batches", "normal"),
        (
            "Alerts raised",
            f"{result['alerts_raised']:,}",
            "above the replay threshold",
            "threat" if result["alerts_raised"] else "normal",
        ),
        (
            "Attacks caught",
            f"{result['attacks_caught']:,}",
            "known attack rows flagged",
            "normal",
        ),
    ]
    for column, (label, value, caption, tone) in zip(counters, counter_cards):
        with column:
            kpi_card(label, value, caption, tone=tone)

    chart_column, feed_column = st.columns([1.4, 1], gap="large")
    with chart_column:
        section_header(
            "Batch trend",
            "Track the share of each 100-flow batch that crossed the alert line.",
        )
        st.plotly_chart(
            _live_history_figure(result["history"]),
            width="stretch",
            alt="Percent of replay flows flagged in each 100-flow batch",
        )
    with feed_column:
        section_header(
            "Alert feed",
            "The latest flagged demo flows, with the strongest explanation for each.",
        )
        if result["alert_feed"]:
            feed = pd.DataFrame(result["alert_feed"])
            st.dataframe(
                feed,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk score", format="%.3f", min_value=0.0, max_value=1.0
                    )
                },
                hide_index=True,
                width="stretch",
                height=330,
                alt="Latest replay alerts with time, risk score, and reason",
            )
        else:
            callout("info", "No flows crossed the alert line during this replay.")


def _show_live_summary(result):
    """Compare model alerts with known labels from the completed replay."""
    section_header(
        "Replay summary",
        f"{result['flows_seen']:,} labeled demo flows scored at threshold {result['threshold']:.3f}.",
    )
    summary_columns = st.columns(3, gap="small")
    summary_cards = [
        (
            "Alerts raised",
            f"{result['alerts_raised']:,}",
            "all flows above threshold",
            "threat" if result["alerts_raised"] else "normal",
        ),
        (
            "Attacks caught",
            f"{result['attacks_caught']:,} / {result['attacks_total']:,}",
            "known DDoS and PortScan flows",
            "normal",
        ),
        (
            "Attacks missed",
            f"{result['attacks_missed']:,}",
            "known attacks below threshold",
            "threat" if result["attacks_missed"] else "normal",
        ),
    ]
    for column, (label, value, caption, tone) in zip(summary_columns, summary_cards):
        with column:
            kpi_card(label, value, caption, tone=tone)
    if result["attacks_missed"]:
        callout(
            "error",
            f"The replay missed {result['attacks_missed']:,} labeled attacks at this threshold.",
        )
    else:
        callout("info", "All labeled attack flows in this replay crossed the alert line.")


def _run_live_monitor(artifact, threshold, sample_rows, speed):
    """Score a deterministic four-phase replay and update the live placeholders."""
    scenario, phase_names, phase_rows = _build_attack_scenario(sample_rows)
    delay = LIVE_MONITOR_SPEED_SECONDS[speed]
    history = []
    alert_feed = []
    alerts_raised = 0
    attacks_total = 0
    attacks_caught = 0
    flows_seen = 0
    total_batches = len(scenario) // LIVE_MONITOR_BATCH_SIZE

    banner_slot = st.empty()
    status_slot = st.empty()
    progress = st.progress(0.0, text="Preparing replay")
    counter_slot = st.empty()
    chart_column, feed_column = st.columns([1.4, 1], gap="large")
    with chart_column:
        section_header(
            "Batch trend",
            "Track the share of each 100-flow batch that crossed the alert line.",
        )
        chart_slot = st.empty()
    with feed_column:
        section_header(
            "Alert feed",
            "The latest flagged demo flows, with a plain-English reason.",
        )
        feed_slot = st.empty()

    for batch_number, start in enumerate(
        range(0, len(scenario), LIVE_MONITOR_BATCH_SIZE),
        start=1,
    ):
        batch = scenario.iloc[start : start + LIVE_MONITOR_BATCH_SIZE]
        scored, _ = score_flows(batch, artifact, threshold=threshold)
        batch_flags = scored["flagged"].to_numpy(dtype=bool)
        batch_labels = scored[LABEL_COL].astype("string").str.strip()
        actual_attacks = batch_labels.ne(BENIGN_LABEL).to_numpy(dtype=bool)
        batch_alerts = int(batch_flags.sum())
        flagged_pct = 100.0 * batch_alerts / len(scored)
        flows_seen += len(scored)
        alerts_raised += batch_alerts
        attacks_total += int(actual_attacks.sum())
        attacks_caught += int((batch_flags & actual_attacks).sum())
        current_phase = phase_names[start]

        history.append(
            {
                "batch": batch_number,
                "phase": current_phase,
                "flagged_pct": flagged_pct,
                "alerts": batch_alerts,
            }
        )
        with status_slot.container():
            st.caption(f"Phase: {current_phase} · Batch {batch_number} of {total_batches}")
        with banner_slot.container():
            threat_banner(
                "CRITICAL" if flagged_pct > 50 else "NORMAL",
                flagged_pct,
                batch_alerts,
            )

        if batch_alerts:
            highest_alerts = scored.loc[scored["flagged"]].nlargest(8, "risk_score")
            for _, alert in highest_alerts.iterrows():
                explanations = explain_flow(alert, artifact, top_k=1)
                reason = (
                    explanations[0]["explanation"]
                    if explanations
                    else "No finite feature difference is available."
                )
                alert_feed.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "risk_score": float(alert["risk_score"]),
                        "reason": reason,
                    }
                )
            alert_feed = alert_feed[-8:]

        with counter_slot.container():
            counters = st.columns(3, gap="small")
            values = [
                ("Flows seen", f"{flows_seen:,}", "scored so far", "normal"),
                (
                    "Alerts raised",
                    f"{alerts_raised:,}",
                    "above the current threshold",
                    "threat" if alerts_raised else "normal",
                ),
                (
                    "Attacks caught",
                    f"{attacks_caught:,}",
                    "known labeled attacks",
                    "normal",
                ),
            ]
            for column, (label, value, caption, tone) in zip(counters, values):
                with column:
                    kpi_card(label, value, caption, tone=tone)
        with chart_slot.container():
            st.plotly_chart(
                _live_history_figure(history),
                width="stretch",
                alt="Percent of replay flows flagged in each 100-flow batch",
            )
        with feed_slot.container():
            if alert_feed:
                st.dataframe(
                    pd.DataFrame(alert_feed),
                    column_config={
                        "risk_score": st.column_config.ProgressColumn(
                            "Risk score", format="%.3f", min_value=0.0, max_value=1.0
                        )
                    },
                    hide_index=True,
                    width="stretch",
                    height=330,
                    alt="Latest replay alerts with time, risk score, and reason",
                )
            else:
                callout("info", "No flows have crossed the alert line yet.")

        progress.progress(
            batch_number / total_batches,
            text=f"Scored {flows_seen:,} of {len(scenario):,} flows",
        )
        if batch_number < total_batches:
            time.sleep(delay)

    result = {
        "flows_seen": flows_seen,
        "alerts_raised": alerts_raised,
        "attacks_total": attacks_total,
        "attacks_caught": attacks_caught,
        "attacks_missed": attacks_total - attacks_caught,
        "history": history,
        "alert_feed": alert_feed,
        "threshold": float(threshold),
        "phase_rows": phase_rows,
    }
    status_slot.caption("Replay complete · all four phases were scored.")
    return result


def render_live_monitor(artifact, threshold, demo_loader):
    """Replay a deterministic labeled scenario and show its detection results."""
    section_header(
        "Live Monitor",
        "Watch normal demo traffic, a DDoS burst, normal traffic, and a port-scan burst.",
        help_text="This is a replay of labeled demo flows; it does not generate network traffic.",
    )
    speed = st.select_slider(
        "Playback speed",
        options=list(LIVE_MONITOR_SPEED_SECONDS),
        value="Normal",
        key="live_monitor_speed",
        help="The replay scores one batch of 100 flows per update.",
    )
    run_scenario = st.button(
        "Run attack scenario",
        type="primary",
        icon=":material/play_arrow:",
        key="run_attack_scenario",
    )
    sample_rows = demo_loader()
    if run_scenario:
        try:
            with st.spinner("Starting deterministic replay…"):
                result = _run_live_monitor(artifact, threshold, sample_rows, speed)
            st.session_state["live_monitor_result"] = result
            _show_live_summary(result)
        except Exception as error:
            callout("error", str(error))
        return

    result = st.session_state.get("live_monitor_result")
    if result:
        _show_live_snapshot(result)
        _show_live_summary(result)


@st.cache_data(max_entries=16)
def _cached_report_json(path_text, modified_ns, file_size):
    """Read one version of a saved JSON report and cache it until it changes."""
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


@st.cache_data(max_entries=16)
def _cached_report_csv(path_text, modified_ns, file_size):
    """Read one version of a saved report table and cache it until it changes."""
    return pd.read_csv(path_text)


@st.cache_data(max_entries=16)
def _cached_report_image(path_text, modified_ns, file_size):
    """Read one saved report figure and cache its bytes."""
    return Path(path_text).read_bytes()


def _report_signature(path):
    """Return file metadata used to refresh cached report artifacts."""
    report_path = Path(path)
    if not report_path.is_file():
        return None
    metadata = report_path.stat()
    return str(report_path), metadata.st_mtime_ns, metadata.st_size


def _load_report_json(path):
    signature = _report_signature(path)
    return None if signature is None else _cached_report_json(*signature)


def _load_report_csv(path):
    signature = _report_signature(path)
    return None if signature is None else _cached_report_csv(*signature)


def _show_report_image(path, alt):
    """Display a saved report image or a clear missing-file message."""
    signature = _report_signature(path)
    if signature is None:
        callout("warning", f"Saved report figure is missing: {Path(path).name}.")
        return
    image_bytes = _cached_report_image(*signature)
    st.image(image_bytes, width="stretch", alt=alt)


def _highlight_random_forest(row):
    """Tint the Random Forest row so the main model is easy to find."""
    model = str(row.get("model", "")).casefold().replace(" ", "")
    if model.startswith("randomforest"):
        return [
            f"background-color: {PURPLE_WASH}; color: {TEXT}; font-weight: 700"
            for _ in row
        ]
    return [""] * len(row)


def _format_saved_metric(value):
    """Format a saved 0-to-1 report metric as a percentage."""
    return "—" if value is None else f"{float(value):.1%}"


def render_model_report_card():
    """Present only the saved evaluation outputs from the reports directory."""
    section_header(
        "Model Report Card",
        "A read-only summary of the saved random-test and day-shift evaluations.",
        help_text="These values and figures are loaded from reports; this tab does not train or score a model.",
    )
    try:
        with st.spinner("Loading saved report files…"):
            metrics = _load_report_json(METRICS_JSON_PATH)
            comparison = _load_report_csv(MODEL_COMPARISON_CSV_PATH)
    except Exception as error:
        metrics = None
        comparison = None
        callout("error", f"Could not read the saved report files: {error}")

    random_test = (metrics or {}).get("random_test", {})
    random_forest = random_test.get("models", {}).get("RandomForest")
    section_header(
        "Held-out test performance",
        "Four saved Random Forest metrics from the untouched random test split.",
    )
    if random_forest is None:
        callout("warning", "Random Forest test metrics are missing from reports/metrics.json.")
    else:
        tile_columns = st.columns(4, gap="small")
        for column, (label, key) in zip(
            tile_columns,
            [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("PR-AUC", "pr_auc"),
            ],
        ):
            with column:
                kpi_card(
                    label,
                    _format_saved_metric(random_forest.get(key)),
                    f"Random Forest · {random_forest.get('n_flows', 0):,} test flows",
                )

    section_header(
        "Model comparison",
        "Compare the saved rules, logistic regression, decision tree, and Random Forest results.",
    )
    if comparison is None or comparison.empty:
        callout("warning", "The saved model comparison table is missing or empty.")
    else:
        if "test_split" in comparison.columns:
            comparison = comparison.loc[
                comparison["test_split"].eq("random_test")
            ].copy()
        visible_columns = [
            name
            for name in (
                "model",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "false_positive_rate",
                "scoring_flows_per_second",
            )
            if name in comparison.columns
        ]
        if "model" not in visible_columns:
            callout("warning", "The saved model comparison has no model-name column.")
        else:
            table = comparison.loc[:, visible_columns].style.apply(
                _highlight_random_forest,
                axis=1,
            )
            st.dataframe(
                table,
                column_config={
                    name: st.column_config.NumberColumn(
                        name.replace("_", " ").title(),
                        format="%.0f" if name == "scoring_flows_per_second" else "%.3f",
                    )
                    for name in visible_columns
                    if name != "model"
                },
                hide_index=True,
                width="stretch",
                height=250,
                alt="Saved test metrics for each baseline and the Random Forest",
            )

    section_header(
        "Confusion matrix",
        "Counts of benign and attack flows by the saved Random Forest decision.",
    )
    _show_report_image(
        CONFUSION_MATRIX_FIGURE_PATH,
        "Saved Random Forest confusion matrix with counts and percentages",
    )

    left, right = st.columns(2, gap="large")
    with left:
        section_header(
            "Detection by attack type",
            "Recall for each original attack label on the random test split.",
        )
        _show_report_image(
            DETECTION_BY_ATTACK_FIGURE_PATH,
            "Saved Random Forest and rules recall by attack type",
        )
    with right:
        section_header(
            "Feature importance",
            "The saved Random Forest's twenty most influential features.",
        )
        _show_report_image(
            FEATURE_IMPORTANCE_FIGURE_PATH,
            "Saved top twenty Random Forest feature importances",
        )

    section_header(
        "Unseen attack types",
        "A day-shift check trains on Monday–Thursday and evaluates on Friday.",
        help_text="The Friday split checks how learned patterns carry to a different day of traffic.",
    )
    day_shift = (metrics or {}).get("dayshift_friday", {}).get("metrics", {})
    if day_shift:
        metric_columns = st.columns(2, gap="small")
        with metric_columns[0]:
            kpi_card(
                "Friday recall",
                _format_saved_metric(day_shift.get("recall")),
                f"{day_shift.get('n_flows', 0):,} Friday flows",
            )
        with metric_columns[1]:
            kpi_card(
                "Friday PR-AUC",
                _format_saved_metric(day_shift.get("pr_auc")),
                "saved day-shift evaluation",
            )
    else:
        callout("info", "Saved Friday metrics are not present in reports/metrics.json.")
    _show_report_image(
        DAYSHIFT_DETECTION_FIGURE_PATH,
        "Saved Friday day-shift recall by attack type",
    )
    callout(
        "warning",
        "The model recognises attack styles it has learned; unfamiliar variants can be missed.",
    )

    with st.expander("Limitations"):
        for limitation in REPORT_LIMITATIONS:
            st.markdown(f"- {limitation}")


def _pipeline_dot():
    """Return the pipeline diagram in Graphviz DOT format."""
    return f"""
digraph CyberSift {{
  graph [rankdir=LR, bgcolor="{BG}", pad="0.2", nodesep="0.3", ranksep="0.4"];
  node [shape=box, style="rounded,filled", fontname="Inter", fontsize="11",
        fontcolor="{TEXT}", color="{BORDER}", margin="0.16,0.12"];
  edge [color="{PURPLE_GLOW}", fontcolor="{MUTED}", fontname="Inter", fontsize="9"];

  raw [label="Raw flows", fillcolor="{SURFACE}"];
  clean [label="Clean + prepare_features()", fillcolor="{CARD}"];
  imputer [label="Median imputer\\nlearned on training data", fillcolor="{CARD}"];
  forest [label="Random Forest\\n100 trees vote", fillcolor="{PURPLE}"];
  score [label="Risk score\\n0 to 1", fillcolor="{CARD}"];
  threshold [label="Alert threshold", fillcolor="{CARD}"];
  alert [label="Alert + explanation", fillcolor="{DEEP_RED}", color="{THREAT_RED}"];

  raw -> clean -> imputer -> forest -> score -> threshold -> alert;
}}
"""


def render_how_it_works():
    """Explain the flow pipeline, dataset, and common terms in plain English."""
    section_header(
        "How it works",
        "Follow one network flow from the cleaned measurements to an analyst alert.",
    )
    st.graphviz_chart(
        _pipeline_dot(),
        width="stretch",
        height=300,
        alt="CyberSift data pipeline from raw flows through a Random Forest to an alert",
    )

    with st.container(border=True):
        st.subheader("Random Forest in 60 seconds")
        st.write(
            "The cleaner selects 69 numeric measurements for each flow. A median imputer "
            "fills missing values using medians learned from training data. One hundred "
            "decision trees each estimate whether the flow is an attack; their combined "
            "probability becomes its risk score. Scores above the chosen threshold become "
            "alerts, and the explanation ranks unusual feature values against benign traffic."
        )

    section_header(
        "Dataset summary",
        "Counts below come from the saved EDA reports and project preprocessing settings.",
    )
    try:
        with st.spinner("Loading the saved dataset summary…"):
            class_balance = _load_report_csv(EDA_CLASS_BALANCE_CSV_PATH)
            attack_types = _load_report_csv(EDA_ATTACK_TYPE_COUNTS_CSV_PATH)
        total_rows = (
            int(pd.to_numeric(class_balance["count"], errors="coerce").sum())
            if class_balance is not None and "count" in class_balance.columns
            else None
        )
        attack_type_count = len(attack_types) if attack_types is not None else None
    except Exception as error:
        total_rows = None
        attack_type_count = None
        callout("warning", f"Could not read the saved EDA summary: {error}")

    summary_columns = st.columns(4, gap="small")
    summary_cards = [
        ("Source", DATASET_SOURCE.split()[0], "labeled flow CSVs"),
        ("Flows", f"{total_rows:,}" if total_rows is not None else "Unavailable", "after cleaning"),
        ("Features", f"{EXPECTED_FEATURE_COUNT}", "numeric model inputs"),
        (
            "Attack types",
            str(attack_type_count) if attack_type_count is not None else "Unavailable",
            "in the saved label summary",
        ),
    ]
    for column, (label, value, caption) in zip(summary_columns, summary_cards):
        with column:
            kpi_card(label, value, caption)
    st.caption(f"Dataset source: {DATASET_SOURCE}")

    with st.container(border=True):
        st.subheader("Preprocessing")
        for step in PREPROCESSING_STEPS:
            st.markdown(f"- {step}")

    with st.expander("Glossary"):
        glossary = pd.DataFrame(GLOSSARY_ENTRIES, columns=["Term", "Meaning"])
        st.dataframe(
            glossary,
            hide_index=True,
            width="stretch",
            height=390,
            alt="Plain-English definitions of ten CyberSift terms",
        )


def render_coming_soon():
    """Show the placeholder for tabs scheduled for a later project phase."""
    callout("info", "coming next")
