"""Threat Desk panels built from saved-model scores and explanations."""

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cybersift.config import (
    ALERT_LOG_LIMIT,
    FLOW_MAP_CACHE_ENTRIES,
    FLOW_MAP_MAX_POINTS,
    FLOW_MAP_PCA_COMPONENTS,
    FLOW_MAP_TOP_RISK_LIMIT,
    LABEL_COL,
    PORT_SERVICE_NAMES,
    PRIORITY_CRITICAL_MIN,
    PRIORITY_HIGH_MIN,
    PRIORITY_QUEUE_SIZE,
    RANDOM_STATE,
    SERVICE_CHART_TOP_PORTS,
)
from cybersift.data import prepare_features
from cybersift.explain import explain_flow
from cybersift.theme import (
    PURPLE_GLOW,
    THREAT_RED,
)
from cybersift.ui_components import callout, section_header, style_figure


def _signed_log1p(values):
    """Compress large positive and negative feature values symmetrically."""
    return np.sign(values) * np.log1p(np.abs(values))


@st.cache_resource(max_entries=1)
def _fit_demo_projection(features, _demo_rows):
    """Fit the behavior-map transform once using only the bundled demo sample."""
    if len(_demo_rows) < FLOW_MAP_PCA_COMPONENTS:
        raise ValueError("The bundled demo sample needs at least two rows for the flow map.")

    demo_features = prepare_features(_demo_rows, features)
    projection = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("signed_log", FunctionTransformer(_signed_log1p, validate=False)),
            ("standardizer", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=FLOW_MAP_PCA_COMPONENTS,
                    random_state=RANDOM_STATE,
                    svd_solver="randomized",
                ),
            ),
        ]
    )
    projection.fit(demo_features)
    return projection


@st.cache_data(max_entries=FLOW_MAP_CACHE_ENTRIES)
def _cached_flow_coordinates(
    feature_rows, features, risk_scores, labels, flow_ids, _projection
):
    """Cache projected points so threshold changes only update their colors."""
    numeric_features = prepare_features(feature_rows, features)
    coordinates = _projection.transform(numeric_features)
    return pd.DataFrame(
        {
            "flow_id": flow_ids,
            "component_1": coordinates[:, 0],
            "component_2": coordinates[:, 1],
            "risk_score": risk_scores,
            "label": labels,
        }
    )


def _service_name(port):
    """Return a familiar service label for a destination port."""
    number = pd.to_numeric(pd.Series([port]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(number):
        return "Port unknown"
    port_number = int(number)
    return PORT_SERVICE_NAMES.get(port_number, f"Port {port_number}")


def _display_number(value):
    """Format a flow measurement without changing its underlying value."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(number):
        return "unknown"
    return f"{float(number):,.0f}"


def _why_text(row, artifact, short=False):
    """Use the explanation module's strongest benign deviation for a flow."""
    explanations = explain_flow(row, artifact, top_k=1)
    if not explanations:
        return "No finite feature deviation is available."
    explanation = explanations[0]
    if short:
        return f"{explanation['friendly_name']} {explanation['direction']}"
    return explanation["explanation"]


def _priority_card(rank, row, artifact):
    """Render one scored alert with its top explanation."""
    risk = float(row["risk_score"])
    if risk >= PRIORITY_CRITICAL_MIN:
        severity = "CRITICAL"
        severity_class = "critical"
    elif risk >= PRIORITY_HIGH_MIN:
        severity = "HIGH"
        severity_class = "high"
    else:
        severity = "MEDIUM"
        severity_class = "medium"

    port = _display_number(row.get("Destination Port"))
    service = _service_name(row.get("Destination Port"))
    if port != "unknown" and service.startswith("Port "):
        service = f"Port {port}"
    duration = _display_number(row.get("Flow Duration"))
    forward = _display_number(row.get("Total Fwd Packets"))
    backward = _display_number(row.get("Total Backward Packets"))
    flow_id = int(row["flow_id"])
    why = _why_text(row, artifact)
    safe_why = html.escape(why, quote=True)
    bar_width = min(max(risk * 100.0, 0.0), 100.0)

    markup = f"""
    <div class="cyber-priority-card">
      <div class="cyber-priority-card__top">
        <span class="cyber-rank-badge">{rank:02d}</span>
        <span class="cyber-severity-chip cyber-severity-chip--{severity_class}">{severity}</span>
      </div>
      <div class="cyber-priority-card__score-row">
        <span class="cyber-priority-card__score">{risk:.3f}</span>
        <span class="cyber-priority-card__score-label">risk score</span>
      </div>
      <div class="cyber-priority-card__track">
        <div class="cyber-priority-card__fill" style="width: {bar_width:.1f}%"></div>
      </div>
      <div class="cyber-priority-card__service">{html.escape(service)}</div>
      <div class="cyber-priority-card__facts">
        Port {html.escape(port)}<br>
        Duration {html.escape(duration)} · Packets {html.escape(forward)} out / {html.escape(backward)} back
      </div>
      <div class="cyber-priority-card__why" title="{safe_why}">
        <strong>Why:</strong> {safe_why}
      </div>
      <div class="cyber-priority-card__footer">
        Row #{flow_id}: open it in Inspect a Flow
      </div>
    </div>
    """
    return "\n".join(line.lstrip() for line in markup.splitlines()).strip()


def render_priority_queue(flagged, artifact):
    """Show the five highest-risk real flows with explanations."""
    section_header(
        "Priority queue",
        "Start with the flows the saved model considers most suspicious.",
        help_text="Flow = one conversation between two computers.",
    )
    if flagged.empty:
        st.markdown(
            '<div class="cyber-all-clear-card"><strong>All clear</strong>'
            "No flows are above the current alert line.</div>",
            unsafe_allow_html=True,
        )
        return

    highest_risk = flagged.nlargest(PRIORITY_QUEUE_SIZE, "risk_score")
    columns = st.columns(len(highest_risk), gap="small")
    for rank, (column, (_, row)) in enumerate(zip(columns, highest_risk.iterrows()), start=1):
        with column:
            st.markdown(
                _priority_card(rank, row, artifact),
                unsafe_allow_html=True,
            )


def render_targeted_services(flagged):
    """Plot the destination services most often present in flagged flows."""
    section_header(
        "Targeted services",
        "Which services are being hit?",
        help_text="Destination port identifies the network service a flow is trying to reach.",
    )
    if flagged.empty or "Destination Port" not in flagged.columns:
        callout("success", "No destination services are above the current alert line.")
        return

    ports = pd.to_numeric(flagged["Destination Port"], errors="coerce").dropna()
    ports = ports[np.isfinite(ports)].astype(int)
    if ports.empty:
        callout("info", "The flagged rows do not contain readable destination ports.")
        return

    counts = ports.value_counts().head(SERVICE_CHART_TOP_PORTS)
    service_labels = [f"{port} · {_service_name(port)}" for port in counts.index]
    figure = go.Figure(
        go.Bar(
            x=counts.to_numpy()[::-1],
            y=service_labels[::-1],
            orientation="h",
            marker_color=THREAT_RED,
            hovertemplate="%{y}<br>Flagged flows: %{x:,}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Flagged flows",
        yaxis_title=None,
        showlegend=False,
        height=330,
    )
    style_figure(figure)
    st.plotly_chart(
        figure,
        width="stretch",
        alt="The most common destination ports in flagged traffic",
    )


def _select_map_rows(scored):
    """Keep the highest-risk thousand and a fixed-seed sample of other flows."""
    row_count = len(scored)
    if row_count <= FLOW_MAP_MAX_POINTS:
        return scored.copy()

    scores = scored["risk_score"].to_numpy(dtype=float)
    top_count = min(FLOW_MAP_TOP_RISK_LIMIT, row_count)
    top_positions = np.argsort(-scores, kind="stable")[:top_count]
    remaining = np.setdiff1d(np.arange(row_count), top_positions, assume_unique=True)
    sample_count = min(FLOW_MAP_MAX_POINTS - top_count, len(remaining))
    random_positions = np.random.default_rng(RANDOM_STATE).choice(
        remaining, size=sample_count, replace=False
    )
    positions = np.concatenate([top_positions, random_positions])
    return scored.iloc[positions].copy()


def _flow_map_figure(map_data, has_label):
    """Draw fixed flow coordinates and color them by the current alert threshold."""
    figure = go.Figure()
    hover_template = "Risk score: %{customdata[0]:.3f}<br>Port: %{customdata[1]}"
    custom_columns = ["risk_score", "port_label"]
    if has_label:
        hover_template += "<br>Label: %{customdata[2]}"
        custom_columns.append("label")
    hover_template += "<extra></extra>"
    for flagged, name, color, size, opacity in (
        (False, "Not flagged", PURPLE_GLOW, 5, 0.42),
        (True, "Flagged", THREAT_RED, 8, 0.92),
    ):
        points = map_data.loc[map_data["flagged"].eq(flagged)]
        if points.empty:
            continue
        customdata = points[custom_columns].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=points["component_1"],
                y=points["component_2"],
                mode="markers",
                name=name,
                customdata=customdata,
                marker=dict(size=size, color=color, opacity=opacity),
                hovertemplate=hover_template,
            )
        )
    figure.update_layout(
        title="Flow behaviour map",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=390,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return style_figure(figure)


def render_threat_map(scored, features, demo_rows):
    """Project real flows onto the cached demo-fit PCA map."""
    section_header(
        "Threat map",
        "Explore which connections behave alike at a glance.",
        help_text="PCA compresses the 69 flow features into two display dimensions; some detail is lost.",
    )
    if scored.empty:
        callout("info", "No flows are available to place on the map.")
        return
    if len(scored) < 50:
        callout(
            "info",
            f"Small sample: all {len(scored)} available flows are shown; clusters are only a rough guide.",
        )

    map_rows = _select_map_rows(scored)
    feature_names = tuple(features)
    projection = _fit_demo_projection(feature_names, demo_rows)
    has_label = LABEL_COL in map_rows.columns
    labels = (
        tuple(map_rows[LABEL_COL].astype("string").fillna("Unknown").astype(str))
        if LABEL_COL in map_rows.columns
        else tuple("Not provided" for _ in range(len(map_rows)))
    )
    flow_ids = tuple(map_rows["flow_id"].astype(int))
    risk_scores = tuple(map_rows["risk_score"].astype(float))
    feature_rows = map_rows.loc[:, list(feature_names)]
    coordinates = _cached_flow_coordinates(
        feature_rows,
        feature_names,
        risk_scores,
        labels,
        flow_ids,
        projection,
    )
    coordinates["flagged"] = map_rows["flagged"].to_numpy(dtype=bool)
    if "Destination Port" in map_rows.columns:
        coordinates["port_label"] = [
            f"{_display_number(port)} / {_service_name(port)}"
            for port in map_rows["Destination Port"]
        ]
    else:
        coordinates["port_label"] = "unknown"

    figure = _flow_map_figure(coordinates, has_label)
    st.plotly_chart(
        figure,
        width="stretch",
        alt="Flow behavior map; red points are flagged connections and purple points are not flagged",
    )
    st.caption(
        "Each dot is one connection. Dots that sit close together behave alike. "
        "Red dots are flagged. (2-D projection for intuition; it loses detail.)"
    )


def _terminal_line(row, artifact):
    """Build one alert line from scored values and its explanation."""
    risk = float(row["risk_score"])
    flow_id = int(row["flow_id"])
    port = _display_number(row.get("Destination Port"))
    service = _service_name(row.get("Destination Port"))
    reason = html.escape(_why_text(row, artifact, short=True))
    safe_port = html.escape(port)
    safe_service = html.escape(service)
    return (
        '<div class="cyber-terminal-line">'
        f'<span class="cyber-terminal-label">[flow #</span>'
        f'<span class="cyber-terminal-value">{flow_id}</span>'
        '<span class="cyber-terminal-label">] </span>'
        '<span class="cyber-terminal-alert">ALERT</span> '
        '<span class="cyber-terminal-label">risk=</span>'
        f'<span class="cyber-terminal-value">{risk:.2f}</span> '
        '<span class="cyber-terminal-label">port=</span>'
        f'<span class="cyber-terminal-value">{safe_port}/{safe_service}</span> '
        '<span class="cyber-terminal-label">reason=</span>'
        f'<span class="cyber-terminal-value">"{reason}"</span>'
        '</div>'
    )


def render_alert_log(flagged, artifact):
    """Show the ten highest-risk alerts in a small terminal-style log."""
    section_header(
        "Alert log",
        "Review the ten highest-risk alerts and the strongest reason for each.",
        help_text="Reasons use the flow explanation approximation against benign traffic.",
    )
    if flagged.empty:
        lines = '<div class="cyber-terminal-empty">No alerts at the current threshold.</div>'
    else:
        highest = flagged.nlargest(ALERT_LOG_LIMIT, "risk_score")
        lines = "".join(_terminal_line(row, artifact) for _, row in highest.iterrows())
    st.markdown(
        '<div class="cyber-alert-log">'
        '<div class="cyber-terminal-header">'
        '<span class="cyber-terminal-dots" aria-hidden="true"><span></span><span></span><span></span></span>'
        '<span>PRIORITY ALERT LOG</span>'
        '</div>'
        f'<div class="cyber-terminal-body">{lines}'
        '<div class="cyber-terminal-cursor">&gt; '
        '<span class="cyber-terminal-cursor__underscore">_</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
