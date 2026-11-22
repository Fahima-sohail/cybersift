"""Reusable visual components for the CyberSift Streamlit app."""

import html
import textwrap

import streamlit as st

from cybersift.config import APP_PAGE_TITLE, APP_TAGLINE, COURSE_LABEL
from cybersift.theme import (
    BORDER,
    FONT_BODY,
    FONT_HEADING,
    FONT_MONO,
    MUTED,
    PURPLE,
    PURPLE_GLOW,
    PURPLE_GRID,
    SURFACE,
    TEXT,
    TRANSPARENT,
)


def _render_html(markup):
    """Remove indentation so Markdown treats the markup as HTML, not code."""
    clean_markup = "\n".join(line.lstrip() for line in markup.splitlines()).strip()
    st.markdown(clean_markup, unsafe_allow_html=True)


def render_hero():
    """Render the shield wordmark and system status at the top of the app."""
    shield = textwrap.dedent(
        f"""
        <svg class="cyber-shield" viewBox="0 0 48 54" role="img" aria-label="CyberSift shield">
          <defs>
            <linearGradient id="cyber-shield-gradient" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="{PURPLE_GLOW}" />
              <stop offset="100%" stop-color="{PURPLE}" />
            </linearGradient>
          </defs>
          <path d="M24 3 43 10v14c0 13-8 21-19 27C13 45 5 37 5 24V10z"
            fill="{SURFACE}" stroke="url(#cyber-shield-gradient)" stroke-width="2.5" />
          <path d="m15 27 6 6 13-14" fill="none" stroke="{PURPLE_GLOW}"
            stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        """
    ).strip()
    _render_html(
        f"""
        <div class="cyber-hero">
          <div class="cyber-brand-lockup">
            {shield}
            <div>
              <div class="cyber-wordmark" role="heading" aria-level="1">CYBERSIFT</div>
              <div class="cyber-tagline">{html.escape(APP_TAGLINE)}</div>
            </div>
          </div>
          <div class="cyber-online-chip">
            <span class="cyber-status-dot"></span>
            <span>SYSTEM ONLINE · MODEL READY</span>
          </div>
        </div>
        <div class="cyber-neon-line"></div>
        """
    )


def render_sidebar_logo():
    """Show a compact CyberSift brand mark in the sidebar."""
    _render_html(
        f"""
        <div class="cyber-sidebar-lockup">
          <svg class="cyber-shield" viewBox="0 0 48 54" aria-hidden="true">
            <path d="M24 3 43 10v14c0 13-8 21-19 27C13 45 5 37 5 24V10z"
              fill="{SURFACE}" stroke="{PURPLE_GLOW}" stroke-width="2.5" />
            <path d="m15 27 6 6 13-14" fill="none" stroke="{PURPLE_GLOW}"
              stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <div class="cyber-wordmark">CYBERSIFT</div>
        </div>
        """
    )


def render_system_status(artifact):
    """Render model details as compact terminal-style status lines."""
    features = len(artifact.get("features", []))
    training_rows = artifact.get("n_train_rows")
    training_text = f"{int(training_rows):,}" if training_rows is not None else "unknown"
    threshold = float(artifact.get("threshold", 0.5))
    _render_html(
        f"""
        <div class="cyber-status-panel">
          <div class="cyber-status-panel__title">SYSTEM STATUS</div>
          <div>&gt; model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RANDOM FOREST / READY</div>
          <div>&gt; features&nbsp;&nbsp;&nbsp; {features}</div>
          <div>&gt; train rows&nbsp; {html.escape(training_text)}</div>
          <div>&gt; threshold&nbsp; {threshold:.3f}</div>
        </div>
        """
    )


def kpi_card(label, value, caption, tone="normal"):
    """Show one compact KPI card in the normal or threat color treatment."""
    tone_class = "cyber-kpi-card--threat" if tone == "threat" else ""
    _render_html(
        f"""
        <div class="cyber-kpi-card {tone_class}">
          <div class="cyber-kpi-label">{html.escape(str(label))}</div>
          <div class="cyber-kpi-value">{html.escape(str(value))}</div>
          <div class="cyber-kpi-caption">{html.escape(str(caption))}</div>
        </div>
        """
    )


def verdict_badge(flagged):
    """Show a purple normal badge or a red suspicious badge."""
    label = "SUSPICIOUS" if flagged else "NORMAL"
    tone_class = "cyber-verdict-badge--threat" if flagged else ""
    _render_html(
        f'<div class="cyber-verdict-badge {tone_class}">{label}</div>'
    )


def render_footer():
    """Add the project and course line after the app tabs."""
    _render_html(
        f'<footer class="cyber-footer">'
        f'<span class="cyber-footer__project">{html.escape(APP_PAGE_TITLE)}</span>'
        f'<span>·</span><span>{html.escape(COURSE_LABEL)}</span>'
        "</footer>"
    )


def threat_banner(level, flagged_pct, flagged_count):
    """Describe the current alert volume with an appropriate visual level."""
    safe_level = str(level).upper()
    if safe_level not in {"NORMAL", "ELEVATED", "CRITICAL"}:
        safe_level = "NORMAL"
    message = (
        f"{float(flagged_pct):.1f}% of traffic is flagged: "
        f"{int(flagged_count):,} flows need analyst review"
    )
    _render_html(
        f"""
        <div class="cyber-threat-banner cyber-threat-banner--{safe_level.lower()}">
          <span class="cyber-threat-dot"></span>
          <span class="cyber-threat-banner__level">{safe_level}</span>
          <span class="cyber-threat-banner__message">{html.escape(message)}</span>
        </div>
        """
    )


def section_header(title, subtitle, help_text=None):
    """Show a heading, plain-English summary, and optional native tooltip."""
    st.subheader(title, help=help_text)
    st.caption(subtitle)


def callout(kind, text):
    """Show a quiet, accessible message for empty states and input errors."""
    safe_kind = str(kind).lower()
    if safe_kind not in {"info", "success", "warning", "error"}:
        safe_kind = "info"
    labels = {
        "info": "NOTE",
        "success": "CLEAR",
        "warning": "CHECK",
        "error": "ERROR",
    }
    role = "alert" if safe_kind == "error" else "status"
    _render_html(
        f"""
        <div class="cyber-callout cyber-callout--{safe_kind}" role="{role}">
          <span class="cyber-callout__label">{labels[safe_kind]}</span>
          <span>{html.escape(str(text))}</span>
        </div>
        """
    )


def style_figure(figure):
    """Apply the shared transparent Plotly surface and subdued grid lines."""
    figure.update_layout(
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(family=FONT_MONO, color=TEXT, size=14),
        title_font=dict(family=FONT_HEADING, color=TEXT, size=17),
        legend=dict(bgcolor=TRANSPARENT, font=dict(color=MUTED)),
        margin=dict(l=14, r=14, t=52, b=14),
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=PURPLE_GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(family=FONT_MONO, color=MUTED, size=12),
        title_font=dict(family=FONT_BODY, color=MUTED, size=14),
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=PURPLE_GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(family=FONT_MONO, color=MUTED, size=12),
        title_font=dict(family=FONT_BODY, color=MUTED, size=14),
    )
    return figure
