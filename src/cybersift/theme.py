"""CyberSift theme colors, typography, and stable Streamlit styling."""

import streamlit as st


BG = "#0A0612"
SURFACE = "#140B26"
CARD = "#1B1033"
BORDER = "#3B2468"
PURPLE = "#8B5CF6"
PURPLE_GLOW = "#A78BFA"
TEXT = "#EEE9FF"
MUTED = "#A396C9"
THREAT_RED = "#FF3B5C"
DEEP_RED = "#B3123A"
ONLINE_GREEN = "#38D17A"
BUTTON_GRADIENT_END = "#6037B8"

TRANSPARENT = "rgba(0, 0, 0, 0)"
PURPLE_WASH = "rgba(139, 92, 246, 0.13)"
PURPLE_GRID = "rgba(167, 139, 250, 0.045)"
SCANLINE = "rgba(238, 233, 255, 0.018)"
PURPLE_BORDER_SOFT = "rgba(167, 139, 250, 0.34)"
PURPLE_SHADOW = "rgba(139, 92, 246, 0.2)"
THREAT_SHADOW = "rgba(255, 59, 92, 0.17)"
SURFACE_WASH = "rgba(20, 11, 38, 0.72)"
CARD_HIGHLIGHT = "rgba(238, 233, 255, 0.035)"
THREAT_CARD_HIGHLIGHT = "rgba(255, 59, 92, 0.08)"
NORMAL_BANNER_WASH = "rgba(139, 92, 246, 0.1)"
THREAT_BANNER_WASH = "rgba(255, 59, 92, 0.12)"
RISK_HISTOGRAM_BINS = 40

FONT_HEADING = '"Rajdhani", "Arial Narrow", sans-serif'
FONT_BODY = '"Inter", "Segoe UI", sans-serif'
FONT_MONO = '"JetBrains Mono", "Consolas", monospace'
GOOGLE_FONTS_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;600&"
    "family=Rajdhani:wght@500;600;700&display=swap"
)


_CSS_TEMPLATE = r"""
@import url('__GOOGLE_FONTS_IMPORT__');

:root {
  color-scheme: dark;
  --cyber-bg: __BG__;
  --cyber-surface: __SURFACE__;
  --cyber-card: __CARD__;
  --cyber-border: __BORDER__;
  --cyber-purple: __PURPLE__;
  --cyber-purple-glow: __PURPLE_GLOW__;
  --cyber-text: __TEXT__;
  --cyber-muted: __MUTED__;
  --cyber-threat: __THREAT_RED__;
  --cyber-deep-red: __DEEP_RED__;
  --cyber-online: __ONLINE_GREEN__;
  --cyber-purple-wash: __PURPLE_WASH__;
  --cyber-grid: __PURPLE_GRID__;
  --cyber-scanline: __SCANLINE__;
  --cyber-purple-border-soft: __PURPLE_BORDER_SOFT__;
  --cyber-purple-shadow: __PURPLE_SHADOW__;
  --cyber-threat-shadow: __THREAT_SHADOW__;
  --cyber-font-heading: __FONT_HEADING__;
  --cyber-font-body: __FONT_BODY__;
  --cyber-font-mono: __FONT_MONO__;
}

[data-testid="stAppViewContainer"] {
  min-height: 100vh;
  color: var(--cyber-text);
  font-family: var(--cyber-font-body);
  background-color: var(--cyber-bg);
  background-image:
    radial-gradient(900px circle at 3% 0%, var(--cyber-purple-wash), transparent 56%),
    linear-gradient(var(--cyber-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--cyber-grid) 1px, transparent 1px),
    repeating-linear-gradient(0deg, var(--cyber-scanline) 0, var(--cyber-scanline) 1px, transparent 1px, transparent 4px);
  background-size: auto, 32px 32px, 32px 32px, 100% 4px;
  background-attachment: fixed;
}

[data-testid="stMain"] {
  background: transparent;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--cyber-surface), var(--cyber-bg));
  border-right: 1px solid var(--cyber-border);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stFooter"] {
  visibility: hidden;
  height: 0;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
  color: var(--cyber-text);
  font-family: var(--cyber-font-heading);
  letter-spacing: 0.025em;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
  color: var(--cyber-text);
  font-family: var(--cyber-font-body);
  font-size: 15px;
  line-height: 1.45;
}

[data-testid="stAppViewContainer"] code,
[data-testid="stAppViewContainer"] pre {
  font-family: var(--cyber-font-mono);
}

[data-testid="stTabs"] [data-testid="stTab"] {
  min-height: 44px;
  padding: 0 16px;
  border: 1px solid transparent;
  border-radius: 999px;
  color: var(--cyber-muted);
  font-family: var(--cyber-font-heading);
  font-size: 17px;
  font-weight: 600;
}

[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  border-color: var(--cyber-purple-border-soft);
  color: var(--cyber-text);
  background: linear-gradient(135deg, var(--cyber-card), var(--cyber-surface));
  box-shadow: 0 0 18px var(--cyber-purple-shadow);
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  min-height: 42px;
  border: 1px solid var(--cyber-purple-glow);
  border-radius: 10px;
  color: var(--cyber-text);
  background: linear-gradient(110deg, var(--cyber-purple), __BUTTON_GRADIENT_END__);
  font-family: var(--cyber-font-heading);
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.025em;
  box-shadow: 0 0 12px var(--cyber-purple-shadow);
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--cyber-purple-glow);
  box-shadow: 0 0 22px var(--cyber-purple-shadow);
  filter: brightness(1.1);
}

[data-testid="stSlider"] {
  padding: 8px 10px;
  border: 1px solid var(--cyber-border);
  border-radius: 12px;
  background: var(--cyber-card);
  accent-color: var(--cyber-purple);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--cyber-border);
  border-radius: 12px;
}

[data-testid="stExpander"] {
  border: 1px solid var(--cyber-border);
  border-radius: 12px;
  background: __SURFACE_WASH__;
}

[data-testid="stAppViewContainer"]::-webkit-scrollbar,
[data-testid="stSidebar"]::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

[data-testid="stAppViewContainer"]::-webkit-scrollbar-track,
[data-testid="stSidebar"]::-webkit-scrollbar-track {
  background: var(--cyber-bg);
}

[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb,
[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
  border: 2px solid var(--cyber-bg);
  border-radius: 999px;
  background: var(--cyber-border);
}

.cyber-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 8px 0 18px;
}

.cyber-brand-lockup,
.cyber-sidebar-lockup {
  display: flex;
  align-items: center;
  gap: 14px;
}

.cyber-shield {
  width: 48px;
  height: 52px;
  flex: 0 0 auto;
  filter: drop-shadow(0 0 10px var(--cyber-purple-shadow));
}

.cyber-wordmark {
  color: var(--cyber-purple-glow);
  background: linear-gradient(100deg, var(--cyber-text), var(--cyber-purple-glow) 55%, var(--cyber-purple));
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-family: var(--cyber-font-heading);
  font-size: 34px;
  font-weight: 700;
  letter-spacing: 0.16em;
  line-height: 1;
}

.cyber-tagline {
  margin-top: 5px;
  color: var(--cyber-muted);
  font-family: var(--cyber-font-body);
  font-size: 15px;
}

.cyber-online-chip {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  padding: 9px 13px;
  border: 1px solid var(--cyber-border);
  border-radius: 999px;
  color: var(--cyber-muted);
  background: var(--cyber-surface);
  font-family: var(--cyber-font-mono);
  font-size: 12px;
  letter-spacing: 0.045em;
  white-space: nowrap;
}

.cyber-status-dot,
.cyber-threat-dot {
  width: 8px;
  height: 8px;
  flex: 0 0 auto;
  border-radius: 50%;
}

.cyber-status-dot {
  background: var(--cyber-online);
  box-shadow: 0 0 9px var(--cyber-online);
  animation: cyber-status-pulse 2.2s ease-in-out infinite;
}

.cyber-neon-line {
  height: 2px;
  margin-bottom: 20px;
  background: linear-gradient(90deg, var(--cyber-purple), var(--cyber-purple-glow) 65%, var(--cyber-threat));
  box-shadow: 0 0 12px var(--cyber-purple-shadow);
}

.cyber-sidebar-lockup {
  padding: 6px 0 18px;
}

.cyber-sidebar-lockup .cyber-shield {
  width: 32px;
  height: 36px;
}

.cyber-sidebar-lockup .cyber-wordmark {
  font-size: 24px;
}

.cyber-group-label {
  margin: 17px 0 8px;
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.12em;
}

.cyber-kpi-card {
  min-height: 134px;
  padding: 15px 16px;
  border: 1px solid var(--cyber-border);
  border-radius: 14px;
  background: linear-gradient(150deg, var(--cyber-card), var(--cyber-surface));
  box-shadow: inset 0 1px 0 __CARD_HIGHLIGHT__;
  transition: transform 180ms ease, box-shadow 180ms ease;
}

.cyber-kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px var(--cyber-purple-shadow);
}

.cyber-kpi-card--threat {
  border-color: var(--cyber-threat);
  box-shadow: inset 0 1px 0 __THREAT_CARD_HIGHLIGHT__, 0 0 18px var(--cyber-threat-shadow);
}

.cyber-kpi-label {
  color: var(--cyber-muted);
  font-family: var(--cyber-font-body);
  font-size: 15px;
  font-weight: 600;
  white-space: nowrap;
}

.cyber-kpi-value {
  overflow: hidden;
  margin-top: 8px;
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-mono);
  font-size: clamp(20px, 2vw, 27px);
  font-weight: 600;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cyber-kpi-card--threat .cyber-kpi-value {
  color: var(--cyber-threat);
}

.cyber-kpi-caption {
  margin-top: 7px;
  color: var(--cyber-muted);
  font-family: var(--cyber-font-body);
  font-size: 15px;
  line-height: 1.25;
  white-space: nowrap;
}

.cyber-threat-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 54px;
  margin: 6px 0 14px;
  padding: 11px 15px;
  border: 1px solid var(--cyber-border);
  border-left: 4px solid var(--cyber-purple);
  border-radius: 12px;
  color: var(--cyber-text);
  background: linear-gradient(90deg, __NORMAL_BANNER_WASH__, var(--cyber-surface));
  font-size: 15px;
}

.cyber-threat-banner--elevated,
.cyber-threat-banner--critical {
  border-color: var(--cyber-deep-red);
  border-left-color: var(--cyber-threat);
  background: linear-gradient(90deg, __THREAT_BANNER_WASH__, var(--cyber-surface));
}

.cyber-threat-banner__level {
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.08em;
  white-space: nowrap;
}

.cyber-threat-banner--elevated .cyber-threat-banner__level,
.cyber-threat-banner--critical .cyber-threat-banner__level {
  color: var(--cyber-threat);
}

.cyber-threat-dot {
  background: var(--cyber-purple-glow);
  box-shadow: 0 0 8px var(--cyber-purple-shadow);
}

.cyber-threat-banner--elevated .cyber-threat-dot,
.cyber-threat-banner--critical .cyber-threat-dot {
  background: var(--cyber-threat);
  box-shadow: 0 0 8px var(--cyber-threat-shadow);
}

.cyber-threat-banner--critical .cyber-threat-dot {
  animation: cyber-critical-pulse 1.25s ease-in-out infinite;
}

.cyber-threat-banner__message {
  font-family: var(--cyber-font-body);
  font-size: 15px;
}

.cyber-section-heading {
  margin: 18px 0 12px;
}

.cyber-section-heading h2 {
  margin: 0;
  color: var(--cyber-text);
  font-family: var(--cyber-font-heading);
  font-size: 25px;
  font-weight: 700;
}

.cyber-section-heading p {
  margin: 2px 0 0;
  color: var(--cyber-muted);
  font-size: 15px;
}

.cyber-callout {
  margin: 8px 0;
  padding: 12px 15px;
  border: 1px solid var(--cyber-border);
  border-left: 4px solid var(--cyber-purple);
  border-radius: 10px;
  color: var(--cyber-text);
  background: var(--cyber-surface);
  font-size: 15px;
  line-height: 1.45;
}

.cyber-callout--warning {
  border-left-color: var(--cyber-purple-glow);
}

.cyber-callout__label {
  margin-right: 8px;
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.cyber-status-panel {
  margin-top: 15px;
  padding: 12px;
  border: 1px solid var(--cyber-border);
  border-radius: 12px;
  background: var(--cyber-card);
  color: var(--cyber-muted);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
  line-height: 1.8;
}

.cyber-status-panel__title {
  margin-bottom: 5px;
  color: var(--cyber-purple-glow);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.1em;
}

.cyber-priority-card {
  min-height: 292px;
  padding: 14px;
  border: 1px solid var(--cyber-deep-red);
  border-left: 4px solid var(--cyber-threat);
  border-radius: 13px;
  color: var(--cyber-text);
  background: linear-gradient(150deg, var(--cyber-card), var(--cyber-surface));
  box-shadow: 0 0 16px var(--cyber-threat-shadow);
}

.cyber-priority-card__top,
.cyber-priority-card__score-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.cyber-rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: 1px solid var(--cyber-purple-border-soft);
  border-radius: 9px;
  color: var(--cyber-purple-glow);
  background: var(--cyber-surface);
  font-family: var(--cyber-font-mono);
  font-size: 14px;
  font-weight: 600;
}

.cyber-severity-chip {
  padding: 4px 8px;
  border: 1px solid var(--cyber-deep-red);
  border-radius: 999px;
  color: var(--cyber-threat);
  background: var(--cyber-surface);
  font-family: var(--cyber-font-mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.cyber-severity-chip--critical {
  border-color: var(--cyber-threat);
  color: var(--cyber-text);
  background: var(--cyber-deep-red);
}

.cyber-priority-card__score {
  margin-top: 15px;
  color: var(--cyber-threat);
  font-family: var(--cyber-font-mono);
  font-size: 23px;
  font-weight: 600;
}

.cyber-priority-card__score-label,
.cyber-priority-card__facts,
.cyber-priority-card__why,
.cyber-priority-card__footer {
  color: var(--cyber-muted);
  font-family: var(--cyber-font-body);
  font-size: 15px;
  line-height: 1.4;
}

.cyber-priority-card__track {
  overflow: hidden;
  height: 7px;
  margin: 9px 0 13px;
  border-radius: 999px;
  background: var(--cyber-bg);
}

.cyber-priority-card__fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--cyber-purple), var(--cyber-threat));
}

.cyber-priority-card__service {
  margin: 3px 0 8px;
  color: var(--cyber-text);
  font-family: var(--cyber-font-heading);
  font-size: 20px;
  font-weight: 700;
}

.cyber-priority-card__facts {
  min-height: 43px;
  font-family: var(--cyber-font-mono);
  font-size: 12px;
}

.cyber-priority-card__why {
  overflow: hidden;
  margin-top: 8px;
  color: var(--cyber-text);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cyber-priority-card__why strong {
  color: var(--cyber-purple-glow);
}

.cyber-priority-card__footer {
  margin-top: 11px;
  padding-top: 9px;
  border-top: 1px solid var(--cyber-border);
  font-size: 13px;
}

.cyber-all-clear-card {
  padding: 22px;
  border: 1px solid var(--cyber-purple-border-soft);
  border-left: 4px solid var(--cyber-purple);
  border-radius: 13px;
  color: var(--cyber-text);
  background: linear-gradient(120deg, var(--cyber-card), var(--cyber-surface));
  box-shadow: 0 0 16px var(--cyber-purple-shadow);
  font-size: 16px;
}

.cyber-all-clear-card strong {
  margin-right: 8px;
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-heading);
  font-size: 20px;
}

.cyber-verdict-badge {
  display: inline-flex;
  align-items: center;
  padding: 9px 14px;
  border: 1px solid var(--cyber-purple-border-soft);
  border-radius: 999px;
  color: var(--cyber-purple-glow);
  background: var(--cyber-purple-wash);
  font-family: var(--cyber-font-mono);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.08em;
}

.cyber-verdict-badge--threat {
  border-color: var(--cyber-threat);
  color: var(--cyber-text);
  background: var(--cyber-deep-red);
  box-shadow: 0 0 14px var(--cyber-threat-shadow);
}

.cyber-alert-log {
  overflow: hidden;
  border: 1px solid var(--cyber-border);
  border-radius: 13px;
  background: var(--cyber-bg);
  box-shadow: inset 0 0 24px var(--cyber-surface);
}

.cyber-terminal-header {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 11px 14px;
  border-bottom: 1px solid var(--cyber-border);
  color: var(--cyber-muted);
  font-family: var(--cyber-font-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
}

.cyber-terminal-dots {
  display: inline-flex;
  gap: 5px;
  margin-right: 7px;
}

.cyber-terminal-dots span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--cyber-border);
}

.cyber-terminal-dots span:first-child {
  background: var(--cyber-purple-glow);
}

.cyber-terminal-dots span:nth-child(2) {
  background: var(--cyber-purple);
}

.cyber-terminal-body {
  overflow-x: auto;
  padding: 12px 14px 14px;
}

.cyber-terminal-line {
  min-height: 25px;
  color: var(--cyber-text);
  font-family: var(--cyber-font-mono);
  font-size: 12px;
  line-height: 1.7;
  white-space: nowrap;
}

.cyber-terminal-label {
  color: var(--cyber-muted);
}

.cyber-terminal-value {
  color: var(--cyber-text);
}

.cyber-terminal-alert {
  color: var(--cyber-threat);
  font-weight: 700;
}

.cyber-terminal-empty {
  color: var(--cyber-muted);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
}

.cyber-terminal-cursor {
  margin-top: 3px;
  color: var(--cyber-purple-glow);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
}

.cyber-terminal-cursor__underscore {
  animation: cyber-cursor-blink 1s steps(2, start) infinite;
}

.cyber-footer {
  display: flex;
  justify-content: center;
  gap: 9px;
  margin-top: 34px;
  padding: 16px 8px;
  border-top: 1px solid var(--cyber-border);
  color: var(--cyber-muted);
  font-family: var(--cyber-font-mono);
  font-size: 13px;
  text-align: center;
}

.cyber-footer__project {
  color: var(--cyber-purple-glow);
  font-weight: 600;
}

@keyframes cyber-cursor-blink {
  to { visibility: hidden; }
}

@keyframes cyber-status-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--cyber-online); }
  50% { opacity: 0.62; box-shadow: 0 0 3px var(--cyber-online); }
}

@keyframes cyber-critical-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--cyber-threat-shadow); }
  50% { opacity: 0.5; box-shadow: 0 0 2px var(--cyber-threat-shadow); }
}

@media (prefers-reduced-motion: reduce) {
  .cyber-status-dot,
  .cyber-threat-banner--critical .cyber-threat-dot {
    animation: none !important;
  }
  .cyber-terminal-cursor__underscore {
    animation: none !important;
  }
  .cyber-kpi-card {
    transition: none !important;
  }
  .cyber-kpi-card:hover {
    transform: none;
  }
}

@media (max-width: 760px) {
  .cyber-hero {
    align-items: flex-start;
    flex-direction: column;
    gap: 12px;
  }
  .cyber-wordmark {
    font-size: 29px;
  }
  .cyber-threat-banner {
    align-items: flex-start;
    flex-direction: column;
    gap: 6px;
  }
  .cyber-kpi-card {
    min-height: 122px;
    padding: 12px;
  }
  .cyber-kpi-caption {
    white-space: normal;
  }
}
"""


def inject_css():
    """Inject the app's shared visual theme once per Streamlit run."""
    tokens = {
        "__GOOGLE_FONTS_IMPORT__": GOOGLE_FONTS_IMPORT,
        "__BG__": BG,
        "__SURFACE__": SURFACE,
        "__CARD__": CARD,
        "__BORDER__": BORDER,
        "__PURPLE__": PURPLE,
        "__PURPLE_GLOW__": PURPLE_GLOW,
        "__TEXT__": TEXT,
        "__MUTED__": MUTED,
        "__THREAT_RED__": THREAT_RED,
        "__DEEP_RED__": DEEP_RED,
        "__ONLINE_GREEN__": ONLINE_GREEN,
        "__PURPLE_WASH__": PURPLE_WASH,
        "__PURPLE_GRID__": PURPLE_GRID,
        "__SCANLINE__": SCANLINE,
        "__PURPLE_BORDER_SOFT__": PURPLE_BORDER_SOFT,
        "__PURPLE_SHADOW__": PURPLE_SHADOW,
        "__THREAT_SHADOW__": THREAT_SHADOW,
        "__BUTTON_GRADIENT_END__": BUTTON_GRADIENT_END,
        "__SURFACE_WASH__": SURFACE_WASH,
        "__CARD_HIGHLIGHT__": CARD_HIGHLIGHT,
        "__THREAT_CARD_HIGHLIGHT__": THREAT_CARD_HIGHLIGHT,
        "__NORMAL_BANNER_WASH__": NORMAL_BANNER_WASH,
        "__THREAT_BANNER_WASH__": THREAT_BANNER_WASH,
        "__FONT_HEADING__": FONT_HEADING,
        "__FONT_BODY__": FONT_BODY,
        "__FONT_MONO__": FONT_MONO,
    }
    css = _CSS_TEMPLATE
    for token, value in tokens.items():
        css = css.replace(token, value)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
