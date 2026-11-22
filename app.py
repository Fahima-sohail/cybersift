"""CyberSift Streamlit application entry point."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cybersift.config import (  # noqa: E402
    APP_LAYOUT,
    APP_PAGE_TITLE,
    APP_TABS,
    DEMO_SAMPLE_PATH,
    RF_MODEL_PATH,
)
from cybersift.predict import load_artifact  # noqa: E402
from cybersift.ui_tabs import (  # noqa: E402
    render_how_it_works,
    render_inspect_flow,
    render_live_monitor,
    render_model_report_card,
    render_sidebar,
    render_threat_desk,
)
from cybersift.theme import inject_css  # noqa: E402
from cybersift.ui_components import render_footer, render_hero  # noqa: E402


@st.cache_resource
def get_artifact():
    """Load the saved pipeline once per app process."""
    return load_artifact()


@st.cache_data
def get_demo_sample():
    """Load the small labeled demo sample for the Threat Desk."""
    return pd.read_csv(DEMO_SAMPLE_PATH)


def main():
    st.set_page_config(page_title=APP_PAGE_TITLE, layout=APP_LAYOUT)
    inject_css()

    if not RF_MODEL_PATH.is_file():
        st.error(
            "The saved Random Forest model is missing. Include "
            "models/cybersift_rf.joblib in the deployment."
        )
        st.stop()

    if not DEMO_SAMPLE_PATH.is_file():
        st.error(
            "The CyberSift demo sample is missing. Include "
            "data/sample/cybersift_demo.csv in the deployment."
        )
        st.stop()

    try:
        artifact = get_artifact()
    except Exception as error:
        st.error(f"CyberSift could not load its saved model: {error}")
        st.stop()

    render_hero()
    threshold, data_source, uploaded_file = render_sidebar(artifact)
    tabs = st.tabs(APP_TABS)

    with tabs[0]:
        render_threat_desk(
            artifact=artifact,
            threshold=threshold,
            data_source=data_source,
            uploaded_file=uploaded_file,
            demo_loader=get_demo_sample,
        )

    with tabs[1]:
        render_inspect_flow(artifact, threshold, get_demo_sample)

    with tabs[2]:
        render_live_monitor(artifact, threshold, get_demo_sample)

    with tabs[3]:
        render_model_report_card()

    with tabs[4]:
        render_how_it_works()

    render_footer()


if __name__ == "__main__":
    main()
