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
    APP_TAGLINE,
    DEMO_SAMPLE_PATH,
)
from cybersift.predict import load_artifact  # noqa: E402
from cybersift.ui_tabs import render_sidebar, render_threat_desk, render_coming_soon  # noqa: E402


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
    st.title(APP_PAGE_TITLE)
    st.caption(APP_TAGLINE)

    try:
        artifact = get_artifact()
    except Exception as error:
        st.error(str(error))
        st.stop()

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

    for tab in tabs[1:]:
        with tab:
            render_coming_soon()


if __name__ == "__main__":
    main()
