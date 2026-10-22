"""
WellScope: Wireline Log Interpretation & Integrated Formation Evaluation.
Main Streamlit application entry point.
"""

import os
import sys

# Ensure src/ is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import streamlit as st
import wellscope
from wellscope.ui import pages

def main():
    st.set_page_config(
        page_title="WellScope — Wireline Petrophysics",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    pages.init_session_state()

    # Sidebar navigation
    pages.render_sidebar()

    # Main page dispatcher
    pages.render_main_content()

if __name__ == "__main__":
    main()
