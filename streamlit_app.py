"""Entry point for Streamlit Cloud — delegates to app/app.py with error handling"""
import sys
import traceback
import streamlit as st

st.set_page_config(
    page_title="CV Evaluator",
    page_icon="\U0001F4C4",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import app.app
    app.app.main()
except Exception:
    st.error("Failed to load application. Check the error below.")
    st.code(traceback.format_exc(), language="python")
