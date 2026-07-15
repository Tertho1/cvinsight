"""Entry point for Streamlit Cloud — delegates to app/app.py with error handling"""
import sys
import traceback

try:
    from app.app import *
except Exception:
    import streamlit as st
    st.set_page_config(page_title="CV Evaluator", page_icon="📄")
    st.error("Failed to load application. Check the error below.")
    st.code(traceback.format_exc(), language="python")
