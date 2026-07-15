"""Entry point for Hugging Face Spaces / Streamlit Cloud — delegates to app/app.py"""
import streamlit as st
import sys
import traceback

try:
    from app.app import *
except Exception as e:
    st.error(f"Failed to load app: {e}")
    st.code(traceback.format_exc(), language="python")
    st.stop()
