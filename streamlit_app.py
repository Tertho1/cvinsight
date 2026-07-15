import streamlit as st

st.set_page_config(page_title="CV Evaluator", page_icon="📄")

st.title("📄 CV Evaluator")
st.info("Testing deployment...")

if st.button("Test Import"):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("John Smith is a software engineer at Google.")
    st.success(f"spaCy works! Entities: {[(e.text, e.label_) for e in doc.ents]}")

st.write(f"Python: {__import__('sys').version}")
