"""
Streamlit V1 — CV Evaluator & Classifier

Upload a CV (PDF/DOCX/TXT) → parse → extract → score (rubric)
→ classify (ML: TF-IDF + XGBoost) → suggest improvements
→ Compare rubric vs ML labels

Usage:
    streamlit run app/app.py
"""

import json
import os
import sys
import warnings
import tempfile

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CV Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
LR_PATH = os.path.join(MODELS_DIR, "lr_baseline.pkl")

LABEL_COLORS = {
    "Strong": "#15803d",
    "Average": "#b45309",
    "Weak": "#b91c1c",
}

LABEL_EMOJIS = {"Strong": "🟢", "Average": "🟠", "Weak": "🔴"}


@st.cache_resource
def load_classifier():
    import joblib
    if os.path.exists(XGB_PATH):
        return joblib.load(XGB_PATH), "XGBoost"
    elif os.path.exists(LR_PATH):
        return joblib.load(LR_PATH), "Logistic Regression"
    return None, None


@st.cache_resource
def ensure_spacy_model():
    """Ensure en_core_web_sm is available — download if missing."""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            with st.spinner("Downloading spaCy model (first run only)..."):
                from spacy.cli.download import download
                download("en_core_web_sm")
                spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"spaCy model error: {e}")
        return False
    return True


@st.cache_resource
def get_parser_extractor_scorer():
    if not ensure_spacy_model():
        st.stop()
    from src.parser.parser import parse_cv
    from src.parser.section_splitter import split_sections
    from src.extractor.extractor import extract_all
    from src.scorer.scorer import score_cv
    from src.suggester.suggester import generate_suggestions
    return parse_cv, split_sections, extract_all, score_cv, generate_suggestions


@st.cache_resource
def load_rubric_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "rubric_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def classify_text(model, text):
    prediction = model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        return prediction, proba
    return prediction, None


def render_metric_card(title, value, subtitle, color):
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:0.75rem; padding:1.25rem;
                    text-align:center; background:white;">
            <div style="font-size:0.8rem; color:#6b7280; margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:2.2rem; font-weight:700; color:{color};">{value}</div>
            <div style="font-size:0.85rem; color:#6b7280; margin-top:0.25rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True
    )


# ══════════════════════════════════════
st.title("📄 CV Evaluator & Quality Classifier")
st.markdown(
    "Upload a CV to extract information, score against a rubric, "
    "and classify quality using both **rule-based scoring** and **ML text classification**."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Techniques:**
        - **NER** — spaCy EntityRuler + PhraseMatcher
        - **Info Extraction** — rule-based section parsers
        - **Keyword Extraction** — skill taxonomy
        - **Text Classification** — TF-IDF + XGBoost
        - **Semantic Similarity** — *(V2)*
        """
    )
    st.divider()
    st.caption("CV Evaluator v1.0")

    model_pipeline, model_name = load_classifier()
    if model_pipeline:
        st.success(f"ML model: **{model_name}**")
    else:
        st.warning("No ML model found. Train with `scripts/vectorize_cvs.py` first.")

    rubric_config = load_rubric_config()
    with st.expander("⚙️ Rubric Weights", expanded=False):
        for section, cfg in rubric_config.items():
            if isinstance(cfg, dict) and "max_points" in cfg:
                st.caption(f"{section}: {cfg['max_points']} pts")

uploaded_file = st.file_uploader(
    "Upload CV",
    type=["pdf", "docx", "txt"],
    help="PDF, DOCX, or plain text"
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Parsing, extracting, scoring..."):
        parse_cv, split_sections, extract_all, score_cv, generate_suggestions = (
            get_parser_extractor_scorer()
        )

        raw_text = parse_cv(tmp_path)
        if not raw_text or len(raw_text.strip()) < 20:
            st.error("Could not extract text. File may be an image-based PDF.")
            os.unlink(tmp_path)
            st.stop()

        sections = split_sections(raw_text)
        cv = extract_all(raw_text, sections=sections)
        if not cv:
            st.error("Extraction failed.")
            os.unlink(tmp_path)
            st.stop()

        cv = score_cv(cv)
        suggestions = generate_suggestions(cv)

        ml_label = None
        ml_proba = None
        if model_pipeline:
            ml_label, ml_proba = classify_text(model_pipeline, raw_text)

    os.unlink(tmp_path)

    total_score = cv.get("total_score", 0)
    rubric_label = cv.get("label", "Unknown")

    st.divider()
    st.subheader("📊 Results")

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        render_metric_card("RUBRIC SCORE",
                           f"{total_score:.0f}/100",
                           f"Label: {rubric_label}",
                           LABEL_COLORS.get(rubric_label, "#888"))
    with kpi_cols[1]:
        if ml_label:
            ml_color = LABEL_COLORS.get(ml_label, "#888")
            conf_str = ""
            if ml_proba is not None:
                idx = {"Average": 0, "Strong": 1, "Weak": 2}.get(ml_label, 0)
                conf_str = f"Confidence: {ml_proba[idx]:.1%}"
            render_metric_card("ML CLASSIFICATION", ml_label, conf_str, ml_color)
        else:
            render_metric_card("ML CLASSIFICATION", "—", "No model", "#888")
    with kpi_cols[2]:
        if ml_label:
            if rubric_label == ml_label:
                st.markdown(
                    "<div style='border:1px solid #15803d; border-radius:0.75rem; "
                    "padding:1.25rem; text-align:center; background:#f0fdf4;'>"
                    "<div style='font-size:0.8rem; color:#6b7280;'>AGREEMENT</div>"
                    "<div style='font-size:2.2rem;'>✅</div>"
                    "<div style='font-size:0.85rem; color:#15803d;'>Rubric & ML agree</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='border:1px solid #b45309; border-radius:0.75rem; "
                    f"padding:1.25rem; text-align:center; background:#fffbeb;'>"
                    f"<div style='font-size:0.8rem; color:#6b7280;'>AGREEMENT</div>"
                    f"<div style='font-size:2.2rem;'>⚠️</div>"
                    f"<div style='font-size:0.85rem; color:#b45309;'>Rubric: {rubric_label}<br>ML: {ml_label}</div></div>",
                    unsafe_allow_html=True
                )
        else:
            render_metric_card("AGREEMENT", "—", "No ML model", "#888")

    # Section breakdown with progress bars
    st.divider()
    st.subheader("📈 Section Scores")
    section_scores = cv.get("section_scores", {})
    score_data = []
    for section, score in section_scores.items():
        cfg = rubric_config.get(section, {})
        max_pts = cfg.get("max_points", 100) if isinstance(cfg, dict) else 100
        pct = score / max_pts * 100 if max_pts > 0 else 0
        score_data.append({"Section": section.capitalize(), "Score": score,
                           "Max": max_pts, "Pct": pct})

    if score_data:
        df_scores = pd.DataFrame(score_data)
        for _, row in df_scores.iterrows():
            cols = st.columns([2, 6, 1])
            cols[0].markdown(f"**{row['Section']}**")
            cols[1].progress(row["Pct"] / 100, text=" ")
            cols[2].markdown(f"**{row['Score']:.0f}**/{row['Max']:.0f}")

    # Candidate details
    st.divider()
    tabs = st.tabs(["📋 Extracted Data", "💡 Suggestions", "📝 Raw Text"])

    with tabs[0]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Name:** {cv.get('name', 'N/A')}")
            st.markdown(f"**Email:** {cv.get('email', 'N/A')}")
            st.markdown(f"**Phone:** {cv.get('phone', 'N/A')}")
            st.markdown(f"**Skills ({len(cv.get('skills', []))}):** "
                        f"{', '.join(cv.get('skills', [])[:10])}"
                        f"{'...' if len(cv.get('skills', [])) > 10 else ''}")
        with col_b:
            st.markdown(f"**Experience:** {len(cv.get('experience', []))} entries")
            for exp in cv.get("experience", [])[:3]:
                title = exp.get("title", "?")
                company = exp.get("company", "?")
                st.caption(f"  {title} @ {company}")
            st.markdown(f"**Education:** {len(cv.get('education', []))} entries")
            for edu in cv.get("education", [])[:2]:
                deg = edu.get("degree", "?")
                inst = edu.get("institution", "?")
                st.caption(f"  {deg} @ {inst}")
            st.markdown(f"**Projects:** {len(cv.get('projects', []))}")
            st.markdown(f"**Certifications:** {len(cv.get('certifications', []))}")

    with tabs[1]:
        if suggestions:
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.info("No suggestions needed.")

    with tabs[2]:
        st.text_area("Extracted text", raw_text, height=250, label_visibility="collapsed")

    # Download
    st.divider()
    cv_json = json.dumps(cv, indent=2, default=str)
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=cv_json,
        file_name=f"{uploaded_file.name}_analysis.json",
        mime="application/json",
    )

else:
    st.info("Upload a CV to begin analysis.")
    st.markdown(
        """
        ---
        **Pipeline:**
        1. **Parse** — extract text from PDF/DOCX/TXT
        2. **Extract** — NER + rule-based extraction of skills, experience, education, projects
        3. **Score** — weighted rubric (0-100) → label (Strong/Average/Weak)
        4. **Classify** — TF-IDF + XGBoost on raw text (ML-based label)
        5. **Suggest** — targeted improvement tips per section
        """
    )
