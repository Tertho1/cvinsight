"""
Streamlit V1 — CV Evaluator & Classifier

Upload a CV (PDF/DOCX/TXT) → parse → extract → score (rubric)
→ classify (ML: TF-IDF + XGBoost) → suggest improvements
→ Compare rubric vs ML labels

Features:
- Adjustable rubric weights per section
- Structured tables for experience/education/projects
- Processing stage indicators
- Session history (last 5 results)
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
LR_PATH = os.path.join(MODELS_DIR, "lr_baseline.pkl")
DEFAULT_RUBRIC_PATH = os.path.join(CONFIG_DIR, "rubric_config.json")
MAX_FILE_MB = 50
PARSE_TIMEOUT = 180

LABEL_COLORS = {
    "Strong": "#15803d",
    "Average": "#b45309",
    "Weak": "#b91c1c",
}


@st.cache_resource
def load_classifier():
    import joblib
    import warnings as _w
    if os.path.exists(XGB_PATH):
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                return joblib.load(XGB_PATH), "XGBoost"
        except Exception as e:
            st.warning(f"XGBoost pickle version mismatch ({e}), trying native format...")
            native_path = os.path.join(MODELS_DIR, "xgb_booster.model")
            vec_path = os.path.join(MODELS_DIR, "xgb_vectorizer.pkl")
            label_path = os.path.join(MODELS_DIR, "xgb_labels.json")
            if os.path.exists(native_path) and os.path.exists(vec_path):
                from sklearn.pipeline import Pipeline
                import xgboost as xgb
                vec = joblib.load(vec_path)
                booster = xgb.Booster(model_file=native_path)
                clf = xgb.XGBClassifier()
                clf._Booster = booster
                pipeline = Pipeline([("tfidf", vec), ("clf", clf)])
                if os.path.exists(label_path):
                    with open(label_path) as f:
                        pipeline.label_classes_ = json.load(f)
                else:
                    pipeline.label_classes_ = ["Average", "Strong", "Weak"]
                return pipeline, "XGBoost (native)"
    elif os.path.exists(LR_PATH):
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return joblib.load(LR_PATH), "Logistic Regression"
    return None, None


@st.cache_resource
def ensure_spacy_model():
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
    if os.path.exists(DEFAULT_RUBRIC_PATH):
        with open(DEFAULT_RUBRIC_PATH) as f:
            return json.load(f)
    return {}


def load_default_weights(config):
    weights = {}
    for section, cfg in config.items():
        if isinstance(cfg, dict) and "max_points" in cfg:
            weights[section] = cfg["max_points"]
    return weights


def classify_text(model, text):
    import numpy as np
    raw = model.predict([text])[0]
    if isinstance(raw, (int, float, np.integer, np.floating, type(None))):
        label_map = getattr(model, "label_classes_", ["Average", "Strong", "Weak"])
        label = label_map[int(raw)]
    else:
        label = str(raw)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
    return label, proba


def safe_parse_cv(path):
    script = textwrap.dedent(r"""
        import sys, warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, {root!r})
        from src.parser.parser import parse_cv
        try:
            text = parse_cv({path!r})
            sys.stdout.write(text)
        except Exception as e:
            sys.stderr.write(str(e))
            sys.exit(1)
    """).strip()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = script.format(root=root, path=path)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=PARSE_TIMEOUT
        )
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Parsing subprocess failed: {result.stderr}")
            return ""
    except subprocess.TimeoutExpired:
        st.error(f"Parsing timed out after {PARSE_TIMEOUT}s.")
        return ""
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        return ""


def make_custom_config(base_config, custom_weights):
    config = json.loads(json.dumps(base_config))
    for section, weight in custom_weights.items():
        if section in config and isinstance(config[section], dict):
            config[section]["max_points"] = weight
    total = sum(custom_weights.values())
    config["total_points"] = total
    return config


def render_section_bar(section_name, score, max_pts):
    pct = score / max_pts * 100 if max_pts > 0 else 0
    color = "#15803d" if pct >= 70 else "#b45309" if pct >= 40 else "#b91c1c"
    cols = st.columns([2, 6, 1])
    cols[0].markdown(f"**{section_name}**")
    cols[1].progress(pct / 100, text=" ")
    cols[2].markdown(
        f"<span style='color:{color};font-weight:600;'>{score:.0f}</span>/{max_pts:.0f}",
        unsafe_allow_html=True,
    )


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


def render_table(items, columns, caption):
    if not items:
        st.caption(f"No {caption} found.")
        return
    rows = []
    for item in items[:10]:
        row = {}
        for col_key, col_label in columns:
            row[col_label] = item.get(col_key, "")
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(items) > 10:
        st.caption(f"Showing 10 of {len(items)} {caption}")


def main():
    st.set_page_config(
        page_title="CV Evaluator",
        page_icon="\U0001F4C4",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("\U0001F4C4 CV Evaluator & Quality Classifier")
    st.markdown(
        "Upload a CV to extract information, score against **customizable rubric weights**, "
        "and classify quality using both rule-based scoring and ML text classification."
    )

    rubric_config = load_rubric_config()
    default_weights = load_default_weights(rubric_config)

    with st.sidebar:
        st.header("\u2699\uFE0F Customize Rubric")
        st.caption("Adjust section weights to match your hiring priorities. Total must not exceed 100.")

        custom_weights = {}
        total = 0
        for section in sorted(default_weights.keys()):
            label = section.replace("_", " ").title()
            default = default_weights[section]
            val = st.slider(
                label, min_value=0, max_value=50,
                value=default, key=f"w_{section}"
            )
            custom_weights[section] = val
            total += val

        st.caption(f"**Total: {total}/100**")
        if total != 100:
            st.warning(f"Weights sum to {total}, not 100. Scores will be relative.", icon="\u26A0\uFE0F")

        use_custom_weights = st.checkbox("Apply custom weights", value=(total != 100))

        st.divider()
        st.header("About")
        st.markdown(
            "**Techniques:**  \n"
            "- **NER** — spaCy EntityRuler + PhraseMatcher  \n"
            "- **Info Extraction** — rule-based section parsers  \n"
            "- **Keyword Extraction** — skill taxonomy  \n"
            "- **Text Classification** — TF-IDF + XGBoost  \n"
            "- **Semantic Similarity** — *(V2)*"
        )
        st.divider()
        model_pipeline, model_name = load_classifier()
        if model_pipeline:
            st.success(f"ML model: **{model_name}**")
        else:
            st.warning("No ML model found.")

        if st.button("\U0001F5D1 Clear History"):
            st.session_state.history = []
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "txt"],
        help="PDF, DOCX, or plain text"
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Max {MAX_FILE_MB} MB.")
            st.stop()

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        status = st.status("Processing CV...", expanded=True)

        with status:
            st.write("\U0001F4E5 Parsing file...")
            parse_cv, split_sections, extract_all, score_cv_fn, generate_suggestions = (
                get_parser_extractor_scorer()
            )

            raw_text = safe_parse_cv(tmp_path) if suffix.lower() == ".pdf" else parse_cv(tmp_path)
            if not raw_text or len(raw_text.strip()) < 20:
                st.error("Could not extract text.")
                os.unlink(tmp_path)
                st.stop()

            st.write("\U0001F50D Extracting entities...")
            sections = split_sections(raw_text)
            cv = extract_all(raw_text, sections=sections)
            if not cv:
                st.error("Extraction failed.")
                os.unlink(tmp_path)
                st.stop()

            st.write("\U0001F3AF Scoring...")
            if use_custom_weights and total > 0:
                custom_config = make_custom_config(rubric_config, custom_weights)
                tmp_config = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                json.dump(custom_config, tmp_config)
                tmp_config_path = tmp_config.name
                tmp_config.close()
                cv = score_cv_fn(cv, config_path=tmp_config_path)
                os.unlink(tmp_config_path)
            else:
                cv = score_cv_fn(cv)

            st.write("\U0001F4A1 Generating suggestions...")
            suggestions = generate_suggestions(cv)

            ml_label = None
            ml_proba = None
            if model_pipeline:
                ml_label, ml_proba = classify_text(model_pipeline, raw_text)

        os.unlink(tmp_path)
        status.update(label="Complete!", state="complete", expanded=False)

        total_score = cv.get("total_score", 0)
        rubric_label = cv.get("label", "Unknown")

        entry = {
            "filename": uploaded_file.name,
            "total_score": total_score,
            "rubric_label": rubric_label,
            "ml_label": ml_label,
        }
        st.session_state.history.insert(0, entry)
        st.session_state.history = st.session_state.history[:5]

        st.divider()
        st.subheader("\U0001F4CA Results")

        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            render_metric_card(
                "RUBRIC SCORE",
                f"{total_score:.0f}/100",
                f"Label: {rubric_label}",
                LABEL_COLORS.get(rubric_label, "#888"),
            )
        with kpi_cols[1]:
            if ml_label:
                ml_color = LABEL_COLORS.get(ml_label, "#888")
                conf_str = ""
                if ml_proba is not None:
                    idx = {"Average": 0, "Strong": 1, "Weak": 2}.get(ml_label, 0)
                    conf_str = f"Confidence: {ml_proba[idx]:.1%}"
                render_metric_card("ML CLASSIFICATION", ml_label, conf_str, ml_color)
            else:
                render_metric_card("ML CLASSIFICATION", "\u2014", "No model", "#888")
        with kpi_cols[2]:
            if ml_label:
                agree = rubric_label == ml_label
                border = "#15803d" if agree else "#b45309"
                bg = "#f0fdf4" if agree else "#fffbeb"
                icon = "\u2705" if agree else "\u26A0\uFE0F"
                sub = "Rubric & ML agree" if agree else f"Rubric: {rubric_label} / ML: {ml_label}"
                st.markdown(
                    f"<div style='border:1px solid {border}; border-radius:0.75rem; "
                    f"padding:1.25rem; text-align:center; background:{bg};'>"
                    f"<div style='font-size:0.8rem; color:#6b7280;'>AGREEMENT</div>"
                    f"<div style='font-size:2.2rem;'>{icon}</div>"
                    f"<div style='font-size:0.85rem; color:{border};'>{sub}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                render_metric_card("AGREEMENT", "\u2014", "No ML model", "#888")

        st.divider()
        st.subheader("\U0001F4C8 Section Scores")
        section_scores = cv.get("section_scores", {})
        for section, score in section_scores.items():
            cfg_entry = rubric_config.get(section, {})
            if isinstance(cfg_entry, dict):
                max_pts = custom_weights.get(section, cfg_entry.get("max_points", 100))
            else:
                max_pts = 100
            render_section_bar(section.replace("_", " ").title(), score, max_pts)

        st.divider()
        tabs = st.tabs([
            "\U0001F4CB Extracted Data",
            "\U0001F4A1 Suggestions",
            "\U0001F4DD Raw Text",
            "\U0001F4CA History",
        ])

        with tabs[0]:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Name:** {cv.get('name', 'N/A')}")
                st.markdown(f"**Email:** {cv.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {cv.get('phone', 'N/A')}")
                skills = cv.get("skills", [])
                st.markdown(f"**Skills ({len(skills)}):** {', '.join(skills[:15])}{'...' if len(skills) > 15 else ''}")
                if cv.get("languages"):
                    st.markdown(f"**Languages:** {', '.join(cv['languages'])}")

            with col_b:
                render_table(
                    cv.get("experience", []),
                    [("title", "Title"), ("company", "Company"), ("duration", "Duration")],
                    "experience entries"
                )
                render_table(
                    cv.get("education", []),
                    [("degree", "Degree"), ("institution", "Institution"), ("year", "Year")],
                    "education entries"
                )
                render_table(
                    cv.get("projects", []),
                    [("name", "Project"), ("tools", "Tools")],
                    "projects"
                )
                if cv.get("certifications"):
                    st.markdown(f"**Certifications:** {len(cv['certifications'])} — {', '.join(c['name'] for c in cv['certifications'][:5])}")

        with tabs[1]:
            if suggestions:
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.info("No suggestions needed.")

        with tabs[2]:
            st.text_area("Extracted text", raw_text, height=250, label_visibility="collapsed")

        with tabs[3]:
            if st.session_state.history:
                hist_df = pd.DataFrame(st.session_state.history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("No previous analyses in this session.")

        st.divider()
        cv_json = json.dumps(cv, indent=2, default=str)
        st.download_button(
            label="\U0001F4E5 Download Full Analysis (JSON)",
            data=cv_json,
            file_name=f"{uploaded_file.name}_analysis.json",
            mime="application/json",
        )

    else:
        st.info("Upload a CV to begin analysis.")

        if st.session_state.history:
            st.divider()
            st.subheader("\U0001F4CA Session History")
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            ---
            **Pipeline:**
            1. **Parse** — extract text from PDF/DOCX/TXT
            2. **Extract** — NER + rule-based extraction of skills, experience, education, projects
            3. **Score** — weighted rubric (0-100) → label (Strong/Average/Weak)
            4. **Classify** — TF-IDF + XGBoost on raw text (ML-based label)
            5. **Suggest** — targeted improvement tips per section

            ---
            **Custom Rubric:** Adjust section weights in the sidebar to match your
            company's hiring priorities before uploading.
            """
        )


if __name__ == "__main__":
    main()
