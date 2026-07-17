"""
Streamlit V1 — CV Evaluator & Classifier

Upload a CV (PDF/DOCX/TXT) → parse → extract → score (rubric)
→ classify (ML: TF-IDF + XGBoost) → suggest improvements
→ Compare rubric vs ML labels

Features:
- Adjustable rubric weights per section
- Structured tables for experience/education/projects
- Pipeline step visualization
- Section mini-score-cards
- Key strengths extraction
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
PURPLE = "#6366F1"
GREEN = "#10B981"


LABEL_COLORS = {
    "Strong": GREEN,
    "Average": "#b45309",
    "Weak": "#b91c1c",
}


def _section_color(score, max_pts):
    pct = score / max_pts * 100 if max_pts > 0 else 0
    if pct >= 70:
        return GREEN
    if pct >= 40:
        return "#b45309"
    return "#b91c1c"


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


def render_pipeline():
    steps = [
        ("\U0001F4E5", "Parse", "Text extraction"),
        ("\U0001F50D", "Extract", "NER + rules"),
        ("\U0001F3AF", "Score", "Rubric 0-100"),
        ("\U0001F9EA", "Classify", "XGBoost ML"),
        ("\U0001F4A1", "Suggest", "Improvement tips"),
    ]
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    for i, (icon, title, desc) in enumerate(steps):
        idx = i * 2
        with cols[idx]:
            st.markdown(
                f"<div style='text-align:center; padding:0.5rem 0.25rem;'>"
                f"<div style='font-size:1.5rem;'>{icon}</div>"
                f"<div style='font-weight:600; font-size:0.85rem; color:{PURPLE};'>{title}</div>"
                f"<div style='font-size:0.7rem; color:#6b7280;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[idx + 1]:
                st.markdown(
                    f"<div style='text-align:center; padding-top:1.25rem; "
                    f"font-size:1.2rem; color:#d1d5db;'>&rarr;</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("---")


def render_section_cards(section_scores, rubric_config, custom_weights):
    cards_data = []
    for section, score in section_scores.items():
        cfg_entry = rubric_config.get(section, {})
        if isinstance(cfg_entry, dict):
            max_pts = custom_weights.get(section, cfg_entry.get("max_points", 100))
        else:
            max_pts = 100
        color = _section_color(score, max_pts)
        label = section.replace("_", " ").title()
        cards_data.append((label, score, max_pts, color))

    cols = st.columns(len(cards_data))
    for col, (label, score, max_pts, color) in zip(cols, cards_data):
        pct = score / max_pts * 100 if max_pts > 0 else 0
        with col:
            st.markdown(
                f"<div style='border:1px solid #e5e7eb; border-radius:0.75rem; "
                f"padding:1rem; text-align:center; background:white;'>"
                f"<div style='font-size:0.75rem; color:#6b7280; margin-bottom:0.25rem;'>{label}</div>"
                f"<div style='font-size:1.8rem; font-weight:700; color:{color};'>{score:.0f}</div>"
                f"<div style='font-size:0.75rem; color:#9ca3af;'>/ {max_pts:.0f}</div>"
                f"<div style='height:4px; background:#e5e7eb; border-radius:2px; "
                f"margin-top:0.5rem;'>"
                f"<div style='height:4px; width:{pct:.0f}%; background:{color}; "
                f"border-radius:2px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


def extract_key_strengths(cv, total_score):
    strengths = []
    skills = cv.get("skills", [])
    if len(skills) >= 5:
        strengths.append(f"Strong skill set ({len(skills)} skills matched)")
    exp_count = len(cv.get("experience", []))
    if exp_count >= 2:
        strengths.append(f"Solid work history ({exp_count} positions)")
    elif exp_count >= 1:
        strengths.append("Relevant work experience")
    edu = cv.get("education", [])
    if edu:
        top_deg = edu[0].get("degree", "")
        if any(k in top_deg.lower() for k in ["bachelor", "master", "phd", "b.tech", "m.tech"]):
            strengths.append(f"Good education background ({top_deg})")
    if cv.get("projects"):
        strengths.append(f"{len(cv['projects'])} projects demonstrated")
    if cv.get("certifications"):
        strengths.append(f"{len(cv['certifications'])} certifications")
    if cv.get("languages"):
        strengths.append(f"Multilingual ({len(cv['languages'])} languages)")
    if total_score >= 70:
        strengths.append("Well-structured CV with clear achievements")
    if not strengths:
        strengths.append("CV parsed successfully")
    return strengths


def main():
    st.set_page_config(
        page_title="CV Evaluator",
        page_icon="\U0001F4C4",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ font-family: system-ui, -apple-system, sans-serif; }}
        .upload-area {{
            border: 2px dashed #d1d5db; border-radius: 1rem;
            padding: 2rem; text-align: center; background: #fafafa;
            transition: border-color 0.2s;
        }}
        .upload-area:hover {{ border-color: {PURPLE}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []
    if "cv_cache" not in st.session_state:
        st.session_state.cv_cache = None

    # --- Top bar ---
    top_cols = st.columns([6, 1])
    with top_cols[0]:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;'>"
            f"<div style='font-size:2rem;'>&#128196;</div>"
            f"<div>"
            f"<div style='font-size:1.8rem; font-weight:700;'>CV Evaluator</div>"
            f"<div style='font-size:0.85rem; color:#6b7280;'>AI-Powered Analysis</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with top_cols[1]:
        if st.button("\U0001F5D1 Clear All", type="secondary"):
            st.session_state.history = []
            st.session_state.cv_cache = None
            st.rerun()

    st.markdown(
        "Upload a CV to extract information, score against **customizable rubric weights**, "
        "and classify quality using both rule-based scoring and ML text classification."
    )

    # --- Sidebar ---
    rubric_config = load_rubric_config()
    default_weights = load_default_weights(rubric_config)

    with st.sidebar:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;'>"
            f"<div style='font-size:1.5rem;'>&#128196;</div>"
            f"<div><div style='font-weight:700;'>CV Evaluator</div>"
            f"<div style='font-size:0.75rem; color:#6b7280;'>AI-Powered Analysis</div></div></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        model_pipeline, model_name = load_classifier()
        if model_pipeline:
            st.markdown(
                f"<div style='border:1px solid #e5e7eb; border-radius:0.75rem; padding:1rem; "
                f"background:white; margin-bottom:1rem;'>"
                f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>"
                f"<span style='color:{GREEN};'>&#10003;</span>"
                f"<span style='font-weight:600;'>{model_name}</span>"
                f"</div>"
                f"<div style='display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; color:#6b7280;'>"
                f"<span style='color:{GREEN}; font-size:0.6rem;'>&#9679;</span> Online"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"<div style='font-weight:600; color:{PURPLE}; margin-bottom:0.5rem;'>&#9881;&#65039; Rubric Weights</div>", unsafe_allow_html=True)
        st.caption("Adjust to match your hiring priorities.")

        custom_weights = {}
        total = 0
        for section in sorted(default_weights.keys()):
            label = section.replace("_", " ").title()
            default = default_weights[section]
            val = st.slider(label, min_value=0, max_value=50, value=default, key=f"w_{section}")
            custom_weights[section] = val
            total += val

        st.caption(f"**Total: {total}/100**")
        if total != 100:
            st.warning(f"Weights sum to {total}, not 100.", icon="\u26A0\uFE0F")

        use_custom_weights = st.checkbox("Apply custom weights", value=(total != 100))

        st.divider()
        st.markdown(
            f"<div style='border:1px solid #ede9fe; border-radius:0.75rem; padding:0.75rem; "
            f"background:#f5f3ff; font-size:0.85rem;'>"
            f"<div style='font-weight:600; margin-bottom:0.25rem;'>Need help?</div>"
            f"<div style='color:#6b7280; margin-bottom:0.5rem;'>Upload a CV and get instant analysis.</div>"
            f"<a href='#' style='color:{PURPLE}; text-decoration:none; font-weight:500;'>Check our guide &rarr;</a>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("\U0001F5D1 Clear History"):
            st.session_state.history = []
            st.rerun()

    # --- Upload + Tips ---
    upload_col, tips_col = st.columns([3, 1])

    with upload_col:
        st.markdown(
            f"<div class='upload-area'>"
            f"<div style='font-size:3rem; color:{PURPLE}; margin-bottom:0.5rem;'>&#11014;&#65039;</div>"
            f"<div style='font-weight:600; font-size:1.1rem; margin-bottom:0.25rem;'>"
            f"Drag & drop your CV here</div>"
            f"<div style='font-size:0.85rem; color:#6b7280; margin-bottom:1rem;'>"
            f"PDF, DOCX, TXT up to 50 MB</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose file", type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tips_col:
        st.markdown(
            f"<div style='border:1px solid #ede9fe; border-radius:0.75rem; padding:1rem; "
            f"background:#f5f3ff; height:100%;'>"
            f"<div style='font-weight:600; font-size:0.9rem; margin-bottom:0.5rem;'>"
            f"Tips for Best Results</div>"
            f"<div style='font-size:0.8rem; color:#374151; line-height:1.8;'>"
            f"&#10003; Use a clear, updated CV<br>"
            f"&#10003; Ensure sections are well-structured<br>"
            f"&#10003; Include measurable achievements<br>"
            f"&#10003; Save as PDF for best accuracy"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # --- Pipeline visualization ---
    render_pipeline()

    # --- Processing ---
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

        st.session_state.cv_cache = {
            "raw_text": raw_text,
            "cv": cv,
            "suggestions": suggestions,
            "ml_label": ml_label,
            "ml_proba": ml_proba,
        }

    # --- Results (from cache or fresh) ---
    if st.session_state.cv_cache is not None:
        cache = st.session_state.cv_cache
        raw_text = cache["raw_text"]
        cv = cache["cv"]
        suggestions = cache["suggestions"]
        ml_label = cache["ml_label"]
        ml_proba = cache["ml_proba"]

        total_score = cv.get("total_score", 0)
        rubric_label = cv.get("label", "Unknown")

        entry = {
            "filename": uploaded_file.name if uploaded_file else "N/A",
            "total_score": total_score,
            "rubric_label": rubric_label,
            "ml_label": ml_label,
        }
        st.session_state.history.insert(0, entry)
        st.session_state.history = st.session_state.history[:5]

        # --- KPI row ---
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
            strengths = extract_key_strengths(cv, total_score)
            st.markdown(
                f"<div style='border:1px solid #e5e7eb; border-radius:0.75rem; padding:1rem; "
                f"background:white; height:100%;'>"
                f"<div style='font-size:0.8rem; color:#6b7280; margin-bottom:0.5rem;'>KEY STRENGTHS</div>"
                + "".join(
                    f"<div style='font-size:0.85rem; color:#374151; margin-bottom:0.25rem;'>"
                    f"<span style='color:{GREEN};'>&#10003;</span> {s}</div>"
                    for s in strengths[:5]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        # --- Section score cards ---
        st.divider()
        st.subheader("\U0001F4C8 Section Breakdown")
        section_scores = cv.get("section_scores", {})
        render_section_cards(section_scores, rubric_config, custom_weights)

        # --- Tabs ---
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
                    certs = cv["certifications"]
                    st.markdown(f"**Certifications ({len(certs)}):** {', '.join(c['name'] for c in certs[:5])}")

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
        col_dl, _ = st.columns([1, 4])
        with col_dl:
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


if __name__ == "__main__":
    main()
