"""
Streamlit V1 — CV-Insight

Upload a CV (PDF/DOCX/TXT) → parse → extract → score (rubric)
→ classify (ML: hybrid XGBoost + semantic embedding) → suggest improvements
→ Match against a job description + rank candidates

Features:
- Adjustable rubric weights per section (with re-score-all)
- Structured tables for experience/education/projects
- Pipeline step visualization
- Section mini-score-cards with rationale + top-3-to-improve
- Key strengths extraction
- Persistent CV database (History, Skill Search, Ranking, CSV/JSON export)
"""

import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import warnings
from datetime import datetime

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
HYBRID_PATH = os.path.join(MODELS_DIR, "classifier_v3_hybrid_synth.pkl")
DEFAULT_RUBRIC_PATH = os.path.join(CONFIG_DIR, "rubric_config.json")
MAX_FILE_MB = 50
PARSE_TIMEOUT = 180
DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cv_database.json")
DATABASE_BAK_PATH = DATABASE_PATH + ".bak"
DATABASE_COUNT_PATH = DATABASE_PATH + ".count"
PURPLE = "#818cf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
MUTED = "#9ca3af"


LABEL_COLORS = {
    "Strong": GREEN,
    "Average": AMBER,
    "Weak": RED,
}


def get_match(entry: dict) -> dict:
    """Read the JD-match result from a DB entry, tolerating the legacy
    'jd_match' key (renamed to 'match' in V2)."""
    m = entry.get("match") or entry.get("jd_match") or {}
    return m if isinstance(m, dict) else {}


def app_logo_data_uri() -> str:
    """Base64 data URI of app/logo.png (empty string if the file is absent)."""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if not os.path.exists(logo_path):
        return ""
    import base64
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_database() -> tuple[dict, str | None]:
    """Load the persisted CV database, returning (db, note).

    If the main file is corrupt, fall back to the .bak snapshot. The note is
    surfaced to the UI so data loss is never silent (previous behaviour
    returned {} on any error, which looked exactly like an innocent empty DB).
    """
    note = None
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                return json.load(f), None
        except (json.JSONDecodeError, OSError) as e:
            note = f"cv_database.json was unreadable ({e}). "
    if os.path.exists(DATABASE_BAK_PATH):
        try:
            with open(DATABASE_BAK_PATH, "r", encoding="utf-8") as f:
                restored = json.load(f)
            note = (note or "") + f"Recovered {len(restored)} CV(s) from backup."
            return restored, note
        except (json.JSONDecodeError, OSError) as e:
            note = (note or "") + f"Backup file also unreadable ({e}). "
    return {}, note


def count_saved_cvs() -> int:
    """Number of CV results persisted on disk, read from a tiny sidecar file so
    the app can lazy-load the full results (a cheap count at boot instead of
    parsing the whole JSON)."""
    if os.path.exists(DATABASE_COUNT_PATH):
        try:
            with open(DATABASE_COUNT_PATH, "r", encoding="utf-8") as f:
                return int(f.read().strip() or 0)
        except (ValueError, OSError):
            pass
    _db, _ = load_database()
    return len(_db)


def save_database(db: dict) -> None:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    # Atomic write: dump to a temp file in the same dir, then replace. Avoids a
    # crash mid-write corrupting the DB (a very real risk for ~MB-sized files).
    tmp_path = DATABASE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, default=str)
    if os.path.exists(DATABASE_PATH):
        try:
            shutil.copy2(DATABASE_PATH, DATABASE_BAK_PATH)
        except OSError:
            pass
    os.replace(tmp_path, DATABASE_PATH)
    try:
        with open(DATABASE_COUNT_PATH, "w", encoding="utf-8") as f:
            f.write(str(len(db)))
    except OSError:
        pass


def build_comparison_df(db: dict) -> pd.DataFrame:
    rows = []
    for cid, entry in db.items():
        cv = entry.get("cv", {})
        engine = entry.get("extractor", "")
        engine = engine.replace("spaCy + ", "")
        rows.append({
            "cv_id": cid,
            "Name": cv.get("name", "Unknown"),
            "Score": cv.get("total_score", 0),
            "Label": cv.get("label", ""),
            "Skills": len(cv.get("skills", [])),
            "Experience": len(cv.get("experience", [])),
            "Education": len(cv.get("education", [])),
            "Projects": len(cv.get("projects", [])),
            "Engine": engine,
            "Flagged": len(entry.get("warnings") or []),
            "Filename": entry.get("filename", ""),
            "Date": entry.get("timestamp", "")[:10] if entry.get("timestamp") else "",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def build_detailed_export_df(db: dict) -> pd.DataFrame:
    rows = []
    for cid, entry in db.items():
        cv = entry.get("cv", {})
        jd = get_match(entry)
        rows.append({
            "Name": cv.get("name", ""),
            "Email": cv.get("email", ""),
            "Phone": cv.get("phone", ""),
            "Score": cv.get("total_score", 0),
            "Label": cv.get("label", ""),
            "Skills": ", ".join(cv.get("skills", [])),
            "Experience": "; ".join(
                f"{e.get('title','')} @ {e.get('company','')}"
                for e in cv.get("experience", []) if e.get('title') or e.get('company')
            ),
            "Education": "; ".join(
                f"{e.get('degree','')} @ {e.get('institution','')}"
                for e in cv.get("education", []) if e.get('degree') or e.get('institution')
            ),
            "Projects": "; ".join(p.get("name", "") for p in cv.get("projects", []) if p.get("name")),
            "Certifications": "; ".join(c.get("name", "") for c in cv.get("certifications", []) if c.get("name")),
            "Languages": "; ".join(l.get("language", "") for l in cv.get("languages", []) if l.get("language")),
            "JD Match %": round(jd.get("final_match_score", 0) * 100, 1),
            "Missing Skills": ", ".join(jd.get("missing_skills", [])),
            "Filename": entry.get("filename", ""),
            "Date": entry.get("timestamp", "")[:10] if entry.get("timestamp") else "",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def _section_color(score, max_pts):
    pct = score / max_pts * 100 if max_pts > 0 else 0
    if pct >= 70:
        return GREEN
    if pct >= 40:
        return AMBER
    return RED


def force_classifier_cpu(model):
    """Pin an XGBoost-backed classifier to CPU.

    The hybrid model pickle was saved with `device=cuda`, and on Windows an
    XGBoost predict that must fall back from a CUDA booster to a CPU DMatrix
    can hard-crash the whole process (seen in app crashes during CV classify).
    Force the booster to CPU at load so predict never touches the device path.
    """
    try:
        reg = getattr(model, "regressor", None)
        if reg is None or not hasattr(reg, "get_booster"):
            return
        reg.set_params(device="cpu")
        booster = reg.get_booster()
        booster.set_param("device", "cpu")
        reg._Booster = booster
    except Exception:
        pass


@st.cache_resource
def load_classifier():
    import joblib
    import warnings as _w
    if os.path.exists(HYBRID_PATH):
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                clf = joblib.load(HYBRID_PATH)
                force_classifier_cpu(clf)
                return clf, "Hybrid (v3 synth)"
        except Exception as e:
            st.warning(f"Hybrid classifier load failed ({e}), falling back to XGBoost...")
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
def load_ner_tagger():
    from src.extractor.ner_tag import load_tagger
    return load_tagger(device_name="cpu")


@st.cache_resource
def load_llm_model(device):
    from src.extractor.hybrid import load_model
    if device == "auto":
        import torch
        device = "gpu" if torch.cuda.is_available() else "cpu"
    return load_model(adapter="models/qwen3-0.6b-cv-lora-v2", device=device)


@st.cache_resource
def load_rubric_config():
    if os.path.exists(DEFAULT_RUBRIC_PATH):
        with open(DEFAULT_RUBRIC_PATH) as f:
            return json.load(f)
    return {}


@st.cache_resource
def preload_matcher():
    """Warm the JD-matching embedder once at app start.

    Loads models/matcher-confit eagerly so the first JD match doesn't pay the
    ~10s model-load cold start mid-processing. Cached for the session; a no-op
    on subsequent runs.
    """
    try:
        from src.matcher.embedder import warm_up
        return warm_up()
    except Exception:
        return False


def load_default_weights(config):
    weights = {}
    for section, cfg in config.items():
        if isinstance(cfg, dict) and "max_points" in cfg:
            weights[section] = cfg["max_points"]
    return weights


def classify_text(model, text):
    import numpy as np
    raw = model.predict([text])[0]
    classes = None
    if hasattr(model, "label_classes_"):
        classes = list(model.label_classes_)
    elif hasattr(model, "classes_"):
        classes = list(model.classes_)
    if classes is not None:
        classes = [str(c) for c in classes]
    if isinstance(raw, (int, float, np.integer, np.floating, type(None))):
        if classes is not None:
            label = classes[int(raw)]
        else:
            label_map = ["Average", "Strong", "Weak"]
            label = label_map[int(raw)]
    else:
        label = str(raw)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
    return label, proba, classes


def current_jd_value():
    """Live JD text from the (persisted) input widget; falls back to the last
    committed_JD text so auto-matching works without pressing Match."""
    value = st.session_state.get("jd_input", "")
    if not value:
        value = st.session_state.get("_jd_text", "")
    return str(value or "")


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


def weights_key(custom_weights):
    """Stable fingerprint of the weight sliders. Stored per CV so the UI can
    tell whether a stored score was computed with the weights now on screen."""
    if not custom_weights:
        return None
    return json.dumps(custom_weights, sort_keys=True)


def write_weights_config(rubric_config, custom_weights):
    """Write the current sliders to a temp rubric JSON and return its path.
    The caller must rm the path when done (see score_and_suggest)."""
    custom_config = make_custom_config(rubric_config, custom_weights)
    tmp_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(custom_config, tmp_config)
    path = tmp_config.name
    tmp_config.close()
    return path


def score_and_suggest(cv, rubric_config, custom_weights, use_custom_weights, total,
                     score_cv_fn, generate_suggestions):
    """Score a CV dict (and rebuild suggestions) against the current weight
    sliders. Uses a temp rubric config so the scorer, the suggester, and the
    stored weights all describe the same rubric. Returns (cv, suggestions,
    weights_key)."""
    weights_key_value = None
    tmp_path = None
    if use_custom_weights and total > 0:
        tmp_path = write_weights_config(rubric_config, custom_weights)
        weights_key_value = weights_key(custom_weights)
    try:
        if tmp_path is not None:
            cv = score_cv_fn(cv, config_path=tmp_path)
            suggestions = generate_suggestions(cv, config_path=tmp_path)
        else:
            cv = score_cv_fn(cv)
            suggestions = generate_suggestions(cv)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return cv, suggestions, weights_key_value


BORDER = "2px solid rgba(128,128,128,0.35)"


def render_metric_card(title, value, subtitle, color):
    st.markdown(
        f"""
        <div style="border:{BORDER}; border-radius:0.75rem; padding:1.25rem;
                    text-align:center;">
            <div style="font-size:0.8rem; color:{MUTED}; margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:2.2rem; font-weight:700; color:{color};">{value}</div>
            <div style="font-size:0.85rem; color:{MUTED}; margin-top:0.25rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True
    )


def format_duration(months):
    if not months:
        return ""
    years, rem = divmod(int(months), 12)
    if years and rem:
        return f"{years}y {rem}m"
    if years:
        return f"{years}y"
    return f"{rem}m"


def jump_to_cv(cid):
    """Switch the displayed CV from anywhere (session picker, history radio,
    ranking, skill search). All CV selectors are remounted via a key that
    includes the active CV id, so their stored widget values cannot override
    the jump on the following rerun."""
    db = st.session_state.cv_database
    if cid not in db or cid == st.session_state.active_cv_id:
        return
    st.session_state.cv_cache = db[cid]
    st.session_state.active_cv_id = cid
    st.rerun()


def delete_cv(cid):
    """Remove a single CV from the database (and session state), keeping the
    user on a consistent CV. Used by the History tab and the detail header."""
    db = st.session_state.cv_database
    if cid not in db:
        return
    del db[cid]
    if cid in st.session_state.session_uploads:
        st.session_state.session_uploads.remove(cid)
    st.session_state.cv_cache = None
    st.session_state.active_cv_id = None
    save_database(db)
    st.rerun()


@st.dialog("Clear all stored CV data")
def confirm_clear_all_dialog():
    """Modal used by the top-bar 🗑 button; a compact centered card instead of
    a full-width inline warning."""
    _n_clear = (
        len(st.session_state.cv_database)
        if st.session_state.cv_database
        else st.session_state.get("_saved_count", 0)
    )
    st.warning(f"This permanently deletes all {_n_clear} stored CV(s). There is no undo.")
    c_y, c_n = st.columns(2)
    with c_y:
        if st.button("Yes, delete everything", type="primary", width="stretch"):
            st.session_state.cv_database = {}
            st.session_state.cv_cache = None
            st.session_state.active_cv_id = None
            st.session_state.session_uploads = []
            st.session_state._processed_keys = set()
            st.session_state.uploader_epoch += 1
            st.session_state["_cvs_loaded"] = True
            st.session_state["_saved_count"] = 0
            st.session_state["_confirm_clear"] = False
            save_database({})
            st.rerun()
    with c_n:
        if st.button("Cancel", width="stretch"):
            st.session_state["_confirm_clear"] = False
            st.rerun()


@st.cache_data(show_spinner=False)
def _rank_cvs_cached(cvs_tuple, jd_text):
    """Stable re-ranking of a CV snapshot against a JD.

    Caches the full embed+score pipeline keyed on (CV data, JD), so tab
    switches / page reruns do not re-embed every CV (~seconds of CPU) each
    time. The cache key changes automatically whenever any CV's text or skills
    change, so results never go stale.
    """
    from src.matcher.ranker import rank_cvs
    cvs = [
        {"cv_id": c, "name": n, "raw_text": r, "skills": list(sk), "total_score": sc}
        for c, n, r, sk, sc in cvs_tuple
    ]
    return rank_cvs(cvs, jd_text)


def render_table(items, columns, caption, formatters=None):
    if not items:
        st.caption(f"No {caption} found.")
        return
    rows = []
    for item in items[:10]:
        row = {}
        for col_key, col_label in columns:
            if formatters and col_key in formatters:
                row[col_label] = formatters[col_key](item)
            else:
                row[col_label] = item.get(col_key, "")
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)
    if len(items) > 10:
        st.caption(f"Showing 10 of {len(items)} {caption}")


def render_pipeline():
    steps = [
        ("\U0001F4E5", "Parse", "Text extraction"),
        ("\U0001F50D", "Extract", "NER + rules"),
        ("\U0001F3AF", "Score", "Rubric 0-100"),
        ("\U0001F9EA", "Classify", "Hybrid ML"),
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
                f"<div style='font-size:0.7rem; color:{MUTED};'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[idx + 1]:
                st.markdown(
                    f"<div style='text-align:center; padding-top:1.25rem; "
                    f"font-size:1.2rem; color:{MUTED};'>&rarr;</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("---")


def render_section_cards(section_scores, rubric_config, custom_weights, criteria_scores=None):
    cards_data = []
    for section, score in section_scores.items():
        cfg_entry = rubric_config.get(section, {})
        if isinstance(cfg_entry, dict):
            max_pts = custom_weights.get(section, cfg_entry.get("max_points", 100))
        else:
            max_pts = 100
        color = _section_color(score, max_pts)
        label = section.replace("_", " ").title()
        rationale = ""
        if criteria_scores:
            for c in criteria_scores:
                if c.get("name") == section:
                    rationale = c.get("rationale", "")
                    break
        cards_data.append((label, score, max_pts, color, rationale))

    cols = st.columns(len(cards_data))
    for col, (label, score, max_pts, color, rationale) in zip(cols, cards_data):
        pct = score / max_pts * 100 if max_pts > 0 else 0
        with col:
            st.markdown(
                f"<div style='border:{BORDER}; border-radius:0.75rem; "
                f"padding:1rem; text-align:center;'>"
                f"<div style='font-size:0.75rem; color:{MUTED}; margin-bottom:0.25rem;'>{label}</div>"
                f"<div style='font-size:1.8rem; font-weight:700; color:{color};'>{score:.0f}</div>"
                f"<div style='font-size:0.75rem; color:{MUTED};'>/ {max_pts:.0f}</div>"
                f"<div style='height:4px; background:rgba(128,128,128,0.15); border-radius:2px; "
                f"margin-top:0.5rem;'>"
                f"<div style='height:4px; width:{pct:.0f}%; background:{color}; "
                f"border-radius:2px;'></div>"
                f"</div>"
                + (f"<div style='font-size:0.72rem; color:{MUTED}; margin-top:0.5rem;'>{rationale}</div>"
                   if rationale else "")
                + "</div>",
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


def top_improvements(cv):
    """Rank rubric criteria worst-first by (score / max_points), return top 3."""
    criteria = cv.get("criteria_scores") or []
    rows = []
    for c in criteria:
        max_pts = c.get("max_points", 0)
        if max_pts <= 0:
            continue
        frac = (c.get("score", 0) / max_pts) * 100
        rows.append((frac, c))
    rows.sort(key=lambda x: x[0])
    return [(c, frac) for frac, c in rows[:3]]


def render_top_improvements(cv):
    items = top_improvements(cv)
    if not items:
        return
    st.markdown("### \U0001F3AF Top areas to improve")
    for c_entry, frac in items:
        label = (c_entry.get("name") or "").replace("_", " ").title()
        pct = max(0, min(100, int(round(frac))))
        color = _section_color(frac, 100.0)
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.6rem; "
            f"margin-bottom:0.35rem;'>"
            f"<span style='flex:0 0 130px; font-weight:600; font-size:0.85rem;'>{label}</span>"
            f"<div style='flex:1; height:6px; background:rgba(128,128,128,0.15); "
            f"border-radius:3px;'>"
            f"<div style='height:6px; width:{pct}%; background:{color}; border-radius:3px;'></div>"
            f"</div>"
            f"<span style='flex:0 0 40px; text-align:right; font-size:0.8rem; "
            f"color:{MUTED};'>{frac:.0f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="CV-Insight",
        page_icon="\U0001F4C4",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ font-family: system-ui, -apple-system, sans-serif; }}

        /* Style the native file_uploader as the upload box */
        [data-testid="stFileUploader"] {{
            border: 3px dashed rgba(128,128,128,0.35);
            border-radius: 1rem;
            transition: border-color 0.2s;
            overflow: hidden;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: {PURPLE};
        }}
        [data-testid="stFileUploader"] > section {{
            border: none !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            height: 255px !important;
            padding: 1.5rem !important;
            text-align: center !important;
            cursor: pointer !important;
            box-sizing: border-box !important;
        }}
        /* Hide the label (takes space above section even when collapsed) */
        [data-testid="stFileUploader"] > label {{
            display: none !important;
        }}
        /* Hide the Upload button span and instruction text */
        [data-testid="stFileUploader"] > section > span {{
            display: none !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] {{
            display: none !important;
        }}
        /* Hide file metadata chips that appear after upload */
        [data-testid="stFileChips"] {{
            display: none !important;
        }}
        /* Hide file metadata after upload */
        div.stFileUploaderFile {{
            display: none !important;
        }}
        /* Custom upload-box icon */
        [data-testid="stFileUploader"] > section::before {{
            content: "⬆️";
            font-size: 3rem;
            display: block;
            margin-bottom: 0.5rem;
            line-height: 1;
        }}
        /* Custom upload-box text */
        [data-testid="stFileUploader"] > section::after {{
            content: "Click or drag your CV here\\A PDF, DOCX, TXT up to 50 MB";
            white-space: pre;
            display: block;
            font-weight: 600;
            font-size: 1.1rem;
            line-height: 1.5;
        }}

        [data-testid="stDataFrame"] > div {{
            background: transparent !important;
        }}
        [data-testid="stDataFrame"] table {{
            background: transparent !important;
        }}

        /* Sidebar header: float the collapse arrow over the top-right so the
           brand (logo + text) shares the same top row instead of sitting
           below a reserved strip. */
        [data-testid="stSidebarContent"] {{
            position: relative;
        }}
        [data-testid="stSidebarHeader"] {{
            position: absolute;
            top: 0;
            right: 0;
            left: auto;
            z-index: 20;
            padding: 0.5rem 0.5rem 0 0;
            background: transparent;
        }}
        [data-testid="stSidebarHeader"] [data-testid="stLogoSpacer"] {{
            display: none;
        }}
        [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] button {{
            background: transparent;
            box-shadow: none;
        }}
        /* Tighten the space around sidebar section dividers */
        [data-testid="stSidebar"] hr {{
            margin: 0.4rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "cv_database" not in st.session_state:
        st.session_state.cv_database = {}
    if "_cvs_loaded" not in st.session_state:
        st.session_state["_cvs_loaded"] = False
    if "_saved_count" not in st.session_state:
        st.session_state["_saved_count"] = count_saved_cvs()
    if "active_cv_id" not in st.session_state:
        st.session_state.active_cv_id = None
    if "cv_cache" not in st.session_state:
        st.session_state.cv_cache = None
    if "session_uploads" not in st.session_state:
        st.session_state.session_uploads = []
    if "_processed_keys" not in st.session_state:
        st.session_state._processed_keys = set()
    if "uploader_epoch" not in st.session_state:
        st.session_state.uploader_epoch = 0

    # --- Deep-link from the Skill-Search "View" column (?open_cv=...): open the
    # app directly on that CV's start (Extracted Data tab). ---
    jump_cid = st.query_params.get("open_cv", "")
    if isinstance(jump_cid, (list, tuple)):
        jump_cid = jump_cid[0] if jump_cid else ""
    jump_cid = str(jump_cid or "")
    if jump_cid in st.session_state.cv_database and jump_cid != st.session_state.active_cv_id:
        st.session_state.cv_cache = st.session_state.cv_database[jump_cid]
        st.session_state.active_cv_id = jump_cid
        st.session_state["main_tabs"] = 0

    # --- Top bar ---
    _logo_uri = app_logo_data_uri()
    top_cols = st.columns([5, 2, 0.7])
    with top_cols[0]:
        _bar_logo = (
            f"<img src='{_logo_uri}' style='height:3rem; width:auto;' />"
            if _logo_uri
            else "<div style='font-size:2rem;'>&#128196;</div>"
        )
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;'>"
            f"{_bar_logo}"
            f"<div>"
            f"<div style='font-size:1.8rem; font-weight:700;'>CV-Insight</div>"
            f"<div style='font-size:0.85rem; color:{MUTED};'>AI-Powered CV Scorer & Job Description Matcher</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with top_cols[1]:
        n_mem = len(st.session_state.cv_database)
        n_saved = n_mem if n_mem else st.session_state.get("_saved_count", 0)
        if st.button(
            f"\U0001F4C2 {n_saved} CV(s) saved",
            key="load_saved_btn",
            width="stretch",
            disabled=(not n_saved),
            help="Load the stored CV results from disk and show them the same "
                 "way as freshly processed CVs.",
        ):
            if not st.session_state.cv_database:
                _db, _db_note = load_database()
                st.session_state.cv_database = _db
                st.session_state["_cvs_loaded"] = True
                st.session_state["_saved_count"] = len(_db)
                if _db_note:
                    st.warning(_db_note)
            if st.session_state.cv_database:
                _best = max(
                    st.session_state.cv_database.items(),
                    key=lambda kv: kv[1].get("cv", {}).get("total_score", 0) or 0,
                )[0]
                st.session_state.cv_cache = st.session_state.cv_database[_best]
                st.session_state.active_cv_id = _best
                st.session_state["main_tabs"] = 0
                st.rerun()
    with top_cols[2]:
        if st.button(
            "\U0001F5D1",
            key="clear_all_btn",
            width="stretch",
            disabled=(not st.session_state.cv_database and not st.session_state.get("_saved_count", 0)),
            help="Clear all stored CV data.",
        ):
            st.session_state["_confirm_clear"] = True

    if st.session_state.get("_confirm_clear"):
        confirm_clear_all_dialog()

    st.markdown(
        "Upload a CV to extract information, score against **customizable rubric weights**, "
        "and classify quality using both rule-based scoring and ML text classification."
    )

    # --- Sidebar ---
    rubric_config = load_rubric_config()
    default_weights = load_default_weights(rubric_config)

    with st.sidebar:
        _logo_img = (
            f"<img src='{_logo_uri}' style='height:2.4rem; width:auto;' />"
            if _logo_uri
            else f"<div style='font-size:1.5rem;'>&#128196;</div>"
        )
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; padding:5px 3rem 0 0;'>"
            f"{_logo_img}"
            f"<div><div style='font-weight:700;'>CV-Insight</div>"
            f"<div style='font-size:0.75rem; color:{MUTED};'>AI-Powered CV Scorer & Job Description Matcher</div></div></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        model_pipeline, _ = load_classifier()

        preload_matcher()

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        extractor = st.selectbox(
            "Extraction engine",
            ["spaCy + DistilBERT NER", "spaCy + Qwen3 LoRA LLM"],
            key="extractor_mode",
            help="'spaCy + DistilBERT NER' (default) runs the spaCy/rule pipeline "
                 "with a fine-tuned DistilBERT tagger fused on top (~10-60ms/CV). "
                 "'spaCy + Qwen3 LoRA LLM' additionally runs our fine-tuned Qwen3 LoRA "
                 "for deepest extraction: highest accuracy but slow (~30s/CV on GPU, "
                 "slower on CPU).",
        )
        if extractor == "spaCy + Qwen3 LoRA LLM":
            llm_device = st.selectbox("LLM device", ["auto", "gpu", "cpu"], key="llm_device",
                                      help="auto = CUDA if available, else CPU.")
        else:
            llm_device = None

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        st.divider()

        st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)

        st.markdown(f"<div style='font-weight:600; color:{PURPLE}; margin-bottom:0.5rem;'>&#9881;&#65039; Rubric Weights</div>", unsafe_allow_html=True)
        st.caption("Adjust to match your hiring priorities.")
        st.divider()

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

        st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)

        _rescore_available = bool(
            st.session_state.cv_database or st.session_state.get("_saved_count", 0)
        )
        if st.button(
            "\U0001F501 Re-score all CVs",
            disabled=(not _rescore_available),
            help="Re-runs scoring + suggestions for every stored CV using the "
                 "weight sliders (and the Apply custom weights toggle).",
        ):
            if not st.session_state.cv_database:
                _db, _db_note = load_database()
                st.session_state.cv_database = _db
                st.session_state["_cvs_loaded"] = True
            parse_cv, split_sections, extract_all, score_cv_fn, generate_suggestions_fn = (
                get_parser_extractor_scorer()
            )
            for _name, _e in st.session_state.cv_database.items():
                _e["cv"], _e["suggestions"], _e["weights_key"] = score_and_suggest(
                    _e["cv"], rubric_config, custom_weights,
                    use_custom_weights, total, score_cv_fn, generate_suggestions_fn,
                )
            save_database(st.session_state.cv_database)
            st.session_state["_rescore_msg"] = (
                f"Re-scored {len(st.session_state.cv_database)} CV(s) with "
                f"{'custom' if use_custom_weights else 'default'} weights."
            )
            st.rerun()
        if st.session_state.get("_rescore_msg"):
            st.success(st.session_state.pop("_rescore_msg"))

    # --- Upload + JD ---
    upload_col, jd_col = st.columns(2)

    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload CVs", type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"cv_uploader_{st.session_state.uploader_epoch}",
        )

    with jd_col:
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;'>"
                f"<span style='font-size:1.3rem;'>&#128269;</span>"
                f"<span style='font-weight:600;'>Job Description (optional)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            jd_text = st.text_area(
                "Paste the job description here for matching...",
                value=st.session_state.get("_jd_text", ""),
                height=120,
                placeholder="e.g. Looking for a Python developer with Django and PostgreSQL experience...",
                label_visibility="collapsed",
                key="jd_input",
            )
            btn_primary, btn_rematch = st.columns(2)
            with btn_primary:
                if st.button("Match", key="jd_match_btn", width="stretch"):
                    if jd_text.strip():
                        st.session_state["_jd_text"] = jd_text
            with btn_rematch:
                if st.button(
                    "Match all stored CVs",
                    key="jd_rematch_all_btn",
                    width="stretch",
                    help="Update stored JD match scores for every CV against the "
                         "JD above. One click instead of visiting each CV.",
                ):
                    if not jd_text.strip():
                        st.warning("Paste a job description first, then click \u201CMatch all stored CVs\u201D.")
                    elif not st.session_state.cv_database:
                        st.warning("Upload CVs first, then match them against the JD.")
                    else:
                        from src.matcher.ranker import match_cv
                        n = 0
                        for e in st.session_state.cv_database.values():
                            try:
                                e["match"] = match_cv(
                                    cv_text=e.get("raw_text", ""),
                                    cv_skills=e.get("cv", {}).get("skills", []),
                                    jd_text=jd_text.strip(),
                                    rubric_score=e.get("cv", {}).get("total_score", 0),
                                )
                                n += 1
                            except Exception:
                                continue
                        st.session_state["_jd_text"] = jd_text
                        st.session_state["_last_matched_jd"] = jd_text.strip()
                        st.session_state["_last_match_all_t"] = datetime.now().isoformat()
                        save_database(st.session_state.cv_database)
                        st.success(f"Re-matched {n} CV(s) to the current JD.")
                        st.rerun()

    # --- Pipeline visualization ---
    render_pipeline()

    # --- Processing ---
    if uploaded_files:
        processed = st.session_state._processed_keys
        new_files = []
        for f in uploaded_files:
            content = f.read()
            f.seek(0)
            cid = hashlib.md5(content).hexdigest()[:12]
            if (cid, extractor) not in processed:
                new_files.append((f, cid))

        if new_files:
            parse_cv, split_sections, extract_all, score_cv_fn, generate_suggestions = (
                get_parser_extractor_scorer()
            )
            progress_bar = st.progress(0, text="Ready")
            for i, (f, cid) in enumerate(new_files):
                content = f.read()
                msg = f"Processing {f.name} ({i+1}/{len(new_files)})"
                progress_bar.progress((i) / len(new_files), text=msg)

                file_size_mb = f.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_MB:
                    st.warning(f"Skipping {f.name}: too large ({file_size_mb:.1f} MB).")
                    continue

                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                status = st.status(f"Processing {f.name}...", expanded=(i == 0))
                pwarns = []

                with status:
                    st.write("\U0001F4E5 Parsing file...")
                    raw_text = safe_parse_cv(tmp_path) if suffix.lower() == ".pdf" else parse_cv(tmp_path)
                    if not raw_text or len(raw_text.strip()) < 20:
                        st.warning(f"Could not extract text from {f.name}.")
                        os.unlink(tmp_path)
                        continue

                    st.write("\U0001F50D Extracting entities...")
                    sections = split_sections(raw_text)
                    cv = extract_all(raw_text, sections=sections)
                    if not cv:
                        st.warning(f"Extraction failed for {f.name}.")
                        os.unlink(tmp_path)
                        continue

                    st.write("\U0001F4AC Fusing DistilBERT NER spans...")
                    try:
                        if cv.get("language") == "bangla":
                            pwarns.append("Bengali CV: English DistilBERT NER fusion skipped.")
                        else:
                            ner_model, ner_tok, _ = load_ner_tagger()
                        from src.extractor.ner_tag import predict_spans, merge_skills, extract_education_gaps
                        groups = {"skill": [], "degree": [], "institution": []}
                        if cv.get("language") != "bangla":
                            groups = predict_spans(ner_model, ner_tok, raw_text)
                            cv["skills"] = merge_skills(cv["skills"], groups)
                        edu_gaps = extract_education_gaps(cv, groups)
                        if edu_gaps:
                            cv["education"] = list(cv["education"] or []) + edu_gaps
                    except Exception as e:
                        st.warning(f"DistilBERT NER fusion failed ({e}); spaCy/rule output used.")
                        pwarns.append(f"DistilBERT NER fusion failed ({e}); spaCy/rule output used.")

                    if extractor == "spaCy + Qwen3 LoRA LLM":
                        if cv.get("language") == "bangla":
                            pwarns.append("Bengali CV: Qwen3 LoRA LLM is English-only — skipped (rule + Bangla NER output used).")
                        else:
                            st.write("\U0001F916 Running Qwen3 LoRA LLM extraction...")
                            try:
                                from src.extractor.hybrid import extract_with_llm, fuse
                                llm_model, llm_tok = load_llm_model(llm_device or "auto")
                                llm_cv = extract_with_llm(raw_text, model=llm_model,
                                                          tokenizer=llm_tok)
                                if llm_cv:
                                    cv = fuse(cv, llm_cv)
                                    st.write("\u2705 Qwen3 LoRA fused (deepest extraction)")
                                else:
                                    st.warning("LLM extraction returned nothing; DistilBERT output used.")
                                    pwarns.append("LLM extraction returned nothing; DistilBERT output used.")
                            except Exception as e:
                                st.warning(f"Qwen3 LoRA fusion failed ({e}); DistilBERT output used.")
                                pwarns.append(f"Qwen3 LoRA fusion failed ({e}); DistilBERT output used.")

                    st.write("\U0001F3AF Scoring...")
                    st.write("\U0001F4A1 Generating suggestions...")
                    cv, suggestions, entry_weights_key = score_and_suggest(
                        cv, rubric_config, custom_weights, use_custom_weights, total,
                        score_cv_fn, generate_suggestions,
                    )

                    ml_label = None
                    ml_proba = None
                    ml_classes = None
                    if model_pipeline and cv.get("language") != "bangla":
                        ml_label, ml_proba, ml_classes = classify_text(model_pipeline, raw_text)

                    jd_match_result = None
                    current_jd = current_jd_value()
                    if current_jd and current_jd.strip():
                        st.write("\U0001F4CB Matching against job description...")
                        try:
                            from src.matcher.ranker import match_cv
                            skills = cv.get("skills", [])
                            jd_match_result = match_cv(
                                cv_text=raw_text, cv_skills=skills,
                                jd_text=current_jd.strip(),
                                rubric_score=cv.get("total_score", 0),
                            )
                            st.session_state["_last_matched_jd"] = current_jd.strip()
                        except Exception as e:
                            st.warning(f"JD matching failed for {f.name}: {e}")
                            pwarns.append(f"JD matching failed: {e}")
                            jd_match_result = None

                os.unlink(tmp_path)
                status.update(label=f"Done: {f.name}", state="complete", expanded=False)

                cv_name = cv.get("name", f.name) or f.name
                st.session_state.cv_database[cid] = {
                    "filename": f.name,
                    "timestamp": datetime.now().isoformat(),
                    "extractor": extractor,
                    "weights_key": entry_weights_key,
                    "warnings": pwarns,
                    "raw_text": raw_text,
                    "cv": cv,
                    "suggestions": suggestions,
                    "ml_label": ml_label,
                    "ml_proba": ml_proba.tolist() if hasattr(ml_proba, 'tolist') else ml_proba,
                    "ml_classes": ml_classes,
                    "match": jd_match_result,
                }
                st.session_state._processed_keys.add((cid, extractor))
                if cid not in st.session_state.session_uploads:
                    st.session_state.session_uploads.append(cid)
                st.session_state.cv_cache = st.session_state.cv_database[cid]
                st.session_state.active_cv_id = cid

            progress_bar.progress(1.0, text=f"Processed {len(new_files)} CV(s)!")
            save_database(st.session_state.cv_database)

            # The uploader is a transient ingest queue. After each batch, empty
            # it (and forget the in-batch dedup keys) so retained files can
            # never be reprocessed silently on a later rerun, while a deliberate
            # re-upload of an existing CV is always re-evaluated in its own batch.
            st.session_state.uploader_epoch += 1
            st.session_state._processed_keys = set()
            st.rerun()

    # --- Results ---
    show_detail = st.session_state.cv_cache is not None

    # --- Detail View ---
    if st.session_state.cv_cache is not None and st.session_state.active_cv_id in st.session_state.cv_database:
        entry = st.session_state.cv_cache
        db = st.session_state.cv_database
        raw_text = entry["raw_text"]
        cv = entry["cv"]
        suggestions = entry["suggestions"]
        ml_label = entry["ml_label"]
        ml_proba = entry.get("ml_proba")
        ml_classes = entry.get("ml_classes")
        if isinstance(ml_proba, str):
            ml_proba = None
        jd_match = get_match(entry)

        current_jd = current_jd_value()
        last_matched = st.session_state.get("_last_matched_jd", "")
        if (current_jd.strip() and current_jd.strip() != last_matched):
            try:
                from src.matcher.ranker import match_cv
                skills = cv.get("skills", [])
                jd_match = match_cv(
                    cv_text=raw_text, cv_skills=skills,
                    jd_text=current_jd.strip(),
                    rubric_score=cv.get("total_score", 0),
                )
                entry["match"] = jd_match
                st.session_state["_last_matched_jd"] = current_jd.strip()
            except Exception as e:
                jd_match = get_match(entry)

        total_score = cv.get("total_score", 0)
        rubric_label = cv.get("label", "Unknown")

        current_weights_key = weights_key(custom_weights) if use_custom_weights and total > 0 else None
        if entry.get("weights_key") != current_weights_key:
            st.warning(
                "\u26A0\uFE0F The rubric sliders changed since this CV was scored. "
                "Use 'Re-score all CVs with these weights' in the sidebar to apply them.",
                icon="\U0001F3AF",
            )

        # --- CV picker ---
        picker_labels = []
        for scid in st.session_state.cv_database:
            e = st.session_state.cv_database[scid]
            n = (e["cv"].get("name") or "Unknown")
            s = e["cv"].get("total_score", 0)
            picker_labels.append((f"{n}  \u2014  {s:.0f}/100", scid))
        if len(picker_labels) > 1:
            cols = st.columns([1, 2, 6])
            with cols[0]:
                st.markdown(f"<div style='font-size:1.25rem; font-weight:700;'>\U0001F4CA Results</div>", unsafe_allow_html=True)
            with cols[1]:
                cur = st.session_state.active_cv_id
                idx = next((i for i, (_, c) in enumerate(picker_labels) if c == cur), 0)
                selected = st.selectbox("CV to view", [l for l, _ in picker_labels],
                    label_visibility="collapsed",
                    index=idx, key=f"session_cv_picker_{cur}")
                picked_cid = dict(picker_labels)[selected]
                if picked_cid != cur:
                    jump_to_cv(picked_cid)
            with cols[2]:
                st.empty()
        else:
            st.markdown(f"<div style='font-size:1.25rem; font-weight:700;'>\U0001F4CA Results</div>", unsafe_allow_html=True)
        badge = entry.get("extractor", "")
        lang_badge = cv.get("language")
        badges = []
        if lang_badge and lang_badge != "en":
            badges.append(f"Language: **{lang_badge.title()}**")
        if badge:
            badges.append(f"Extraction engine: **{badge}**")
        if badges:
            st.caption("\n".join(badges))
        entry_warnings = entry.get("warnings") or []
        if entry_warnings:
            for _w in entry_warnings:
                st.warning(_w, icon="\u26A0\uFE0F")
        kpi_cols = st.columns(4)
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
                if ml_proba is not None and isinstance(ml_proba, (list, tuple)):
                    idx = 0
                    if ml_classes and ml_label in ml_classes:
                        idx = ml_classes.index(ml_label)
                    elif isinstance(ml_label, str) and ml_label in ("Average", "Strong", "Weak"):
                        idx = {"Average": 0, "Strong": 1, "Weak": 2}[ml_label]
                    if idx < len(ml_proba):
                        conf_str = f"Confidence: {ml_proba[idx]:.1%}"
                render_metric_card("ML CLASSIFICATION", ml_label, conf_str, ml_color)
            else:
                render_metric_card("ML CLASSIFICATION", "\u2014", "No model", "#888")
        with kpi_cols[2]:
            if jd_match:
                match_pct = f"{jd_match['final_match_score'] * 100:.0f}%"
                sem = jd_match.get("semantic_similarity", 0)
                render_metric_card("JD MATCH", match_pct, f"Semantic: {sem:.2f}", PURPLE)
            else:
                render_metric_card("JD MATCH", "\u2014", "No JD provided", "#888")
        with kpi_cols[3]:
            strengths = extract_key_strengths(cv, total_score)
            st.markdown(
                f"<div style='border:{BORDER}; border-radius:0.75rem; padding:1rem; "
                f"height:100%;'>"
                f"<div style='font-size:0.8rem; color:{MUTED}; margin-bottom:0.5rem;'>KEY STRENGTHS</div>"
                + "".join(
                    f"<div style='font-size:0.85rem; margin-bottom:0.25rem;'>"
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
        render_section_cards(
            section_scores, rubric_config, custom_weights,
            criteria_scores=cv.get("criteria_scores", []),
        )
        render_top_improvements(cv)

        # --- Tabs ---
        st.divider()
        tabs = st.tabs([
            "\U0001F4CB Extracted Data",
            "\U0001F4A1 Suggestions",
            "\U0001F4DD Raw Text",
            "\U0001F4C2 History",
            "\U0001F91D JD Match",
            "\U0001F3C6 Ranking",
            "\U0001F50D Skill Search",
        ], key="main_tabs")

        with tabs[0]:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Name:** {cv.get('name', 'N/A')}")
                st.markdown(f"**Email:** {cv.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {cv.get('phone', 'N/A')}")
                skills = cv.get("skills", [])
                st.markdown(f"**Skills ({len(skills)}):**")
                if skills:
                    chips = "".join(
                        f"<span style='display:inline-block;background:rgba(129,140,248,0.12);"
                        f"border:1px solid {PURPLE};border-radius:0.375rem;padding:0.05rem 0.6rem;"
                        f"margin:0.1rem 0.2rem;font-size:0.85rem;'>{html.escape(str(s))}</span>"
                        for s in skills
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No skills detected.")
                if cv.get("languages"):
                    langs = ", ".join(
                        (l.get("language", "") if isinstance(l, dict) else str(l))
                        for l in cv["languages"] if l
                    )
                    if langs:
                        st.markdown(f"**Languages:** {langs}")

            with col_b:
                render_table(
                    cv.get("experience", []),
                    [("title", "Title"), ("company", "Company"), ("duration", "Duration")],
                    "experience entries",
                    formatters={"duration": lambda i: format_duration(i.get("duration_months"))},
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
            try:
                from src.parser.section_splitter import split_sections
                sections = split_sections(raw_text)
            except Exception:
                sections = {}
            if sections:
                st.markdown("**Sections found**")
                col_s = st.columns(4)
                sec_items = sorted(sections.items())
                for _si, (_k, _v) in enumerate(sec_items):
                    with col_s[_si % 4]:
                        txt = str(_v or "")
                        st.markdown(
                            f"<div style='border:{BORDER}; border-radius:0.5rem; "
                            f"padding:0.4rem 0.6rem; margin-bottom:0.4rem;'>"
                            f"<span style='font-weight:600; font-size:0.8rem; color:{PURPLE};'>"
                            f"{html.escape(_k)}</span> "
                            f"<span style='font-size:0.75rem; color:{MUTED};'>"
                            f"{len(txt)} chars</span></div>",
                            unsafe_allow_html=True,
                        )
                st.divider()
            else:
                st.caption("No sections detected by the splitter.")
            st.markdown("**Raw extracted text**")
            st.text_area("Extracted text", raw_text, height=250, label_visibility="collapsed")

        with tabs[3]:  # History — CV selector + comparison table
            if db:
                sort_col, _ = st.columns([1, 3])
                with sort_col:
                    sort_by = st.selectbox("Sort by:", ["Rubric Score", "JD Match"], key="history_sort")
                cv_items = list(db.items())
                for _cid, e in cv_items:
                    if sort_by == "JD Match":
                        jd = get_match(e)
                        e["_sort_score"] = jd.get("final_match_score", 0) * 100
                    else:
                        e["_sort_score"] = e["cv"]["total_score"]
                cv_items.sort(key=lambda x: x[1]["_sort_score"], reverse=True)

                radio_labels = {}
                for _cid, e in cv_items:
                    name = e["cv"].get("name", "Unknown") or "Unknown"
                    score = e["_sort_score"]
                    label = f"{name}  \u2014  {score:.0f}/100"
                    radio_labels[label] = _cid

                default_idx = list(radio_labels.values()).index(st.session_state.active_cv_id) if st.session_state.active_cv_id in radio_labels.values() else 0
                selected_label = st.radio(
                    "Select CV:", list(radio_labels.keys()),
                    index=default_idx, key=f"history_cv_radio_{st.session_state.active_cv_id}",
                    label_visibility="collapsed",
                )
                target_cid = radio_labels[selected_label]
                if target_cid != st.session_state.active_cv_id:
                    jump_to_cv(target_cid)

                st.divider()
                filter_q = st.text_input("Filter by name:", "", key="cv_filter", placeholder="Type to filter...")
                df = build_comparison_df(db)
                if filter_q:
                    df = df[df["Name"].str.lower().str.contains(filter_q.lower(), na=False)]
                st.dataframe(df, width='stretch', hide_index=True, use_container_width=True)
                st.caption(f"{len(db)} CV(s) total")

                st.divider()
                del_rows = [l for l in radio_labels if st.session_state.active_cv_id != radio_labels[l]]
                if del_rows:
                    c_del_a, c_del_b, _ = st.columns([2, 2, 3])
                    with c_del_a:
                        del_label = st.selectbox(
                            "Remove a CV:", ["(select a CV)"] + del_rows,
                            key="hist_delete_pick", label_visibility="collapsed",
                        )
                    with c_del_b:
                        if st.button(
                            "\U0001F5D1 Delete selected CV", type="secondary",
                            disabled=(del_label == "(select a CV)"),
                            key="hist_delete_btn",
                        ) and del_label in radio_labels:
                            delete_cv(radio_labels[del_label])
                else:
                    st.caption("Upload more CVs to enable single-CV removal.")

                st.divider()
                dl_cols = st.columns(2)
                with dl_cols[0]:
                    all_json = json.dumps(
                        {cid: e.get("cv", {}) for cid, e in db.items()},
                        indent=2, default=str,
                    )
                    st.download_button(
                        label="\U0001F4E5 All CVs (JSON)",
                        data=all_json,
                        file_name="all_cvs_analysis.json",
                        mime="application/json",
                    )
                with dl_cols[1]:
                    csv_data = build_detailed_export_df(db).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="\U0001F4C4 All CVs (CSV)",
                        data=csv_data,
                        file_name="all_cvs_data.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No CVs in database.")

        with tabs[4]:
            if jd_match:
                match = jd_match
                st.markdown(f"## Match Score: **{match['final_match_score'] * 100:.1f}%**")
                st.progress(match["final_match_score"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Semantic Similarity", f"{match['semantic_similarity']:.3f}")
                col2.metric("Skill Overlap", f"{match['skill_overlap']:.0%}")
                col3.metric("Missing Skills", str(len(match['missing_skills'])))
                st.divider()
                match_cols = st.columns(2)
                cv_skills = cv.get("skills", [])
                jd_skills_found = set()
                current_jd_tab = current_jd_value()
                if current_jd_tab:
                    from src.extractor.skill_extractor import extract_skills
                    jd_skills_found = {s.lower().strip() for s in extract_skills(current_jd_tab)}
                matched = sorted(jd_skills_found & {s.lower().strip() for s in cv_skills})
                missing = match.get("missing_skills", [])
                with match_cols[0]:
                    st.markdown("### \u2705 Matched Skills")
                    if matched:
                        for s in matched:
                            st.markdown(f"- {s}")
                    else:
                        st.caption("No skills matched.")
                with match_cols[1]:
                    st.markdown("### \u274C Missing Skills")
                    if missing:
                        for s in missing:
                            st.markdown(f"- {s}")
                    else:
                        st.caption("No missing skills.")
                st.divider()
                st.caption(
                    "Match score = 50% semantic similarity + 30% skill overlap "
                    "+ 20% rubric score (normalized)."
                )
            else:
                st.info("Paste a job description in the JD field above and re-upload the CV to see match results.")

        with tabs[5]:  # Candidate Ranking (multi-CV vs JD)
            current_jd = current_jd_value().strip()
            if not current_jd:
                st.info("Paste a job description in the JD field above, then open this tab to rank all stored CVs against it.")
            elif not db:
                st.info("No CVs in database to rank.")
            else:
                try:
                    cvs_tuple = []
                    for cid, entry in db.items():
                        cv_d = entry.get("cv", {}) or {}
                        cvs_tuple.append((
                            cid,
                            cv_d.get("name") or cv_d.get("filename") or "Unknown",
                            entry.get("raw_text", ""),
                            tuple(cv_d.get("skills", [])),
                            cv_d.get("total_score", 0),
                        ))
                    ranked = _rank_cvs_cached(tuple(cvs_tuple), current_jd)
                    st.markdown(f"### \U0001F3C6 Ranked candidates — {len(ranked)} CV(s)")
                    rows = []
                    for idx, r in enumerate(ranked, start=1):
                            miss = r.get("missing_skills") or []
                            rows.append({
                                "Rank": idx,
                                "Name": r.get("name", "Unknown"),
                                "Match %": round(r["final_match_score"] * 100, 1),
                                "Semantic": round(r["semantic_similarity"], 3),
                                "Skill Overlap": round(r["skill_overlap"] * 100, 1),
                                "Rubric Score": r.get("total_score", 0),
                                "Missing Skills": ", ".join(miss[:5]) + ("\u2026" if len(miss) > 5 else "")
                                            if miss else "\u2014",
                            })
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, use_container_width=True)

                    with st.expander("Why this ranking? (weights)"):
                        w = ranked[0]["weights"] if ranked else {}
                        st.caption(
                            f"Final = {w.get('semantic', 0.5)*100:.0f}% semantic + "
                            f"{w.get('skill', 0.3)*100:.0f}% skill overlap + "
                            f"{w.get('rubric', 0.2)*100:.0f}% rubric (+ "
                            f"{w.get('bm25', 0.0)*100:.0f}% BM25 lexical, if enabled)."
                        )

                    st.divider()
                    pick = st.selectbox("View ranked CV:", [""] + [r["name"] for r in ranked],
                                        key=f"ranking_pick_{st.session_state.active_cv_id}")
                    if pick:
                        hit = next(
                            (c for c, n, _, _, _ in cvs_tuple if n == pick), None
                        )
                        if hit:
                            jump_to_cv(hit)
                except Exception as e:
                    st.warning(f"Ranking failed: {e}")

        with tabs[6]:  # Skill Search
            if db:
                query = st.text_input(
                    "Search by skill(s):",
                    placeholder="e.g. Python, Django, React",
                    key="skill_search_input",
                )
                mode = st.radio("Match mode:", ["ALL (AND)", "ANY (OR)"], horizontal=True, key="skill_search_mode")
                if query:
                    query_skills = {s.strip().lower() for s in query.split(",") if s.strip()}
                    from src.extractor.skill_extractor import expand_skill_set
                    results = []
                    for cid, entry in db.items():
                        cv_s = entry.get("cv", {}).get("skills", [])
                        cv_skills_lower = expand_skill_set(cv_s)
                        matched_skills = query_skills & cv_skills_lower
                        missing = query_skills - cv_skills_lower
                        if mode.startswith("ALL"):
                            if matched_skills == query_skills:
                                results.append((cid, entry, matched_skills, missing))
                        else:
                            if matched_skills:
                                results.append((cid, entry, matched_skills, missing))

                    results.sort(
                        key=lambda r: (r[1].get("cv", {}).get("total_score", 0) or 0),
                        reverse=True,
                    )

                    if results:
                        st.write(f"**{len(results)}** CV(s) match:")
                        rows = []
                        cids = []
                        for cid, entry, matched_skills, missing in results:
                            cv_d = entry.get("cv", {})
                            rows.append({
                                "Name": cv_d.get("name", "Unknown"),
                                "Score": cv_d.get("total_score", 0),
                                "Matched Skills": ", ".join(sorted(matched_skills)),
                                "Unmatched Skills": ", ".join(sorted(missing)) if missing else "\u2014",
                                "View": False,
                            })
                            cids.append(cid)
                        res_df = pd.DataFrame(rows)
                        st.caption("Tick the **\u2611 View** box (rightmost column) to open that CV's detail section \u2014 no page reload.")
                        edited = st.data_editor(
                            res_df,
                            hide_index=True,
                            disabled=["Name", "Score", "Matched Skills", "Unmatched Skills"],
                            column_config={"View": st.column_config.CheckboxColumn("\u2611 View")},
                            width="stretch",
                            key=f"skill_search_view_{st.session_state.active_cv_id}",
                        )
                        new_views = [bool(v) for v in edited["View"].tolist()]
                        prev_views = st.session_state.get("skill_search_prev_views")
                        if prev_views is None or len(prev_views) != len(new_views):
                            prev_views = [False] * len(new_views)
                        for i, (cur, prev) in enumerate(zip(new_views, prev_views)):
                            if cur and not prev:
                                st.session_state["main_tabs"] = 0
                                jump_to_cv(cids[i])
                                break
                        st.session_state["skill_search_prev_views"] = new_views
                    else:
                        st.info("No CVs match the specified skills.")
                else:
                    st.caption("Enter comma-separated skills above to search across all analyzed CVs.")
            else:
                st.info("No CVs in database. Upload CVs to search them.")

        st.divider()
        col_json, col_csv = st.columns(2)
        with col_json:
            cv_json = json.dumps(cv, indent=2, default=str)
            cur_name = os.path.splitext(entry.get("filename", "cv_analysis"))[0]
            st.download_button(
                label="\U0001F4E5 This CV (JSON)",
                data=cv_json,
                file_name=f"{cur_name}_analysis.json",
                mime="application/json",
            )
        with col_csv:
            one_entry = {st.session_state.active_cv_id: entry}
            one_csv = build_detailed_export_df(one_entry).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="\U0001F4C4 This CV (CSV)",
                data=one_csv,
                file_name=f"{cur_name}_analysis.csv",
                mime="text/csv",
            )

    elif not st.session_state.cv_database:
        st.info("Upload CVs to begin analysis.")


if __name__ == "__main__":
    main()
