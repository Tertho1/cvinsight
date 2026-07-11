# CV Evaluator & Ranking System — Agent Guide

## Project Overview

Automatic CV evaluation, scoring, and ranking tool. Upload a CV (PDF/DOCX/TXT), extract entities, score against a rubric, classify quality, and get improvement suggestions. Optionally match against a job description and rank multiple candidates.

**Versions:** V1 (extract→score→suggest), V2 (+JD matching & ranking), V3 (fine-tune NER, stretch)

---

## Repository Structure

```
cvinsight/
├── app/                  # Streamlit UI (not yet built)
├── config/               # JSON configuration files
│   ├── rubric_config.json    # Scoring weights & thresholds
│   └── skill_taxonomy.json   # ~270 skills in 8 categories
├── data/
│   ├── raw/              # Original downloaded datasets (do not edit)
│   └── processed/        # Cleaned CSVs + extracted_cvs.json (3000 CVs)
├── demo/                 # Sample CVs for demo (not yet populated)
├── models/               # Trained .pkl files (not yet built)
├── notebooks/            # Jupyter notebooks for EDA & eval
│   ├── day3_eda.ipynb
│   ├── week1_summary.ipynb
│   ├── week2_summary.ipynb
│   └── week3_extraction_eval.ipynb
├── scripts/              # Utility scripts (dataset loading, debugging)
├── src/                  # All source code
│   ├── parser/           # PDF/DOCX/TXT/OCR parsing (✅ done)
│   ├── extractor/        # NER + rule-based extraction (✅ ~90% done)
│   ├── scorer/           # Rubric scoring engine (❌ not started)
│   ├── matcher/          # JD similarity & ranking (❌ not started)
│   └── suggester/        # Suggestion generation (❌ not started)
├── tests/                # 285 unit tests (14 files, all passing)
├── AGENTS.md             # This file
├── TODO.md               # Current task tracking
├── progress.md           # Weekly progress report
├── project_plan.md       # Full week-by-week roadmap
└── requirements.txt      # Python dependencies
```

---

## Key Files & Conventions

### Data Flow
```
CV file → parser.py → raw text → section_splitter.py → sections dict
       → extract_all() → CVSchema (Pydantic) → scorer → score + label
       → suggester → suggestions
       Optional: + JD → matcher → rank_cvs() → ranked list
```

### CVSchema (src/schema.py)
Pydantic model with nested models (Education, Experience, Project, Certification, Language, SectionScores, JDMatch). All modules read/write this structure. Fields: `cv_id`, `name`, `email`, `phone`, `education`, `experience`, `skills`, `projects`, `certifications`, `languages`, `achievements`, `leadership`, `section_scores`, `total_score`, `label`, `suggestions`, `jd_match`.

### Testing
- Framework: **pytest** (plain asserts, no unittest.TestCase)
- Run all: `pytest tests/`
- 285 tests across 14 files
- No tests yet for scorer/suggester/matcher (modules not built)

### Python
- Version: 3.14 (project plan says 3.10 but env is 3.14)
- Style: PEP 8, no type annotations required
- Warnings: Use raw strings `r"..."` for regex to avoid SyntaxWarnings

---

## Build Order (DO NOT DEVIATE)

1. **Week 4 — Scoring Engine** (critical dependency)
   - `src/scorer/section_scorers.py` — 7 section scoring functions
   - `src/scorer/scorer.py` with `score_cv()` — master scorer
   - `src/scorer/feature_builder.py` — feature vectors for classifiers
   - `src/suggester/suggester.py` — `generate_suggestions()`
   - Output: `labeled_cvs.csv`, `borderline_review.csv`

2. **Week 5 — Classifier + Streamlit V1**
   - Train Logistic Regression + XGBoost
   - `app/app.py` — Streamlit UI: upload → score → display
   - Deploy to Hugging Face Spaces

3. **Week 6 — JD Matching & Ranking (V2)**
   - `src/matcher/embedder.py`, `semantic_scorer.py`, `skill_overlap.py`, `ranker.py`
   - Add ranking tab to Streamlit

4. **Week 7 — Fine-Tuning & Final Report (stretch)**
   - Fine-tune NER, retrain classifier, evaluation report

---

## Agent Guidelines

- **Read first:** Before editing any file, read it to understand conventions
- **No new dependencies** without updating `project_plan.md`
- **Tests required** for new modules (scorer, suggester, matcher, app)
- **Config-driven** — scoring weights live in `rubric_config.json`, not hardcoded
- **Schema-first** — all modules read/write `CVSchema`
- **Commit discipline** — only commit when asked; use descriptive messages
- **No comments** in code unless necessary for clarity
- **Mark progress** in `TODO.md` and `progress.md` after completing tasks
