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
│   └── processed/        # Cleaned CSVs + extracted_cvs.json (4500 CVs)
├── demo/                 # Sample CVs for demo (not yet populated)
├── models/               # Trained .pkl files (not yet built)
├── notebooks/            # Jupyter notebooks for EDA & eval
│   ├── day3_eda.ipynb
│   ├── week1_summary.ipynb
│   ├── week2_summary.ipynb
│   ├── week3_extraction_eval.ipynb
│   └── eda_extraction_all_datasets.ipynb  # Comprehensive EDA + extraction eval
├── scripts/              # Utility scripts (dataset loading, debugging, batch extraction, analysis)
├── src/                  # All source code
│   ├── parser/           # PDF/DOCX/TXT/OCR parsing (✅ done)
│   ├── extractor/        # NER + rule-based extraction (✅ ~95%, bugs fixed in Phase 1)
│   │   └── adapters.py   # Dataset normalization adapters (Phase 2)
│   ├── scorer/           # Rubric scoring engine (✅ done)
│   ├── matcher/          # JD similarity & ranking (❌ not started)
│   └── suggester/        # Suggestion generation (✅ done)
├── tests/                # 343 unit tests (16 files, all passing)
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

Structured CSV path:
  CSV → adapters.py → sections dict → extract_all() → CVSchema
```

### CVSchema (src/schema.py)
Pydantic model with nested models (Education, Experience, Project, Certification, Language, SectionScores, JDMatch). All modules read/write this structure. Fields: `cv_id`, `name`, `email`, `phone`, `education`, `experience`, `skills`, `projects`, `certifications`, `languages`, `achievements`, `leadership`, `section_scores`, `total_score`, `label`, `suggestions`, `jd_match`.

### Testing
- Framework: **pytest** (plain asserts, no unittest.TestCase)
- Run all: `pytest tests/`
- 343 tests across 16 files (parser: 133, extractor: 285, scorer+suggester: 51)

### Python
- Version: 3.14 (project plan says 3.10 but env is 3.14)
- Style: PEP 8, no type annotations required
- Warnings: Use raw strings `r"..."` for regex to avoid SyntaxWarnings

---

## Build Order (DO NOT DEVIATE)

### ✅ Completed (Pre-Week-5)

0. **Phase 1 — Extractor Bug Fixes** (prerequisite for batch scoring)
   - Fix `try_parse_structured` tuple + string-item handling
   - Extract languages from `skills.languages` sub-key
   - Add NETSOL key fallbacks (`title`→`name`, `degree_title`→`degree.level`)
   - Add "till date" to experience date regex
   - Run batch scoring → `labeled_cvs.csv`, `borderline_review.csv`

1. **Phase 2 — Dataset Normalization Adapters** (additive, no existing code changes)
   - `src/extractor/adapters.py` — one adapter per dataset
   - Unlocks extraction on NETSOL, NER, ATS, classification datasets

2. **Phase 3 — Text-Path Rewrites** (before V1 production deploy)
   - Experience: title, description, flexible dates
   - Education: multi-line paragraph parsing
   - Projects: tool extraction from description
   - Languages: language-name detection

### ✅ Completed

3. **Phase 1c — Rule-Based Extractor Fixes** (✅ completed 2026-07-15)
   - DOCX table-aware parsing (cells on separate lines, not pipe-joined)
   - Experience title/company swap detection on PDFs
   - Education institution filtering (exclude degree names from ORG entities)
   - Languages: skip tech category lines (e.g. "Frameworks:", "Tools:")
   - Phone: add Indian number pattern `\d{5}[-.\s]?\d{5}`
   - PDF: clean `(cid:127)` markers from text output
   - All 9 demo CVs improved; 343 tests passing

4. **LLM-Based CV Extraction — Custom LoRA Fine-Tuning** (✅ v1 completed 2026-07-15)
   - `scripts/generate_training_dataset.py` — 4,612 CVs → Qwen3 chat JSONL
   - `scripts/train_llm.py` — PEFT LoRA on Qwen3-0.6B (2 epochs, eval_loss 0.499)
   - `scripts/test_finetuned.py` — eval on 9 demo CVs (valid JSON, needs data quality fix)
   - Both ready-made adapters evaluated and found unsuitable

### 🔜 Upcoming

5. **Week 5 — ML Text Classifier + Streamlit V1**
   - TF-IDF vectorization of raw CV text → XGBoost for genuine text classification
   - Compare with Logistic Regression baseline
   - `app/app.py` — Streamlit UI: upload → extract → score → classify → suggest
   - Deploy to Hugging Face Spaces

6. **Week 6 — JD Matching & Ranking (V2)**
   - `src/matcher/embedder.py`, `semantic_scorer.py`, `skill_overlap.py`, `ranker.py`
   - Add ranking tab to Streamlit

7. **Week 7 — Fine-Tuning & Final Report**
   - Custom LoRA fine-tune Qwen3-0.6B on labeled CVs
   - Bangla CV support (multilingual: Onneshon dataset, B-NER, AI4Bharat Sangraha)
   - Side-by-side eval: rule-based vs fine-tuned LLM
   - Full evaluation metrics, final report

---

## Agent Guidelines

- **Read first:** Before editing any file, read it to understand conventions
- **No new dependencies** without updating `project_plan.md`
- **Tests required** for new modules (scorer, suggester, matcher, app)
- **Config-driven** — scoring weights live in `rubric_config.json`, not hardcoded
- **Schema-first** — all modules read/write `CVSchema`
- **Commit discipline** — only commit when asked; use descriptive messages
- **No code pushes** to GitHub without explicit user approval
- **No comments** in code unless necessary for clarity
- **Mark progress** in `TODO.md` and `progress.md` after completing tasks
