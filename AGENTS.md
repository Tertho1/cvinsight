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
├── demo/                 # Sample CVs for demo + demo/benchmark/ (scenario CVs)
│   └── benchmark/         # 10 controlled scenario CVs + manifest.json + _baseline.json
├── docs/                 # Documentation (index: docs/README.md)
│   ├── extraction_audit.md        # Audit findings, fixes, prioritized improvements
│   ├── extraction_improvements.md # Full improvement list with effort tiers
│   ├── research_ner_hybrid_extraction.md  # NER research review
│   └── research_text_reformatting.md      # Text normalization research review
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
│   ├── extractor/        # NER + rule-based extraction (✅ ~95%, audit fixes in 2026-08-04)
│   │   └── adapters.py   # Dataset normalization adapters (Phase 2)
│   ├── scorer/           # Rubric scoring engine (✅ done)
│   ├── matcher/          # JD similarity & ranking (✅ done, Week 6)
│   └── suggester/        # Suggestion generation (✅ done)
├── tests/                # 361 unit tests (16 files, all passing)
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
Pydantic model with nested models (Education, Experience, Project, Certification, Language, SectionScores, CriterionScore, JDMatch). All modules read/write this structure. Fields: `cv_id`, `name`, `email`, `phone`, `education`, `experience`, `skills`, `projects`, `certifications`, `languages`, `achievements`, `leadership`, `criteria_scores`, `total_score`, `label`, `suggestions`, `match` (legacy `jd_match` accepted on load).

### Testing
- Framework: **pytest** (plain asserts, no unittest.TestCase)
- Run all: `pytest tests/`
- 387 tests across 18 files (parser: 133, extractor: 285, scorer+suggester+criteria: 61)

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

5. **Extraction Audit & Benchmark** (✅ 2026-08-04)
   - Root cause fixed: DOCX paragraphs join with single `\n` → multi-job
     experience collapsed to 1 entry; refactored to whole-section date-anchored
     parsing (`_parse_experience_text`, `_find_all_dates`, `_looks_like_job_header`)
   - Comma-first `_parse_title_company` (fixes "University of Texas at Austin")
   - Rubric degree-key gap fixed (`M.Sc`/`M.Tech`/`M.A`/`M.E`/`B.A`/`B.E`)
   - `demo/benchmark/` — 10 reproducible scenario CVs + baseline (46.4 → 53.6)
   - Docs: `docs/extraction_audit.md`, `docs/extraction_improvements.md`
   - 361 tests passing; demo mean 53.8 unchanged
   - Follow-on hardenings (ORG-FP span repair, project titles, academic/skill
     aliases, date-first headers): benchmark mean **56.4**, demo mean **56.2**

6. **Schema v2 — Criteria Scores + Rationales** (✅ 2026-08-04)
   - `config/default_criteria.json` — configurable criterion list (weight,
     `method` tag, rationale) instead of the hardcoded 7-section loop
   - `src/scorer/scorer.py` writes `criteria_scores` (name/score/max_points/
     weight/method/rationale/overridden_by) + legacy `section_scores` dict;
     weights derive from rubric cap so the app's custom-weights stay consistent
   - `src/schema.py`: `CriterionScore` model; `jd_match` → `match`
     (legacy `jd_match` still accepted on load; `jd_match` property alias kept)
   - `app/app.py` shows rationale under each section card; DB reads tolerate
     both `match` and `jd_match`
   - 387 tests passing; benchmark 56.4 / demo 56.2 unchanged

### ✅ Completed (2026-08-04) — Week 5/6/7

- **Week 5 — ML Text Classifier + Streamlit V1**: TF-IDF → XGBoost (0.8765 acc /
  0.8754 F1) beats LR baseline (0.8581 / 0.8642); `app/app.py` upload → extract →
  score → classify → suggest. Classifier deployed (`models/xgb_classifier.pkl`).
- **Week 6 — JD Matching & Ranking (V2)**: `src/matcher/` (embedder, semantic_scorer,
  skill_overlap, ranker); ranking tab + JD-match in app. 18 matcher tests.
- **Week 7 — Fine-Tuning & Final Report**: side-by-side evals (rule vs grounded LLM
  mean 65.6 vs 53.6, 10/10; classifier LR vs XGBoost), NER `models/ner-v1`
  (in-domain F1 0.998), full metrics → `models/week7_metrics.json` (NDCG@5 0.98,
  Spearman ρ 0.314), matcher fixes A/B/C/D, `docs/final_report.md` +
  `docs/matcher_datasets_latency.md`.

### 🔜 Backlog (research/deploy, not core Week tasks)

- Matcher research backlog — see `TODO.md` "Matcher improvement pipeline":
  nothing pending there now; remaining backlog is app-side (multi-CV ranking tab).
  E (hybrid BM25) implemented 2026-08-05 (`src/matcher/bm25_scorer.py`, opt-in, default
  weight 0.0); quantifying a nonzero hybrid default is pending. ConFit-style contrastive
  fine-tune done 2026-08-05 (`scripts/train_matcher_confit.py` → `models/matcher-confit`,
  ATS ρ 0.314→0.436, **adopted as the default embedder** in `embedder.py`); the training
  script + dataset-eval remain available for future re-runs.
  App-start eager warm-up done 2026-08-05 (`warm_up()` + app `preload_matcher()`);
  `resume-job-description-fit` dataset eval done (`scripts/eval_resume_jd_fit.py`,
  confit binary-fit ρ 0.332 vs base 0.216). NETSOL cross-check done (`scripts/eval_netsol_crosscheck.py`, confit ρ 0.345 vs base 0.329).
  Multi-CV ranking tab shipped in the app ("🏆 Ranking") 2026-08-05.
- Bangla CV support (multilingual: Onneshon, B-NER, AI4Bharat Sangraha) — not built.
- Entity-level NER eval done 2026-08-05 (`scripts/eval_ner_entity_level.py`,
  in-domain span F1 0.988 vs token 0.998; real-resume spans are in-text by construction).
- Extract seqeval entity-level accuracy on real resumes (current NER F1 is in-domain).
- Deployment: Streamlit Cloud OOM → migrate to Render.com/Railway.

---

## Agent Guidelines

- **Read first:** Before editing any file, read it to understand conventions
- **No new dependencies** without updating `project_plan.md`
- **Tests required** for new modules (scorer, suggester, matcher, app)
- **Config-driven** — scoring weights live in `rubric_config.json`, not hardcoded
- **Schema-first** — all modules read/write `CVSchema`
- **Commit discipline** — only commit when explicitly asked; use descriptive messages
- **CRITICAL: No code pushes to GitHub** without explicit user approval — test locally first, demo changes in-session. User will say "push" when ready.
- **No comments** in code unless necessary for clarity
- **Mark progress** in `TODO.md` and `progress.md` after completing tasks
- **Streamlit Cloud deploy**: currently DOWN (OOM, 1GB RAM insufficient for torch + easyocr). Migrate to Render.com/Railway before attempting redeploy.
