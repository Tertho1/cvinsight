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
├── tests/                # 482 unit tests (23 files, all passing)
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
- 482 tests across 23 files (parser: 133, extractor: ~285, scorer+suggester+criteria: 61, bangla: 33, hybrid/classifier/ner-skill: ~28)

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
  **Replaced 2026-08-08** by the hybrid v3 (`models/classifier_v3_hybrid_synth.pkl`,
  XGBoost + engineered features + DistilBERT semantic embedding; 7/10 benchmark agree
  vs 4/10, score-level Spearman +0.758; `load_classifier()` prefers it, XGB fallback).
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
  weight 0.0); **default settled 2026-08-05 = 0.0** (`scripts/eval_bm25_hybrid.py`: any
  nonzero BM25 lowers ρ + NDCG@10; BM25 stays a pool pre-filter via `score_corpus`).
  Learning-to-rank probed 2026-08-05 and **rejected** (`scripts/train_ranker_ltr.py`:
  XGBoost `rank:ndcg`, test NDCG@10 0.6816 vs pure semantic 0.7805 — auxiliary/lexical
  feats dilute the ConFit signal; pure semantic stays the ranker). ConFit-style contrastive
  fine-tune done 2026-08-05 (`scripts/train_matcher_confit.py` → `models/matcher-confit`,
  ATS ρ 0.314→0.436, **adopted as the default embedder** in `embedder.py`); the training
  script + dataset-eval remain available for future re-runs.
  App-start eager warm-up done 2026-08-05 (`warm_up()` + app `preload_matcher()`);
  `resume-job-description-fit` dataset eval done (`scripts/eval_resume_jd_fit.py`,
  confit binary-fit ρ 0.332 vs base 0.216). NETSOL cross-check done (`scripts/eval_netsol_crosscheck.py`, confit ρ 0.345 vs base 0.329).
  Multi-CV ranking tab shipped in the app ("🏆 Ranking") 2026-08-05.
- **LLM backend re-integrated 2026-08-08**: fine-tuned Qwen3-0.6B LoRA restored as the deeper of **two** 
  app extraction modes: **`spaCy + DistilBERT NER`** (default fast tier: spaCy/rule pipeline + DistilBERT
  `models/ner-v1` fused, ~40-90ms/CV) and **`spaCy + Qwen3 LoRA LLM`** (adds Qwen3 LoRA fusion).
  `src/extractor/hybrid.py` exposes `extract_with_llm()` (default `adapter` fixed; lazy heavy imports;
  empty-dict graceful fallback) + `fuse()`. App: device selectbox (auto/gpu/cpu) + model loaded once via
  `load_llm_model` `@st.cache_resource`. GPU ~27-32s/CV. `tests/test_hybrid.py` (+6) and
  `tests/test_ner_skill_filter.py` (+11); suite 428.
  The NER skill merge was hardened 2026-08-08: `_skill_parts()` splits chained spans, strips punctuation,
  drops URL/email/location junk while preserving real `.js` skills (e.g. Vue.js, D3.js); measured skill-adds
  +49 noisy → +21 clean.
  Watch out: a stray `grp.py` in the temp working dir shadowed stdlib `grp` (torch→tarfile) and caused a
  bogus "partially initialized module 'torch'" — always run from `D:\Projects\cvinsight` (not a temp cwd)
  when loading torch.
- Bangla CV support research done 2026-08-05 (`docs/research_bangla_cv_support.md`):
  corrects the plan (no public labeled Bangla *resume-NER* exists; B-NER is generic,
  celloscopeai name-only, Onneshon section-only). Recommend Phase 2 translate route
  (IndicTransv2 → existing `extract_all()`); native Phase 3 gated on demand. §9 reviewed
  ready-made HF Bangla NER models (mBERT-wikiann F1 0.971 names/orgs; arafatfahim/BanglaTag
  adds DATE/ORG/INST but F1 0.749, news-domain, no SKILL/DEGREE) — none are resume-domain;
  they only serve as name/org/date support or bootstrap labelers.
- Bangla section classifier built 2026-08-05 (Onneshon): `scripts/train_bangla_section_classifier.py`
  → `models/bangla_section_classifier.pkl` (char-ngram TF-IDF + LR, 5-fold CV acc 0.9454);
  loader `src/extractor/bangla_section.py` (lazy singleton, Onneshon→CVSchema section map,
  graceful None); 13 tests (`tests/test_bangla_section.py`), suite 410. Section detection only —
  not entity extraction. **Wired into `extract_all()` (2026-08-08)** as the Bangla route's sectioning
  fallback (see Bangla native route below).
- **Bangla native route shipped 2026-08-08** (`src/extractor/bangla_extractor.py`):
  `extract_all()` detects Bengali script (`is_bangla`, U+0980–U+09FF ≥10% + ≥3 chars) and routes to
  `extract_bangla()` — transliterates Bengali digits (০-৯→0-9), months (জানুয়ারি→January), date
  markers (বর্তমান→present), degree words (স্নাতকোত্তর→Master, বিএসসি→B.Sc), spoken languages
  (বাংলা→Bengali) and section headings (দক্ষতা→TECHNICAL SKILLS) so the existing English extractors
  fire; Latin tech terms, emails and phones pass through untouched. `language: "bangla"` on the CVSchema
  (new field, default "en"). App shows a "Language: Bangla" badge and skips English DistilBERT-NER +
  hybrid-ML classifier for Bengali CVs. Full extract→score→suggest verified (score 60/Average).
  +17 tests (`tests/test_bangla_extractor.py`); suite 467.
  Hardened 2026-08-09 for real CVs (`demo/banglacv1.txt` 22→54, `demo/banglacv2.txt` 6→26): more section-heading
  phrasings, dotted degrees, job titles, skills, institution words, BD phone normalization, dash-format language
  pairs; +3 regression tests → suite **482**. App skips the Qwen3 LoRA step for Bengali CVs (English-only LLM).
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
