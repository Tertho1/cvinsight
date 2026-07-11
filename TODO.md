# TODO — CV Evaluator & Ranking System

Last updated: 2026-07-11 | Git: `d519cda` (v0.3) | Tests: 343 passing | Phase 3 complete

---

## Current Priority

**Week 5: Classifier Training & Streamlit V1** — Train Logistic Regression + XGBoost on `labeled_cvs.csv`, build Streamlit web app, deploy to Hugging Face Spaces.

---

## Active Tasks

### Phase 2 — Dataset Normalization Adapters ✅

- [x] **Create `src/extractor/adapters.py`** with 4 adapter functions
- [x] **`adapt_netsol(row)`** — maps NETSOL JSON keys to normalized sections dict (normalizes `end_date`→dates.end, `degree_title`→degree.level, `university`→institution.name)
- [x] **`adapt_ner(row)`** — passes raw text through `section_splitter`
- [x] **`adapt_ats(row)`** — passes raw text through `section_splitter`
- [x] **`adapt_classification(row)`** — passes raw text through `section_splitter`
- [x] **Create `scripts/batch_extract_all.py`** — incremental save every 500 CVs
- [ ] **Full batch extraction** (long-running: ~26k CVs, run overnight with `python scripts/batch_extract_all.py`)

### Phase 3 — Text-Path Rewrites ✅

- [x] **`_parse_experience_text`**: Title + description extraction, YYYY-YYYY/MM/YYYY date support, "at"/","/"-" title/company parsing
- [x] **`_parse_education_text`**: Paragraph-level splitting, cross-line field association, NER ORG + keyword institution detection
- [x] **`extract_projects` text path**: Tools extracted via `skill_extractor.extract_skills()` on description
- [x] **`extract_languages` text path**: Known language name list, `Language (Proficiency)` and `Language, Proficiency` format parsing

### Week 5 — Classifier Training & Streamlit V1

- [ ] Train Logistic Regression baseline → `models/lr_baseline.pkl`
- [ ] Train XGBoost classifier → `models/xgb_classifier.pkl`
- [ ] `app/app.py` — Streamlit V1: upload CV → pipeline → score + label + suggestions
- [ ] Polish UI: color coding, section breakdown, JSON download
- [ ] Deploy to Hugging Face Spaces
- [ ] Tag v1.0

### Week 6 — JD Matching & Ranking (V2)

- [ ] `src/matcher/embedder.py` — sentence-transformer embedding
- [ ] `src/matcher/semantic_scorer.py` — cosine similarity
- [ ] `src/matcher/skill_overlap.py` — skill overlap + missing skills
- [ ] `src/matcher/ranker.py` — ranking formula
- [ ] Evaluate Spearman correlation ≥ 0.65
- [ ] Add V2 ranking tab to Streamlit
- [ ] Tag v2.0

### Week 7 — Fine-Tuning & Final Report (stretch)

- [ ] Fine-tune spaCy NER on Mehyaar dataset
- [ ] Retrain XGBoost with fine-tuned features
- [ ] Full evaluation metrics (NER F1, classifier F1, NDCG@5, Spearman ρ)
- [ ] Prepare demo/ folder with test CVs + JDs
- [ ] Final git push, tag v3.0

---

## Recently Completed

### Week 1 — Foundation & Datasets ✅
- [x] GitHub repo + virtualenv + dependencies
- [x] All 5 datasets downloaded to `data/raw/`
- [x] All 5 datasets cleaned in `data/processed/`
- [x] `config/rubric_config.json` + `config/skill_taxonomy.json`
- [x] `src/schema.py` + `src/schema_validator.py`
- [x] Tag v0.1

### Week 2 — CV Parser ✅
- [x] `src/parser/pdf_parser.py` — 3-strategy PDF extraction
- [x] `src/parser/docx_parser.py` + `src/parser/txt_parser.py`
- [x] `src/parser/ocr_parser.py` — pytesseract fallback
- [x] `src/parser/parser.py` — unified `parse_cv()`
- [x] `src/parser/section_splitter.py` — 80+ heading aliases
- [x] `src/parser/cleaner.py` — 5-step cleaning pipeline
- [x] 133 parser tests, 95% success on 20 real CVs
- [x] Tag v0.2

### Week 3 — Information Extraction ✅ (~90%)
- [x] Contact extractor (email, phone, LinkedIn)
- [x] Skill extractor (PhraseMatcher on taxonomy)
- [x] Education extractor (degree, institution, year, GPA)
- [x] Experience extractor (title, company, dates, duration)
- [x] Misc extractor (projects, certifications, languages, etc.)
- [x] `src/extractor/extractor.py` — master orchestrator
- [x] `src/extractor/utils.py` — structured parsing helpers
- [x] 285 tests (14 files), 3000 CVs extracted
- [x] Extraction coverage: name/email/phone 94%, education 99.9%, experience 100%, skills 99.8%, projects 99.2%

### Week 4 — Scoring Engine ✅ (~95%)
- [x] `src/scorer/section_scorers.py` — 7 scoring functions
- [x] `src/scorer/scorer.py` — `score_cv()` batch scoring
- [x] `src/scorer/feature_builder.py` — feature vector builder
- [x] `src/suggester/suggester.py` — suggestion generator
- [x] 51 scorer + suggester tests
- [x] Config restructured to match code expectations
- [x] Pipeline verified end-to-end
- [x] `labeled_cvs.csv` generated (4500 CVs scored)
- [x] `score_distribution.png` generated
- [x] `borderline_review.csv` generated (1382 borderline CVs)
- [x] Rubric weights adjusted (experience bands expanded, projects 8pts/project, skills target=10, label thresholds: Strong 72+)

---

## Known Issues

### Fixed in Phase 1 ✅

1. ~~**Languages trapped in skills column**~~ — `skills.languages` sub-key now routes to languages field in `extract_all()`
2. ~~**`try_parse_structured` ignores tuples**~~ — Tuple → list conversion added, fixes datasetmaster certs
3. ~~**`try_parse_structured` drops plain string list items**~~ — Non-JSON strings are now preserved in list results
4. ~~**NETSOL key name mismatches**~~ — `title` fallback for projects, `degree_title`/`university` fallback for education
5. ~~**Date format "Till Date" not matched**~~ — "Till Date" added to `_PRESENT` set and `_DATE_RANGE_RE`

### Moderate (Phase 2 or later)

6. **Text-based fallbacks assume newlines** — ATS/classification texts are single-line
7. **datasetmaster_clean text column is raw JSON** — Fixed by Phase 2 adapters
8. **Certification coverage 0.1%** — datasetmaster has only 9 non-null certification rows
9. **SyntaxWarning `"\/"` in test** — Harmless
10. **Python 3.14 vs 3.10** — All packages work

### Data Quality
11. **No Strong CVs before config tweak** — Label thresholds adjusted to 72+/50-71/0-49 to match data reality
12. **Borderline review needed** — 1382 CVs flagged; manually review 50+ for calibration
