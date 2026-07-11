# TODO — CV Evaluator & Ranking System

Last updated: 2026-07-11 | Git: `b3bff3a` (v0.2) | Tests: 285 passing

---

## Current Priority

**Build the Scoring Engine (Week 4)** — This is the critical bottleneck. Everything downstream (classifier, Streamlit app, suggester, matcher) depends on `score_cv()` working.

---

## Active Tasks

### Week 4 — Scoring Engine & Label Generation (0% → target 100%)

- [x] `src/scorer/section_scorers.py` — 7 scoring functions reading weights from `rubric_config.json`
- [x] `src/scorer/scorer.py` with `score_cv(cv_schema) → scored CVSchema`
- [x] `src/scorer/feature_builder.py` — `build_features(cv_schema) → np.array`
- [x] `src/suggester/suggester.py` — `generate_suggestions(cv_schema) → list[str]`
- [x] Unit tests for scorer and suggester (51 tests)
- [ ] Run scorer on all 3000 extracted CVs → inspect score distribution
- [ ] Generate pseudo-labels → `labeled_cvs.csv`
- [ ] Flag and manually review 50 borderline CVs
- [ ] Weekly review, git push, tag v0.3

### Week 5 — Classifier Training & Streamlit V1 (0% → target 100%)

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

---

## Known Issues

- Certifications and languages have ~0% coverage in datasetmaster dataset (not stored in source data)
- `SyntaxWarning: invalid escape sequence "\/"` in skill_taxonomy regex — needs raw strings
- No scorer/suggester/matcher tests exist yet (modules not built)
- Python 3.14 used instead of project-plan 3.10 — all packages work but watch for API changes
