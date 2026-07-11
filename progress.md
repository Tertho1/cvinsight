# CV Evaluator & Ranking System — Progress Report

**Generated:** July 11, 2026  
**Git:** `b3bff3a` (1 commit ahead of v0.2, no v1.0 tag yet)  
**Python:** 3.14.3  
**Overall Completion:** ~64%

---

## 1. What's Implemented (Working)

### Week 1 — Foundation & Dataset Setup ✅ (100%)
- GitHub repo initialized with README
- Python virtualenv with all dependencies installed (spaCy, XGBoost, sentence-transformers, Streamlit, etc.)
- All 5 datasets downloaded to `data/raw/`
- All 5 datasets cleaned and saved to `data/processed/`
- `config/rubric_config.json` — scoring weights
- `config/skill_taxonomy.json` — ~270 skills across 8 categories
- `src/schema.py` — `CVSchema` Pydantic model with all sub-models + helper methods
- `src/schema_validator.py` — `validate_cv()`, `validate_cv_from_json()`, `quick_check()`
- Git tag `v0.1`

### Week 2 — CV Parser & Text Extraction ✅ (100%)
- `src/parser/pdf_parser.py` — 3-strategy PDF extraction (pdfplumber → pdfminer → pypdf)
- `src/parser/docx_parser.py` — DOCX parsing with table cell extraction
- `src/parser/txt_parser.py` — Multi-encoding TXT reader (utf-8, latin-1, cp1252)
- `src/parser/ocr_parser.py` — pytesseract OCR fallback for scanned PDFs
- `src/parser/parser.py` — Unified `parse_cv()` dispatching to all format handlers
- `src/parser/section_splitter.py` — 80+ heading aliases mapped to 12 canonical sections
- `src/parser/cleaner.py` — 5-step text cleaning pipeline
- **133 unit tests** (6 test files) — all passing
- Tested on 20 real CVs — 95% pass rate (19/20)
- Git tag `v0.2`

### Week 3 — Information Extraction (NER) ✅ (~90%)
- `src/extractor/extractor.py` — Master extractor orchestrating all sub-extractors
- `src/extractor/contact_extractor.py` — Email, phone, LinkedIn extraction with aggressive false-positive filtering
- `src/extractor/skill_extractor.py` — PhraseMatcher on skill taxonomy
- `src/extractor/education_extractor.py` — Degree level, institution, year, GPA parsing
- `src/extractor/experience_extractor.py` — Title, company, dates, duration, responsibilities
- `src/extractor/misc_extractor.py` — Projects, certifications, languages, achievements, leadership
- `src/extractor/utils.py` — `try_parse_structured()`, `parse_json_field()` helpers
- **285 unit tests** (14 test files) — all passing
- **3000 CVs extracted** to `data/processed/extracted_cvs.json`

**Extraction quality (3000 CVs from datasetmaster/resumes):**

| Field          | Coverage | Notes |
|----------------|----------|-------|
| Name           | 94.4%    | Real names extracted |
| Email          | 94.1%    | Real emails extracted |
| Phone          | 94.1%    | Real phone numbers extracted |
| Education      | 99.9%    | Real degrees, institutions, years |
| Experience     | 100.0%   | Real titles & companies, duration months computed |
| Skills         | 99.8%    | Clean lists, no artifacts |
| Projects       | 99.2%    | Real project names, tools, descriptions |
| Certifications | 0.1%     | Minimal data in this dataset |
| Languages      | 0.0%     | Not stored in this dataset |

---

## 2. What's Missing (Not Started)

### Week 4 — Scoring Engine & Label Generation ✅ (~80%)
- `src/scorer/section_scorers.py` — 7 section scoring functions, reads weights from `rubric_config.json`
- `src/scorer/scorer.py` — `score_cv()` master scorer with `score_cvs()` batch mode + `reload_config()`
- `src/scorer/feature_builder.py` — `build_features(cv_schema) → np.array` with 12 numeric features
- `src/suggester/suggester.py` — `generate_suggestions()` with config-driven thresholds, capped at 5 tips
- `config/rubric_config.json` — restructured: flat layout, all code-expected keys present, `borderline_bands` added, expanded degree mappings
- **51 unit tests** (test_scorer.py + test_suggester.py) — all passing
- **Pipeline verified end-to-end**: extract_all → score_cv → generate_suggestions (day28 script)
- `scripts/clean_datasets.py` & 8 scripts — fixed `structured_resumes_clean.csv` → `datasetmaster_clean.csv`
- **Remaining:** Run scorer on 4500 CVs → `labeled_cvs.csv`, inspect distribution, flag borderline CVs

### Week 5 — Classifier Training & Streamlit V1 ❌ (0%)
- `models/lr_baseline.pkl` — Logistic Regression baseline (Week 5)
- `models/xgb_classifier.pkl` — XGBoost classifier (Week 5)
- `app/app.py` — Streamlit UI (Week 5)
- No deployment to Hugging Face Spaces

### Week 6 — JD Matching, Ranking & V2 App ❌ (0%)
- `src/matcher/embedder.py` — sentence-transformer embedder
- `src/matcher/semantic_scorer.py` — cosine similarity
- `src/matcher/skill_overlap.py` — skill overlap computation
- `src/matcher/ranker.py` — ranking formula engine
- Streamlit V2 with ranking tab

### Week 7 — Fine-Tuning, Evaluation & Final Report ❌ (0%)
- NER fine-tuning on Mehyaar dataset
- Retrained XGBoost with fine-tuned features
- Full evaluation metrics suite (NDCG@5, Spearman ρ)
- `demo/` — empty (no test CVs or demo script)
- README is a 24-line skeleton
- No v1.0 or later tags

---

## 3. Issues & Areas for Improvement

### Extraction Quality

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| Certifications/languages have 0% coverage | **Data issue** | The `datasetmaster/resumes` dataset doesn't store these fields. Test on `ner_resumes_clean.csv` or `classification_clean.csv` which may have richer content. |
| Education year parsing | **Minor** | Some years appear as "20" instead of "2020". Structured parser handles correctly; text fallback still has edge cases. |
| SyntaxWarning `"\/"` in test | **Low** | Pre-existing warning in `test_extractor.py`, does not affect functionality. |

### Code Quality

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| No scorer/suggester/matcher unit tests | **High** | Scoring, suggestion, and matching modules aren't written yet. Tests will be needed once implemented. |
| Extractor quality metrics not validated on ground truth | **Medium** | Coverage metrics are computed from the same structured data the extractor reads. Need manual ground-truth comparison on plain-text CVs. |
| Python 3.14 compatibility | **Low** | Project plan specifies 3.10 but env is 3.14. All packages work, but some API changes may exist. |

### Architecture Recommendations

1. **Build the scoring engine next** — Scoring is the critical dependency for Weeks 4-7. The scorer reads `rubric_config.json` and populates `section_scores`, `total_score`, and `label`. Without it, the classifier, suggester, and app all stall.

2. **Test on plain-text CVs** — All current testing uses the `datasetmaster/resumes` (JSON-structured) dataset. The extractors also support plain-text parsing via section_splitter → text-based extraction, but this path hasn't been tested. Run on `ner_resumes_clean.csv` (which has real plain-text CVs) to validate.

3. **Streamlit app should be minimal V1 first** — Start with: file upload → call pipeline → display score + label + suggestions. No JD matching, no ranking, no multi-CV upload. Get the core loop working end-to-end.

4. **Explore the noran-mohamed dataset** — For certifications/languages, the classification dataset may have richer semi-structured data that the extractor can handle.

---

## 4. Quick-Start Commands

```bash
# Run all tests
pytest tests/

# Run extraction on all 3000 CVs
python scripts/extract_all_cvs.py

# Launch Streamlit (when app.py exists)
streamlit run app/app.py
```

---

## 5. Summary

| Week | Area | Status | Weight | Contribution |
|------|------|--------|--------|-------------|
| W1 | Foundation & datasets | ✅ 100% | 15% | 15% |
| W2 | Parser (PDF/DOCX/TXT/OCR) | ✅ 100% | 15% | 15% |
| W3 | NER extraction | ✅ ~90% | 20% | ~18% |
| W4 | Scoring engine | ✅ ~80% | 20% | ~16% |
| W5 | Classifier + Streamlit V1 | ❌ 0% | 15% | 0% |
| W6 | JD matching & ranking | ❌ 0% | 10% | 0% |
| W7 | Fine-tuning & final report | ❌ 0% | 5% | 0% |
| **Total** | | | **100%** | **~64%** |

**Next milestone:** Run scorer on all 4500 CVs → `labeled_cvs.csv`, inspect distribution, flag borderline CVs (Day 24-25). Then move to Week 5 — Classifier training & Streamlit V1.
