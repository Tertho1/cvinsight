# CV Evaluator & Ranking System — Progress Report

**Generated:** July 10, 2026  
**Git:** `v0.2-1-g2b63120` (1 commit ahead of v0.2, no v1.0 tag yet)  
**Python:** 3.14.3  
**Overall Completion:** ~40%

---

## 1. What's Implemented (Working)

### Week 1 — Foundation & Dataset Setup ✅ (100%)
- GitHub repo initialized with README
- Python virtualenv with all dependencies installed (spaCy, XGBoost, sentence-transformers, Streamlit, etc.)
- All 5 datasets downloaded to `data/raw/`
- All 5 datasets cleaned and saved to `data/processed/`
- `config/rubric_config.json` — scoring weights
- `config/skill_taxonomy.json` — ~270 skills across 8 categories (was ~146)
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
- **~133 unit tests** (6 test files) — all passing
- Tested on 20 real CVs — 95% pass rate (19/20)
- Git tag `v0.2`

### Week 3 — Information Extraction (NER) ⚠️ (70% — fixed)
- `src/extractor/extractor.py` — Master extractor with structured + text fallback
- `src/extractor/contact_extractor.py` — Reads from `personal_info` JSON column + text fallback with aggressive false-positive filtering
- `src/extractor/skill_extractor.py` — PhraseMatcher on skill taxonomy
- `src/extractor/education_extractor.py` — JSON-aware: parses degree level, institution, year, GPA
- `src/extractor/experience_extractor.py` — JSON-aware: parses title, company, dates, duration, responsibilities
- `src/extractor/misc_extractor.py` — JSON-aware: projects, certifications, languages, achievements, leadership
- `src/extractor/utils.py` — **New:** Centralized `try_parse_structured()` handling Python repr + implicit string concatenation
- 30 CVs extracted to `data/processed/extracted_cvs.json`

**Extraction quality (after fixes):**

| Field        | Coverage | Notes |
|--------------|----------|-------|
| Name         | 13.3%    | Dataset stores "Unknown" for 26/30 records; correctly returns empty |
| Email        | 6.7%     | Dataset has "Unknown" in most records |
| Phone        | 6.7%     | Same data limitation |
| Education    | 96.7%    | Real degrees (ME, B.E., HSC, SSC), real institutions, correct years |
| Experience   | 100.0%   | Real titles & companies, duration months computed |
| Skills       | 96.7%    | Clean lists, no "unknown" or escaped slash artifacts |
| Projects     | 86.7%    | Real project names, tools, descriptions |
| Certifications | 0%     | No data in this dataset |
| Languages    | 0%       | No data in this dataset |

---

## 2. What's Missing (Not Started)

### Week 4 — Scoring Engine & Label Generation ❌ (0%)
- `src/scorer/section_scorers.py` — 7 section scoring functions
- `src/scorer/scorer.py` with `score_cv()` — master scoring function
- `src/scorer/feature_builder.py` — feature vector for classifier
- `labeled_cvs.csv` — 200+ labeled CVs
- Score distribution analysis
- Borderline CV manual review
- `src/suggester/suggester.py` with `generate_suggestions()`

### Week 5 — Classifier Training & Streamlit V1 ❌ (0%)
- `models/lr_baseline.pkl` — Logistic Regression baseline
- `models/xgb_classifier.pkl` — XGBoost classifier
- `app/app.py` — **Directory is empty**, no Streamlit code at all
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
- No v3.0 or v2.1 tag

---

## 3. Issues & Areas for Improvement

### Extraction Quality

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| Name/email/phone have low coverage | **Data issue** | Use **noran-mohamed/Resume-Classification-Dataset** or **Mehyaar/Annotated_NER_PDF_Resumes** which have real plain-text CVs with real names/contact info. The `datasetmaster` dataset stores "Unknown" for most contact fields — this is not an extraction bug. |
| Education year parsing | **Minor** | Some years appear as "20" instead of "2020" due to regex matching "20" from "2016". The structured parser handles this correctly now, but the text fallback still has this issue. |
| Experience start/end dates | **Medium** | The dataset stores dates as "Unknown" for many records, resulting in 0 duration_months. The structured parser handles this correctly where dates exist. |

### Code Quality

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| No extractor unit tests | **High** | There are 133 parser tests but **zero tests for any extractor module**. Need `tests/test_contact_extractor.py`, `tests/test_education_extractor.py`, etc. |
| Extractor quality metrics not validated | **Medium** | The F1=1.0 reported in week3_extraction_eval.ipynb is unreliable — it was computed against the same data the extractor read from. Need a proper manual ground-truth comparison. |
| SyntaxWarnings in output | **Low** | `invalid escape sequence "\/"` warnings appear in the skill_taxonomy regex matching. Should use raw strings in regex patterns. |
| Python 3.14 compatibility | **Low** | Project plan specifies 3.10 but the env is 3.14. All tested packages work, but some API changes may exist (e.g., `datetime.now()` deprecation path). |

### Architecture Recommendations

1. **Add `personal_info` to `section_cols` in the notebook** — The extraction notebook passes `sections` to `extract_all()` but doesn't include the `personal_info` column. The fix is already in the extractor code (it accepts `contacts=sections`), but the notebook needs updating: `sections["personal_info"] = str(row.get("personal_info", ""))`.

2. **Test on plain-text CVs** — All current testing uses the `datasetmaster/resumes` (JSON-structured) dataset. The extractors also support plain-text parsing via section_splitter → text-based extraction, but this path hasn't been tested. Run on `ner_resumes_clean.csv` (which has real plain-text CVs) to validate.

3. **Build the scoring engine next** — Scoring is the critical dependency for Weeks 4-7. The scorer reads `rubric_config.json` and populates `section_scores`, `total_score`, and `label`. Without it, the classifier, suggester, and app all stall.

4. **Streamlit app should be minimal V1 first** — Start with: file upload → call pipeline → display score + label + suggestions. No JD matching, no ranking, no multi-CV upload. Get the core loop working end-to-end.

---

## 4. Quick-Start Commands

```bash
# Run tests
pytest tests/

# Run extraction on 30 CVs
python -c "import sys; sys.path.insert(0, '.'); import pandas as pd; from src.extractor.extractor import extract_all; df = pd.read_csv('data/processed/structured_resumes_clean.csv').head(30); [extract_all(str(r.text), sections={c: str(r.get(c,'')) for c in ['education','experience','skills','projects','certifications','languages','achievements','leadership','personal_info']}) for _, r in df.iterrows()]"

# Launch Streamlit (when app.py exists)
streamlit run app/app.py
```

---

## 5. Summary

| Week | Area | Status | Weight | Contribution |
|------|------|--------|--------|-------------|
| W1 | Foundation & datasets | ✅ 100% | 15% | 15% |
| W2 | Parser (PDF/DOCX/TXT/OCR) | ✅ 100% | 15% | 15% |
| W3 | NER extraction | ⚠️ ~70% | 20% | ~14% |
| W4 | Scoring engine | ❌ 0% | 20% | 0% |
| W5 | Classifier + Streamlit V1 | ❌ 0% | 15% | 0% |
| W6 | JD matching & ranking | ❌ 0% | 10% | 0% |
| W7 | Fine-tuning & final report | ❌ 0% | 5% | 0% |
| **Total** | | | **100%** | **~44%** |

**Next milestone:** Week 4 — Scoring engine (`scorer.py`, `section_scorers.py`). This is the hard dependency for everything that follows.
