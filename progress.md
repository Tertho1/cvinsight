# CV Evaluator & Ranking System — Progress Report

**Generated:** July 11, 2026  
**Git:** `11bca6e` (Phase 3: text-path rewrites + Phase 2 adapters + population analysis)  
**Python:** 3.14.3  
**Overall Completion:** ~77%

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
- **285 unit tests** (14 test files) — all passing (see tests/)
- **3000 CVs extracted** to `data/processed/extracted_cvs.json`

**⚠️ Correction — Languages ARE in datasetmaster (inside skills column)**

The datasetmaster `skills` column has a `"languages"` sub-key with entries like:
```python
{"technical": {...}, "languages": [{"name":"English","level":"native"}, ...]}
```
99.8% of rows have languages here. The 0% coverage is a **pipeline bug**: `extract_all()` calls `extract_languages(sections.get("languages", ""))` but there's no separate `languages` column, so it always gets empty string. Meanwhile `_extract_skills_from_section()` puts these language names into the **skills list** instead. Languages field stays empty despite rich data existing.

**Extraction quality (3000 CVs from datasetmaster/resumes):**

| Field          | Coverage | Notes |
|----------------|----------|-------|
| Name           | 94.4%    | Real names extracted |
| Email          | 94.1%    | Real emails extracted |
| Phone          | 94.1%    | Real phone numbers extracted |
| Education      | 99.9%    | Real degrees, institutions, years |
| Experience     | 100.0%   | Real titles & companies, duration months computed |
| Skills         | 99.8%    | Clean lists, but includes language names (bug) |
| Projects       | 99.2%    | Real project names, tools, descriptions |
| Certifications | 0.1%     | Comma-separated JSON objects parse as tuples → lost |
| Languages      | 0.0%     | **Bug:** stuck inside skills column, pipeline never reads them |

---

## 2. What's Missing (Not Started)

### Week 4 — Scoring Engine & Label Generation ✅ (~95%)
- `src/scorer/section_scorers.py` — 7 section scoring functions, reads weights from `rubric_config.json`
- `src/scorer/scorer.py` — `score_cv()` master scorer with `score_cvs()` batch mode + `reload_config()`
- `src/scorer/feature_builder.py` — `build_features(cv_schema) → np.array` with 12 numeric features
- `src/suggester/suggester.py` — `generate_suggestions()` with config-driven thresholds, capped at 5 tips
- `config/rubric_config.json` — restructured: flat layout, all code-expected keys present, `borderline_bands` added, expanded degree mappings
- **51 unit tests** (test_scorer.py + test_suggester.py) — all passing  
- **Total: 343 tests** across 16 files — all passing
- **Pipeline verified end-to-end**: extract_all → score_cv → generate_suggestions (day28 script)
- `scripts/clean_datasets.py` & 8 scripts — fixed `structured_resumes_clean.csv` → `datasetmaster_clean.csv`
- **Phase 1 bug fixes ✅** — all 6 extractor bugs verified fixed (tuple handling, plain string items, languages routing, NETSOL key fallbacks, "Till Date" support)
- **Batch scoring completed** — 4500 CVs scored → `labeled_cvs.csv` + `score_distribution.png` generated
- **Borderline CVs flagged** — 1382 CVs in `borderline_review.csv` (bands 45-51, 68-74)
- **Rubric weights adjusted** — experience bands expanded (7 tiers), projects 8pts/project, skills target=10, label thresholds: Strong 72+
- **Distribution now:** Strong 24.6%, Average 73.7%, Weak 1.7% (mean 66.3, std 7.4)
- **Remaining:** Manual review of 50 borderline CVs (Day 26), then proceed to Phase 2

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

### Extraction Quality — datasetmaster only

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| Languages stuck inside skills column | **High** | `extract_all()` must extract `skills.languages` sub-key and route to languages field |
| Certifications in tuple format | **Medium** | Comma-separated JSON `{...},{...}` parses as tuple; `try_parse_structured` doesn't handle tuples |
| Education year parsing | **Minor** | Some years appear as "20" instead of "2020". Structured parser handles correctly; text fallback still has edge cases. |
| SyntaxWarning `"\/"` in test | **Low** | Pre-existing warning in `test_extractor.py`, does not affect functionality. |

### Cross-Dataset Extractor Compatibility (5 datasets analyzed)

The extractors were designed primarily for datasetmaster's format. Analysis of all 5 datasets revealed:

| Dataset | Skills | Education | Experience | Projects | Certs | Languages | Verdict |
|---------|--------|-----------|------------|----------|-------|-----------|---------|
| **datasetmaster_raw** (structured cols) | ✅ works | ✅ works | ✅ works | ✅ works | ⚠️ tuple format | ❌ stuck in skills col | **Pipeline fix needed** |
| **datasetmaster_clean** (text col) | ❌ garbage | ❌ garbage | ❌ garbage | ❌ garbage | ❌ | ❌ | Text col is raw JSON/Python repr, not natural language |
| **ats_scores** | ✅ OK | ⚠️ partial | ❌ no newlines | ⚠️ partial | ⚠️ | ❌ | Skills OK, rest broken |
| **classification** | ✅ OK | ⚠️ partial | ❌ no date ranges | ⚠️ partial | ⚠️ | ❌ noisy | Skills OK, rest limited |
| **ner_resumes** | ✅ OK | ✅ OK | ❌ date "Till Date" | ⚠️ partial | ⚠️ | ❌ catastrophically noisy | Skills/education OK |
| **netsol** | ✅ JSON array | ❌ key mismatch | N/A empty | ❌ `title` vs `name` | N/A | ❌ | Silent data loss on 3 fields |

### Critical Bugs Found

| # | Bug | Affected | Impact |
|---|-----|----------|--------|
| 1 | `try_parse_structured` ignores tuples | datasetmaster certifications | 9 CVs' certs silently lost |
| 2 | `try_parse_structured` drops plain string items in lists | NETSOL achievements | All achievements silently lost |
| 3 | Languages extracted from `skills.languages` but put in skills list | All datasetmaster CVs | 99.8% of CVs have 0 languages despite data existing |
| 4 | NETSOL uses `title` not `name` for projects | NETSOL projects | All projects silently lost |
| 5 | NETSOL uses `degree_title`/`university` not `degree.level`/`institution.name` | NETSOL education | All education silently lost |
| 6 | `_DATE_RANGE_RE` doesn't match "Till Date" | ner_resumes experience | Experience extraction fails on this dataset |
| 7 | Text-based fallbacks assume newlines exist | ATS, classification, ner_resumes | Experience/languages/projects extractors break |
| 8 | `datasetmaster_clean` text column is raw JSON/Python repr | datasetmaster_clean | All extractors produce garbage on text column |

### Architecture Recommendations

1. **Fix language pipeline first** — Languages exist in `skills.languages` sub-key, just need proper routing in `extract_all()`.

2. **Make extractors resilient to key name variations** — NETSOL uses different keys (`title` vs `name`, `degree_title` vs `degree.level`). Add fallback key lookups.

3. **Handle tuples in `try_parse_structured`** — Add `isinstance(parsed, tuple)` → convert to list.

4. **Handle plain string items in lists** — Keep non-JSON-object strings as-is instead of dropping them.

5. **Add "till date" to experience date regex** — Minor fix for ner_resumes compatibility.

6. **Dataset schedule:** For now, only `datasetmaster_raw` (structured columns) is fully production-ready. Other datasets need extractor fixes before they can be used at scale.

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
| W3 | NER extraction | ✅ ~95% | 20% | ~19% |
| W4 | Scoring engine + Phase 1b | ✅ ~95% | 20% | ~19% |
| P2 | Dataset adapters (cross-dataset) | ✅ 100% | — | +8% |
| W5 | Classifier + Streamlit V1 | ❌ 0% | 15% | 0% |
| W6 | JD matching & ranking | ❌ 0% | 10% | 0% |
| W7 | Fine-tuning & final report | ❌ 0% | 5% | 0% |
| P3 | Text-path rewrites | ✅ 100% | — | +5% |
| **Total** | | | **100%** | **~77%** |

**Execution progress (3-phase plan):**

1. **Phase 1** — ✅ Fix 6 extractor bugs (already applied before this session) → batch-score 4500 CVs → `labeled_cvs.csv` → `borderline_review.csv` → rubric weights adjusted (24.6% Strong, 73.7% Average, 1.7% Weak)
2. **Phase 2** — ✅ Create `src/extractor/adapters.py` with 4 adapters (netsol, ner, ats, classification) + `scripts/batch_extract_all.py` for incremental batch extraction
3. **Phase 3** — ✅ Rewrite text-path extractors: experience (title + description + YYYY-YYYY/MM/YYYY dates), education (paragraph-level + cross-line association), projects (tools from description via skill_extractor), languages (name detection + `Language (Proficiency)` parsing)

**Next: Week 5 — Classifier Training & Streamlit V1**
