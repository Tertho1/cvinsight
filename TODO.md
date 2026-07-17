# TODO — CV-Insight

Last updated: 2026-07-18 | App: `streamlit run app/app.py` | Tests: 361 passing

---

## Current Priority — Adopted from V2 Proposals

These items were selected from the V2 platform proposals (archived in `project_plan.md` Appendix A) as the highest-ROI improvements for our current codebase:

- [ ] **Dynamic `criteria_scores`** — replace fixed 7-section rubric with a configurable criteria list loaded from `config/default_criteria.json`. Each criterion has independent weight, `method` tag, and `rationale`. Scorer reads from config, not hardcoded keys.
- [ ] **Score rationales** — every criterion score includes a human-readable rationale string. Objective scores use template strings (e.g. "8 years experience → 10/10"), no LLM needed.
- [ ] **Schema update** — `section_scores` → `criteria_scores` list; add `method`, `rationale`, `overridden_by` to each entry; rename `jd_match` → `match`
- [ ] **Multi-CV comparison view** — simple side-by-side table in Personal Mode when multiple CVs uploaded, same categories as rows

### Future (Deferred — see project_plan.md Appendix A)

- [ ] Company Mode (batch upload, criteria builder, results board, CSV export)
- [ ] Async batch processing for 100+ CVs
- [ ] User auth + data isolation (Personal vs. Company accounts)
- [ ] Optional LLM scorer for contextual criteria (achievement quality, leadership)
- [ ] Fairness warning system
- [ ] UI rewrite beyond Streamlit

---

## Recent Improvements (2026-07-18)

- [x] **`src/matcher/embedder.py`** — lazy-loaded `multi-qa-MiniLM-L6-cos-v1`, added `embed_texts()` batch method
- [x] **`src/matcher/semantic_scorer.py`** — cosine similarity with empty-text guards
- [x] **`src/matcher/skill_overlap.py`** — refactored to reuse `extract_skills()` from skill_extractor, no duplicate taxonomy logic
- [x] **`src/matcher/ranker.py`** — added `_cv_to_text()` fallback, `match_cv()` accepts `cv=` dict
- [x] **18 matcher tests** — all passing
- [x] **`notebooks/matching_eval.ipynb`** — Spearman ρ evaluation vs HF `0xnbk/resume-ats-score-v1-en`
- [x] **JD matching in app** — JD text area beside upload, match % KPI card, `JD Match` tab with matched/missing skills
- [x] **Theme-aware colors** — brighter purple/green/amber/red for light+dark contrast, no hardcoded white backgrounds
- [x] **Native upload box** — `st.file_uploader` styled directly as dashed clickable box via CSS pseudo-elements (icon + text), no overlay hack
- [x] **JD form wrapping** — `st.form` with Match button; auto-matches on initial upload, re-matches on form submit
- [x] **Equal column widths** — `st.columns(2)` for symmetric upload/JD layout
- [x] **Fixed upload box height (255px)** — matches JD form height, no resize on file upload
- [x] **History dedup** — prevents duplicate entries on rerun
- [x] **Hidden native elements** — label, Upload button span, drag-drop instructions, file chips all hidden
- [x] **All skills displayed** — scrollable text_area instead of truncated first-15
- [x] **Borders** — 2–3px rgba(128,128,128,0.35) for visibility on both themes

## Current Priority

**Week 6 — JD Matching & Ranking (V2)**
Add job description matching, semantic similarity, skill overlap, and CV ranking to the Streamlit app.

**Deployment:** Streamlit Community Cloud (https://cvinsight-io.streamlit.app) — **running**
- Text PDF, DOCX, TXT — working
- Scanned/image-based PDFs — crashes (easyocr + torch exceed 1GB RAM)
- 1GB RAM insufficient for torch + easyocr in subprocess during scanned PDF OCR
- Fix: migrate to Docker host (Render.com/Railway) for system-package OCR, or drop easyocr

---

## LLM-Based Extraction (Custom Fine-Tuning Phase)

Hardware: RTX 5070 Ti (local inference & training).

### ✅ Completed — Custom LoRA Fine-Tuning on Qwen3-0.6B

- [x] **`scripts/generate_training_dataset.py`** — converts 4,612 datasetmaster CVs → Qwen3 chat JSONL
  - Reconstructs natural resume text from structured JSON sections
  - Runs `extract_all()` to get CVSchema as ground truth
  - Filters out CVs without valid names (167/4779 skipped)
- [x] **`scripts/train_llm.py`** — PEFT LoRA training on Qwen3-0.6B (10M trainable params, 1.67% of total)
  - Qwen/Qwen3-0.6B base, LoRA rank=16, BF16, batch_size=4, grad_accum=4
  - Trained 2 epochs (552 steps) in ~2 hours on RTX 5070 Ti
  - Best eval loss: **0.499** (epoch 1). Slight overfit by epoch 2 (eval loss 0.505)
  - Model saved to `models/qwen3-0.6b-cv-lora/`
- [x] **Side-by-side eval on 9 demo CVs**
  - **Strengths:** Extracts all major CVSchema fields (name, email, phone, skills, education, experience, projects, languages); got Barry Allen name right (both rule-based + sandeeppanem failed)
  - **Issues:** Experience entries duplicated in output; skill duplicates inherited from training data; JSON truncated on longer CVs (needs compact output format); ~42s/CV inference
  - **Verdict:** Pipeline works, needs data quality improvements for production use

### 🔜 Later — Fine-Tuning Improvements (Post V1)

- [ ] **Deduplicate training data** — collapse near-duplicate experience entries, deduplicate skills per CV
- [ ] **Compact JSON format** — train with `indent=None` instead of `indent=2` to save ~60% output tokens
- [ ] **Add more datasets** — include NER (3.3k), ATS (5k), Classification (12k) for diversity
- [ ] **Retrain** — expect better quality with cleaner data
- [ ] **Integrate LLM extractor** into pipeline as optional backend

---

## ✅ Completed — Week 5: ML Text Classifier + Streamlit V1

- [x] **`scripts/vectorize_cvs.py`** — TF-IDF vectorization on 4,612 CVs, LR + XGBoost training
- [x] **Logistic Regression baseline** — 85.81% accuracy, 0.8642 weighted F1
- [x] **XGBoost classifier** — **87.65% accuracy, 0.8754 weighted F1** (best model saved)
- [x] **`scripts/build_classifier_data.py`** — rebuilds training data with reconstructed resume text + rubric labels
- [x] **`app/app.py`** — Streamlit V1: upload → parse → extract → score → classify → suggest with side-by-side rubric vs ML comparison

### NLP Techniques Used

| Technique | Where | Implementation |
|-----------|-------|----------------|
| NER | Extractor | spaCy EntityRuler + PhraseMatcher |
| Information extraction | Extractor | Rule-based section parsers |
| Keyword extraction | Extractor | Skill PhraseMatcher on taxonomy |
| **Text classification** | **Classifier (Week 5)** | **TF-IDF + XGBoost on raw CV text** |
| Semantic similarity | Matcher (Week 6) | sentence-transformers all-MiniLM-L6-v2 |
| Ranking model | Matcher (Week 6) | Weighted formula: 0.5×semantic + 0.3×skill + 0.2×rubric |

---

- [x] **DOCX table-aware parsing** — Put each table cell on its own line so section_splitter detects headings
- [x] **Experience title/company swap** — Detect when company name is treated as title on PDFs
- [x] **Education institution filtering** — Exclude degree-name ORG entities (e.g. "Bachelor of Science")
- [x] **Languages: skip tech categories** — Filter "Frameworks:", "Tools:", "Databases:" lines
- [x] **Phone: Indian number pattern** — Add `\d{5}[-.\s]?\d{5}` for Indian phone formats
- [x] **PDF: clean (cid:127) markers** — Remove `(cid:127)` from PDF text output
- [x] **Run `test_cv_files.py` on demo/** — All 9 CVs improved; 343 tests passing
- [x] **Evaluate `sandeeppanem/qwen3-0.6b-resume-json` LoRA** — Profile summarizer, not detailed extractor
- [x] **Evaluate `nimendraai/NuExtract-tiny-Resume-Data-Extractor`** — Basic fields only, hallucinates experience

### Phase 2 — Dataset Normalization Adapters ✅

- [x] **Create `src/extractor/adapters.py`** with 4 adapter functions
- [x] **`adapt_netsol(row)`** — maps NETSOL JSON keys to normalized sections dict
- [x] **`adapt_ner(row)`**, **`adapt_ats(row)`**, **`adapt_classification(row)`**
- [x] **Create `scripts/batch_extract_all.py`** — incremental save every 500 CVs
- [ ] **Full batch extraction** (~26k CVs, run overnight)

### Phase 3 — Text-Path Rewrites ✅

- [x] All text-path extractors rewritten (experience, education, projects, languages)

### Bangla/Bengali CV Support — Three-Phase Plan

- [ ] **Phase 1 (now)**: Fine-tune Qwen3-0.6B on English CVs only
- [ ] **Phase 2**: Collect 200-500 Bangla CVs + translate → fine-tune mixed model
- [ ] **Phase 3**: Native Bangla extraction via Onneshon/B-NER datasets

### Week 6 — JD Matching & Ranking (V2) ✅

- [x] `src/matcher/embedder.py` — sentence-transformer embedder
- [x] `src/matcher/semantic_scorer.py` — cosine similarity CV vs JD
- [x] `src/matcher/skill_overlap.py` — skill overlap + missing skills
- [x] `src/matcher/ranker.py` — ranking formula with _cv_to_text fallback
- [x] Streamlit integration — JD text area, match KPI, JD Match tab
- [x] 18 tests, evaluation notebook

### Week 7 — Fine-Tuning & Final Report

- [ ] Custom LoRA fine-tune Qwen3-0.6B on improved training data
- [ ] Bangla CV support evaluation
- [ ] Side-by-side eval: rule-based vs fine-tuned LLM
- [ ] Side-by-side eval: rubric classifier vs TF-IDF+XGBoost classifier
- [ ] Full evaluation metrics (NER F1, classifier F1, Spearman ρ, NDCG@5)
- [ ] Multi-CV ranking tab in Streamlit
- [ ] Final report

---

## Archived: V2 Proposals (Future Scaling)

Full proposals archived in `project_plan.md` Appendix A. Summary of decisions:

| Proposal | Decision | Notes |
|----------|----------|-------|
| Two-Mode Platform (Personal + Company) | ❌ Deferred | Needs auth system, DB, async queue, ATS UI |
| Hybrid Scoring (Objective + LLM) | 🔶 Partial | Adopted criteria_scores + rationales; LLM scorer deferred |
| Schema v2 (criteria_scores) | ✅ Adopted | See Current Priority above |
| Self-Hosted Track (spaCy+XGBoost) | ✅ Already done | Our current pipeline matches their V3 spec |
| Evaluation Metrics v2 | 📋 Reference | Aspirational targets for future Company Mode |
| Score Rationales (Auditability) | ✅ Adopted | See Current Priority above |
| Fairness Warning System | ❌ Skipped | Out of scope |
| Async Batch Processing | ❌ Deferred | With Company Mode |
| OCR Fallback Trigger | ✅ Already done | < 50 char threshold implemented |
| Multi-CV Side-by-Side Comparison | 📋 TODO | See Current Priority above |

---

## Completed — UI Overhaul (2026-07-17)

- [x] **Branded header** — document icon + "CV-Insight" title + "AI-Powered CV Scorer & Job Description Matcher" subtitle
- [x] **Styled upload area** — dashed-border container with cloud icon, side-by-side tips column with best-results checklist
- [x] **Pipeline visualization** — 5 horizontal step boxes (Parse → Extract → Score → Classify → Suggest) with arrow connectors
- [x] **Custom rubric weights** — sliders per section in sidebar, custom config passed to `score_cv()` via tempfile
- [x] **Section mini-cards** — color-coded score cards with progress bars, green/yellow/red by percentage
- [x] **Key strengths card** — auto-extracted from CV (skills, experience, education, projects, certifications, languages)
- [x] **Structured data tables** — pandas DataFrames for experience/education/projects with columns
- [x] **Session history** — last 5 results in sidebar + dedicated tab, Clear All button
- [x] **Processing stage indicators** — `st.status()` with step-by-step progress, collapses on completion
- [x] **Top-right Clear All** — resets session state

## Bug Fixes (2026-07-16)

- [x] **easyocr crash #1**: `_preprocess_for_ocr()` returned PIL Image — easyocr's `readtext()` only accepts numpy array
- [x] **easyocr crash #2**: `detail=1` returns `(bbox, text, conf)` tuples, code destructured as `(text, conf)` → `TypeError`
- [x] **Preprocessing**: Removed hard binarization, softer autocontrast + contrast + sharpen only
- [x] **`_fix_easyocr_errors()`**: Added digit context patterns — O→0 between digits, l→1 at number endings
- [x] **Comparison test**: tesseract (1973 chars) vs easyocr (1900 chars) on `demo/ocrtest.pdf`
- [x] **All 343 tests passing** locally

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

## Possible Extraction Improvements (From NuExtract Comparison)

Insights from testing `NuExtract-tiny` on 9 demo CVs revealed areas where our rule-based system could still improve:

### 1. Experience: Location Leakage into Duration
Both our extractor and NuExtract struggle with locations appearing in the duration field. E.g., "Pune" appears as duration for Ananya's entry, "New York" for Mathew's.

**Fix ideas:** Add city/state/country gazetteer to filter known location words from parsed duration strings. Could use `spaCy` GPE entities on the raw duration text.

### 2. Education: Company-Like Institution Names
NuExtract sometimes treats "Computer Information Systems" as an institution name (Mathew's CV). Our extractor handles this better but could still improve by:
- Cross-referencing detected institution against a university gazetteer
- Using NER label confidence scores to filter non-ORG entities

### 3. Skill Deduplication
Both systems produce duplicate skills. Our extractor is better but still gets occasional duplicates from multi-section scanning.

**Fix:** Add a `deduplicate_skills()` post-processing step that normalizes casing, removes near-duplicates (e.g. "React.js" vs "React"), and preserves the first occurrence.

### 4. Phone Normalization
NuExtract produces cleaner formatting (e.g. "+91 87654 32109" with consistent spacing). Our extractor can return raw formats.

**Fix:** Add `normalize_phone()` utility to standardize Indian and US phone formats consistently.

### 5. Name Extraction Edge Cases
NuExtract failed completely on Barry Allen (empty name), while our extractor gets it right. But our extractor still struggles on certain edge cases (pro-cv-template gets "Email" as name).

**Fix:** Add name-confidence heuristic — if extracted name looks like a section header or common word, fall back to alternative extraction strategies (e.g., first line of resume before any heading).

### 6. Cross-Field Hallucination Detection (Inspiration from NuExtract's failure mode)
NuExtract produces hallucinated experience entries. Our system doesn't hallucinate, but we could add:
- **Cross-field consistency check:** Detect if experience company is a location word (city name) → flag for review
- **Duration sanity check:** If duration contains location words or is unreasonably long (>20 years for junior role), flag

### 7. JSON Output Reliability (Applies to LLM Pipeline)
NuExtract failed to produce valid JSON for the burgundy DOCX (truncated output).

**For LLM pipeline:** Add a robust JSON extraction fallback using regex (find outermost `{...}`) and `json5` parser for lenient parsing of single-quoted strings and trailing commas.

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

### Deployment (Streamlit Cloud) — CRITICAL
13. **1 GB RAM insufficient** — torch (CUDA, ~800MB) + easyocr + spaCy + XGBoost + matplotlib exceed limit. App crashes on startup with OOM (no traceback, generic "Oh no" error, "connection reset by peer" on health check). Fix: CPU-only torch not viable — changes uv dependency tree, breaks package resolution. Need Docker host (Render.com/Railway) or drop easyocr from Cloud deployment.
14. **CPU-only torch pin breaks app** — `--extra-index-url https://download.pytorch.org/whl/cpu` with pinned `torch==2.13.0`/`torchvision==0.28.0` causes uv resolver to downgrade packages (numpy 2.5.1→2.4.4, certifi 2026.6.17→2022.12.7), breaking app startup.
15. **easyocr crash bugs fixed locally** (343 tests passing) — PIL→numpy conversion, tuple destructuring, preprocessing improvements. Working on local machine but cannot verify on Cloud due to OOM.
16. **Recommended migration**: Render.com (512MB RAM, 750h/mo free, Dockerfile supports `apt-get tesseract-ocr` + `poppler-utils`) or Railway.app.

### Data Quality
11. **No Strong CVs before config tweak** — Label thresholds adjusted to 72+/50-71/0-49 to match data reality
12. **Borderline review needed** — 1382 CVs flagged; manually review 50+ for calibration
