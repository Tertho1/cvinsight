# TODO — CV-Insight

Last updated: 2026-08-05 | App: `streamlit run app/app.py` | Tests: 387 passing
---

## Extraction Audit (2026-08-03/04)

- [x] **Research threads** — `docs/research_ner_hybrid_extraction.md` (NER) + `docs/research_text_reformatting.md` (text normalization) saved to root
- [x] **Demo audit** — per-file per-section scores; mean 53.8; lows are placeholder/sparse content, not bugs
- [x] **Corpus scan** — `labeled_cvs.csv` p5=53; bottom 12 CVs are genuinely thin synthetic profiles
- [x] **Fix: company-on-next-line** — `experience_extractor.py` falls back to the line after the date range (table-flattened DOCX), before ORG fallback
- [x] **Fix: title separator strip** — "Web Developer -" → "Web Developer"
- [x] **Fix: duplicated-block truncation** — DOCX merged-cell duplicates now truncated at line boundary
- [x] **`docs/extraction_audit.md`** — findings + prioritized improvements (span repair, project titles, section aliases; parser-level normalization per research thread 2)
- [x] **Fix: multi-entry experience (root cause)** — DOCX paragraphs join with single `\n` (no blank lines), so old blank-line splitting collapsed 2 jobs → 1 entry. Refactored `_parse_experience_text` to whole-section date-anchored stream; added `_find_all_dates()` with overlap dedup (`_DATE_RANGE_RE`+`_YYYY_RANGE_RE` both match "Jan 2021 - Present") and `_looks_like_job_header()` bullet-line skip
- [x] **Fix: "at" false split** — `_parse_title_company` tries comma/pipe split before the `at/@|–|—|-` regex → "Teacher's Assistant, University of Texas at Austin" keeps full company
- [x] **Fix: rubric degree keys** — extractor emits `M.Sc`/`M.Tech`/`M.A`/`M.E` (also `B.A`/`B.E`) but `rubric_config.json` lacked them → master's degrees scored 0. Added keys to `degree_points`
- [x] **Benchmark CV set** — `scripts/generate_benchmark_cvs.py` → `demo/benchmark/` (10 scenarios, one per failure mode, + `manifest.json` + `_baseline.json`). Baseline mean 46.4 → 53.6 after fixes
- [x] **Verification** — 361 tests pass; original demo mean unchanged (53.8); benchmark entries/counts now correct (01:2, 03:2, 05:3, 06:3, 08:3)
- [ ] **Tier 1 next:** span repair for experience title/company (Rebecca "Canvas", srbhr "Front-End" ORG FPs); project title extraction; academic section aliases (invited talks/conferences); fix 04 duplicated-cells skills=7

---

## LLM Extraction (added 2026-08-03)

- [x] **LLM extraction path** — Qwen3-0.6B LoRA fine-tune done (`scripts/train_llm.py`)
- [x] **Curated fine-tune data** — `scripts/curate_dataset.py` →
  `data/processed/curated_{train,val,test}.jsonl` (3,928/345/345), canonical degrees, edge cases, leakage-safe split
- [x] **`scripts/train_llm.py`** — curated split + **masked JSON-only loss**, rsLoRA, NEFTune, CPU fallback, checkpoint/`--resume`, `--resume-from`
- [x] **CUDA env** — `torch 2.11.0+cu128` (+torchvision 0.26.0) on RTX 5070 Ti; peft↔transformers 5.8 shim
- [x] **`scripts/gate_llm_vs_rules.py`** — v2 (1-epoch) beats rules on **10/10 demo CVs** (mean 68 vs 53.6); uses `json-repair`
- [x] **Full 3-epoch run (v3) overfits** — demo mean drops to 60, 5/10; keep **v2 (1-epoch)** as model
- [x] **Veracity audit + grounding filter** — `scripts/audit_llm_verity.py` + `src/extractor/grounding.py`;
  drops invented skills (burg-8, resume_02-8, resume_03-15 dropped); LLM still beats rules 10/10 with grounded
  skills (mean 65.6 vs 53.6)
- [x] **seqeval entity-level NER eval** — `scripts/eval_ner_entity_level.py` →
  `models/ner_entity_level.json`. Hand-rolled span-level (seqeval-equivalent) P/R/F1 on the
  in-domain test split: **P=0.981 R=0.995 F1=0.988** (token-level 0.998 modestly overstates
  entity accuracy). Real resumes: 183 spans across 20 demo/benchmark files, all in-text by
  construction (7 strict-verbatim misses are join/tokenization artifacts, not hallucination).
- [x] **Span-NER comparison (scan 2026-08-04)** — trained `models/ner-v1` (distilbert, ~13s/37s train),
  gated vs rules/LLM. NER skills ≈ rules/LLM coverage (10.8 vs 12.9/10.1) but low total (18.6) —
  no dates/relations. Kept as fast CPU `Rule + NER` app option; **LLM hybrid set aside** (CPU too slow).
- [x] **Small-model research (2026-08-04)** — encoder NER is the only CPU-real-time family; generative
  small LLMs (0.5–1.7B) are ~10–60s/CV. Best external candidate: `oksomu/resume-ner` (65M, ~15ms,
  13 resume labels). Our fine-tunes **`models/ner-v1` already covers this schema**.
- [x] **Streamlit UI**: rule-based default + `Rule + NER (fast)` option; LLM hybrid unplugged
- [ ] **Bangla CV support** via fine-tune/multilingual (Week 7 left)

## Current Priority — Adopted from V2 Proposals

These items were selected from the V2 platform proposals (archived in `project_plan.md` Appendix A) as the highest-ROI improvements for our current codebase:

- [x] **Dynamic `criteria_scores`** — replace fixed 7-section rubric with a configurable criteria list loaded from `config/default_criteria.json`. Each criterion has independent weight, `method` tag, and `rationale`. Scorer reads from config, not hardcoded keys.
- [x] **Score rationales** — every criterion score includes a human-readable rationale string. Objective scores use template strings (e.g. "1 roles totalling 5.0 years of experience → 18/25"), no LLM needed.
- [x] **Schema update** — `section_scores` → `criteria_scores` list; add `method`, `rationale`, `overridden_by` to each entry; rename `jd_match` → `match` (legacy `jd_match` still accepted on load for old saved DBs)
- [x] **Multi-CV comparison view** — simple side-by-side table in Personal Mode when multiple CVs uploaded, same categories as rows

### Future (Deferred — see project_plan.md Appendix A)

- [ ] Company Mode (batch upload, criteria builder, results board, CSV export)
- [ ] Async batch processing for 100+ CVs
- [ ] User auth + data isolation (Personal vs. Company accounts)
- [ ] Optional LLM scorer for contextual criteria (achievement quality, leadership)
- [ ] Fairness warning system
- [ ] UI rewrite beyond Streamlit

---

## Recent Improvements (2026-07-24)

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
- [x] **Batch upload** — `accept_multiple_files=True` on uploader, progress bar, serial processing loop
- [x] **Persistent CV database** — `data/processed/cv_database.json`, auto-save after each analysis, load on startup
- [x] **Comparison table** — "All CVs" tab with sortable DataFrame (Name, Score, Label, Skills, Experience, Education)
- [x] **Skill search** — "Skill Search" tab with AND/OR modes, matches against extracted skills across all stored CVs
- [x] **CV selection** — click-to-view-detail from comparison table or skill search results, loads into existing KPI/section/tab UI
- [x] **Cross-session persistence** — CV database survives page reloads; Clear Database button in sidebar

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

- [x] **Side-by-side eval: rule-based vs fine-tuned LLM** — `models/gate_v2_vs_rules.json`: grounded LLM mean 65.6 vs rules 53.6, wins 10/10 demo CVs (~27.6s/CV CPU; kept as batch option, not default)
- [x] **Side-by-side eval: rubric classifier vs TF-IDF+XGBoost classifier** — `classifier_comparison.csv`: XGBoost 0.8765 acc / 0.8754 F1 vs LR 0.8581 / 0.8642, both above majority baseline (0.7346 / 0.6222)
- [x] **Full evaluation metrics** — `scripts/week7_eval.py` → `models/week7_metrics.json`: NER token F1 0.998 (in-domain), NDCG@5 0.98 (benchmark)
- [x] **Matcher measurement fixed (A)** — recompute Spearman live: vs human label ρ=0.288 (embedder-independent) + vs jina ats_score ρ=0.267 (reference only); NDCG@5 stays the headline ranking metric
- [x] **Matcher bug fixes (B)** — `skill_overlap`: empty JD → 0.0, non-empty JD with no taxonomy skills → neutral 0.5 (was: silent 1.0 full credit); `_cv_to_text`: weighted structured text (experience/education/projects + skills) instead of flat list
- [x] **`docs/final_report.md`** — consolidated Week 7 report
- [ ] Custom LoRA fine-tune Qwen3-0.6B on improved training data (v2 kept; v3 overfits — see LLM section)
- [ ] Bangla CV support evaluation
- [x] **Multi-CV ranking tab** — `app/app.py` "🏆 Ranking" tab: ranks all stored CVs
  against the current JD via `rank_cvs()`, showing Match % / semantic / skill overlap /
  rubric scores, the weight breakdown, and a jump-to-CV picker. Verified end-to-end with
  rank_cvs on the DB entry format. (TODO.md:203 original item.)

### Matcher improvement pipeline (after A+B)

- [x] **C: Section-level embedding** — added `score_sections()` / `score_sections_cv_dict()` + `match_cv(mode="section")`. **Finding:** ties whole-doc on the benchmark (NDCG@5 0.98) and is unavailable on the ATS dataset (its resumes have no section headings, so it falls back to whole-doc; identical ρ=0.259). On clean demo CVs it can LOWER the score when skills are inline (no detected skills section). Net: not an improvement yet — keep whole-doc default; revisit once a structured-CV ranking gold set exists.
- [x] **D: Embedder upgrade + learned weights** — default embedder now `BAAI/bge-small-en-v1.5` (env `CV_EMBEDDER` overrides). Measured on ATS human labels: MiniLM ρ=0.259 → **bge-small ρ=0.348** (bge-base 0.260, slower). Full-suite semantic-vs-human ρ now 0.314 (was 0.288). Ranker weights configurable via `weights=` dict / `CV_RANK_WEIGHTS` env; `scripts/learn_ranker_weights.py` fits semantic:skill ratio → **best_w=1.00** (pure semantic ρ=0.338 vs 0.5-blend 0.075 on ATS: skill-overlap hurts on unstructured ATS text; keep as a hint, not a hard default)
- [x] **E: Hybrid BM25 + semantic** — `src/matcher/bm25_scorer.py` (hand-rolled Okapi, no new dep). Exposed as an **opt-in 4th ranker signal** (default weight 0.0, so 0.5/0.3/0.2 behaviour unchanged); `score_corpus()` does pool-wide pre-filtering. JD as query / CV as doc with stopword-filtered tokens; single-pair score normalized to [0,1]. Enable via `weights={"bm25": ...}` or 4-part `CV_RANK_WEIGHTS="sem;skill;rub;bm25"`. +10 tests (35 matcher / 397 total passing). Next: quantify a bm25+semantic hybrid against pure semantic on ATS/benchmark before adopting a nonzero default.
- [x] **ConFit-style contrastive fine-tune of bge-small** — `scripts/train_matcher_confit.py`
  fine-tunes `BAAI/bge-small-en-v1.5` with MultipleNegativesRankingLoss on the 6,241
  human-labeled `cnamuangtoun/resume-job-description-fit` train pairs (resume anchor /
  matching-JD positive; in-batch JDs are hard negatives). Saved to `models/matcher-confit`.
  **Results:** held-out test binary-fit Spearman ρ 0.216 → **0.332** (+54% rel); our ATS
  human-label ρ **0.314 → 0.436** (+39% rel); NDCG@5 **0.985** (no regression, benchmark
  ceiling). **Adopted as the default embedder** (`src/matcher/embedder.py`
  `_DEFAULT_MODEL="models/matcher-confit"`) — same per-pair CPU latency as bge-small
  (~35 ms/pair), so no speed cost. Roll back anytime with `CV_EMBEDDER=BAAI/bge-small-en-v1.5`.
- [x] **App-start eager warm-up** of the embedder — `warm_up()` added to `src/matcher/embedder.py`;
  `app/app.py` `preload_matcher()` (`@st.cache_resource`) loads matcher-confit once at app start
  so the first JD match skips the ~10s cold start (idempotent; app compiles clean).
- [x] **NEW dataset eval:** `cnamuangtoun/resume-job-description-fit` (6,241 human-labeled
  resume–JD pairs, MIT) as a second matching benchmark — `scripts/eval_resume_jd_fit.py` →
  `models/resume_jd_fit_eval.json`. Held-out test (1,759 pairs): binary-fit Spearman w/ matcher-confit
  **0.332** (base bge-small 0.216); sampled retrieval NDCG@10 0.309 (base 0.296). Cross-confirms the
  ConFit gain on a human-labeled set independent of our ATS data.
- [x] **NETSOL cross-check** of bge-small + learned weights — `scripts/eval_netsol_crosscheck.py` →
  `models/netsol_crosscheck.json` (849 real candidate-JD pairs, numeric score). Independent
  validation: matcher-confit pure-semantic ρ **0.345** vs bge-small 0.329; learned
  semantic-dominant (1.0) beats the hand 0.5 blend (0.345 vs 0.329). The confit/learned-weight
  advantage is not an ATS-only fluke (smaller absolute gap here, as expected: continuous
  generated score + structured resumes).

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
