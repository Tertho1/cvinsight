# CV-Insight — Progress Report

**Generated:** July 18, 2026  
**2026-08-05 note:** Bangla CV support research compiled — `docs/research_bangla_cv_support.md`  

## 2026-08-08 (evening) — App fixes: skill search + experience duration

- **Skill Search fixed**: skills stored as chained spans (e.g. `React.js, Next.js, Vue.js`) failed
  exact-set matching, so `vue.js`/`next.js`/`shopify` returned nothing even though they were visible
  in the Extracted Data tab. Added `src/extractor/skill_extractor.expand_skill_set()` — keeps each
  full entry plus comma/slash/`&`/`|`/`and`/`or`-separated parts (`+` deliberately excluded so `C++`
  stays one skill); Skill Search tab now queries through it. Verified with Streamlit AppTest:
  `vue.js` → Ananya Patel; `postgresql` → Vikram, Rahul. `tests/test_skill_extractor.py` +8 → 23.
- **Experience Duration column fixed**: entries store `duration_months` but the Extracted Data table
  read a non-existent `duration` key, so Duration was always blank. `app.py` now has `format_duration()`
  (`48` → `4y`, `19` → `1y 7m`) and `render_table()` gained a per-column `formatters` hook. Verified in
  AppTest: Duration `4y` / `1y 7m` for Ananya's entries.  
- **Skills area no longer editable/stale**: the fixed-key `text_area` kept the previous CV's (edited) value.
  Replaced with read-only skill **chips** rendered fresh each run — switching CV in the Results picker now
  updates name + chips immediately (AppTest: RAHUL 20 chips → ANANYA 16, `Vue.js` visible).
- **Upload reprocessing fixed**: `file_uploader` retains every uploaded file, so after any Clear the old
  files (still attached to the widget) re-triggered full processing even though DB was empty. Both Clear
  buttons now bump an `uploader_epoch` that is used in the uploader's widget `key`, forcing Streamlit to
  remount the widget and drop its held files; processing also skips any file whose content-id is already
  stored in the DB. “Clear All / Clear Database” now genuinely wipes everything.
- **CV switching fixed across selectors**: three widgets drove the shown CV (top `session_cv_picker`
  selectbox, History radio, Ranking/Skill "jump" points). A stale persisted widget value overrode an
  external jump on the next rerun (and the app famously bounced to the Extracted Data tab). Fix:
  `jump_to_cv(cid)` is the single jump path; all CV selectors remount via a key containing the active CV
  id (e.g. `session_cv_picker_{cur}`), and `st.tabs(..., key="main_tabs")` preserves the open tab, so a
  Skill Search hit stays on Skill Search. Verified in AppTest (multi-CV `git` → 3 "View" buttons, click
  jumps to Rahul and the selection sticks across subsequent reruns). Suite still **436 passing**.  
- **Re-upload now re-evaluates**: previously any file whose content-id already existed in the DB was
  silently skipped. The file_uploader is now a *transient ingest queue* — after each batch it is remounted
  empty (`uploader_epoch`) and the in-batch dedup keys are cleared, so (a) adding a **new** CV evaluates
  only that one, and (b) deliberately re-uploading an **existing** CV re-runs the full pipeline and
  overwrites its DB entry (timestamp updates, same content-id preserved). Verified end-to-end in AppTest
  on `demo/senior_python_dev.txt` (identical re-upload → `timestamp` refreshed, score re-computed).
- **Skill Search now shows unmatched skills**: the results table gained an **Unmatched Skills** column.
  Previously ANY-mode entries were hardcoded to an empty missing-set and ALL-mode showed nothing to gap,
  so a query like `python,git` hid which queried skills a candidate lacked (e.g. ANANYA matched `git`,
  missing `python`). `missing = query − matched` is now computed and displayed for every returned CV;
  ALL mode still returns only full matches (no gaps by definition).  

## 2026-08-08 — Rule + LLM optional backend in the app

Re-integrated the fine-tuned Qwen3-0.6B LoRA as the deeper of **two** extraction modes in the Streamlit app:
**`spaCy + DistilBERT NER`** (default fast tier) and **`spaCy + Qwen3 LoRA LLM`** (deeper, slower).

- **`src/extractor/hybrid.py`**: `extract_with_llm(raw_text, ..., device="auto")` with a corrected
  default `adapter="models/qwen3-0.6b-cv-lora-v2"`; lazy torch/peft/transformers import (fast path
  never pays the load); empty-dict on any failure so the caller falls back cleanly.
  `fuse(rules_cv, llm_cv)` merges per-field (skills = grounded union; experience/edu prefer the source
  with more dated entries; leadership/achievements rule-only; dedup on shared entries).
- **`app/app.py`**: two extraction modes. The **fast default** now always fuses the spaCy/rule pipeline
  with the fine-tuned DistilBERT tagger (`merge_skills`/`extract_education_gaps`, ~40-90ms/CV). The
  **Qwen LoRA mode** additionally runs the LLM and `fuse()`s it on top; device selectbox (auto/gpu/cpu);
  `load_llm_model` wrapped in `@st.cache_resource` so the model loads once per session, not per CV.
- **NER skill hardening (2026-08-08)**: `_skill_parts()` decomposes the tagger's noisy comm-joined/URL/
  geo spans; drops junk while preserving real `.js` skills (Vue.js, D3.js). Side-by-side on 10 demo CVs:
  skill-adds went **+49 noisy → +21 clean**, remaining appends are genuine off-taxonomy skills the rules
  missed (e.g. Webpack, UX, Express.js). Education gaps stayed +0 everywhere (rules already catch
  degree/institution).
- **Measured on RTX 5070 Ti:** Qwen3 load ~5-30s once (cached), then **~27-32s/CV** pure generation
  (greedy, ~1700-2100 output tokens). Verified on junior_dev.txt (rules 1 exp → fused 2) and
  `demo/priya_dwivedi_repo_MathewElliot.docx` (fused 11 skills, name correct).
- **tests**: `tests/test_hybrid.py` (+6, LLM graceful degradation + fuse policy) and
  `tests/test_ner_skill_filter.py` (+11, cleaned skill merge). Suite now **428**.
- **Env pitfall fixed:** a stray `grp.py` left in the temp working dir shadowed stdlib `grp` (imported by
  torch→tarfile), causing a bogus "partially initialized module 'torch'" circular import and a
  datasets-dill pickle crash. Removed; loads fine from the workspace root. This was the reason the
  smoke test appeared blocked by a library bug, not a real dependency issue.  
**Git:** https://github.com/Tertho1/cvinsight  
**Deployment:** Streamlit Community Cloud previously at https://cvinsight-io.streamlit.app — **currently DOWN (OOM, 1GB RAM insufficient for torch + easyocr)**; needs migration to Render.com/Railway. See TODO "Deployment" + `docs/final_report.md` §5.  
**Python:** 3.14.6 (Cloud) / 3.14.3 (local)  

---

## 2026-08-05 — Extraction hardening + matcher LTR probe

- **04 duplicated-cells company edge fixed** — header-scoped ORG fallback in
  `experience_extractor.py` now requires a **whole-word** match instead of a bare substring,
  so spacy's "Develop" ORG span (a fragment of "Developer" from a bullet) no longer becomes
  the company. `demo/benchmark/04_duplicated_cells.docx` now recovers "Brightpath Design".
  +1 regression test (30 in `test_experience_extractor.py`); **suite 411 passing**; benchmark
  mean unchanged **56.4**.
- **Learning-to-rank probed, rejected** — `scripts/train_ranker_ltr.py`: XGBoost `rank:ndcg`
  over 6 feats {semantic, skill, bm25, token-iou, len_cv, len_jd}, qid=JD, on the
  human-labeled resume-job-description-fit set. Test NDCG@10: hand blend 0.7401, pure
  semantic (ConFit) 0.7805, LTR 0.6816 — early stop fired at round 1. Same pattern as
  BM25: auxiliary/lexical features dilute the ConFit embedding signal. **Not adopted**; pure
  semantic stays the ranker. Script + resumable feature cache (`models/ranker_ltr_emb_*`)
  kept for reference/re-runs.

## 2026-08-05 — Bangla section classifier (Onneshon)

- **`scripts/train_bangla_section_classifier.py`** → `models/bangla_section_classifier.pkl`
  (+ `models/bangla_section_eval.json`). Char n-gram TF-IDF + LR on `data/raw/onneshon_raw.csv`.
  - Data: 1,739 segments (Experience 823 / Skill 446 / Education 370 / Objective 100),
    347 exact-dups → 1,392 deduped (leak-safe — every duplicate text has one label).
  - Results: **5-fold CV acc 0.9454 / macro-F1 0.952**; held-out 0.922 vs majority 0.520;
    held-out F1 Education 0.949, Experience 0.928, Objective 1.000, Skill 0.874.
    Confusion is only Skill↔Experience (short tech fragments).
- **`src/extractor/bangla_section.py`** — lazy-loaded `BanglaSectionClassifier` singleton +
  `classify_section()`; maps Onneshon labels → CVSchema sections (`summary`/`experience`/
  `skills`/`education`); returns None if model/text unavailable (mirrors `embedder.py`).
- **Tests** — `tests/test_bangla_section.py`, 13 tests; full suite **410 passing** (397 + 13).
- **Scope:** section detection only, not entity extraction — a building block for native
  Bangla sectioning, not end-to-end Bangla scoring (see `docs/research_bangla_cv_support.md` §8).
  Not yet wired into `extract_all()`/`app.py`; full upload-demo integration (detection +
  route + Bangla entities/skills/rubric) is a planned `TODO.md` item, not started.

## 2026-08-05 — Bangla CV support research

- **`docs/research_bangla_cv_support.md`** — feasibility scan for the `progress.md:631`
  three-phase Bangla plan. Verifies each named resource (Onneshon / B-NER / Sangraha /
  celloscopeai) actually exists and corrects the plan's key assumption.
  - **Key correction:** B-NER gives *generic* PER/ORG/LOC entities and celloscopeai gives
    *person names only* — **no labeled Bangla resume-NER (SKILL/DEGREE/TITLE/COMPANY)
    dataset exists publicly**. Native entity extraction is not turnkey; it needs a custom
    labeled set (Onneshon is section-level only). BanNERD (ACL Findings NAACL 2025) is the
    highest-quality generic Bangla NER source; ANCHOLIK-NER covers dialects only.
  - **Models:** `sagorsarker/bangla-bert-base`, `csebuetnlp/banglabert` (+ `bnlp-toolkit`
    for normalization/tokenization/lemmatization), `csebuetnlp/banglishbert` (Latin-script
    Bengali = the "Banglish" sub-problem). Matcher side already handles Bengali today
    (`models/matcher-confit`, bge-based, no code change); easyocr (already a dependency)
    is Bengali-capable so scanned-image OCR needs nothing new.
  - **Ready-made HF Bangla NER reviewed (§9):** `sagorsarker/mbert-bengali-ner` (wikiann,
    PER/ORG/LOC, F1 0.971), `Suchandra/bengali_language_NER` (0.967), `Davlan/xlm-roberta-
    base-wikiann-ner` (multilingual 20-lang), `arafatfahim/BanglaTag` (B-NER, adds DATE/
    ORG/INST/TITLE but overall F1 0.749, news-domain). **None are resume-domain** — no
    SKILL/DEGREE/COMPANY labels; usable only as name/org/date support or bootstrap
    labelers. All are 0.1–0.3B transformers, conflicting with the CPU-only / 1GB Cloud
    constraint — a mandatory per-CV Bangla NER pass is not viable; a lazy candidate-role
    one might be. Native Phase 3 = fine-tune our own resume-NER from banglabert/bangla-bert.
  - **Recommendation:** Phase 2 (translate-to-English via IndicTransv2, then existing
    `extract_all()`) is the pragmatic first cut — touches only the parse path, low risk.
    Native (Phase 3) is a large effort (Bangla regex equivalents + a Bangla resume-NER to
    build + Banglish normalization) and should be gated on confirmed Bengali-language CV
    demand. Defer is the current reality per `docs/final_report.md:157-158`.

## 2026-08-05 — Entity-level (span) NER evaluation

- **`scripts/eval_ner_entity_level.py`** → `models/ner_entity_level.json`. Hand-rolled
  seqeval-equivalent span-level P/R/F1 (no seqeval dependency) to expose what the token
  F1 overstates.
  - **In-domain test split (345 resumes):** P=0.981 R=0.995 **F1=0.988** (vs token 0.998).
    Per type: COMPANY/INSTITUTION/PERSON/PROJECT/TITLE F1 1.000; SKILL 0.978 (the only
    soft spot, P=0.966).
  - **Real demo/benchmark resumes (20 files):** NER found 183 spans; all in-text by
    construction. 7 strict-verbatim "misses" are span-joining/tokenization artifacts
    (comma-split skill lists, case/newline splits), not hallucinated tokens.
- Docs/TODO/final_report §2.3 updated.

---

## 2026-08-05 — Multi-CV ranking tab in Streamlit

- **`app/app.py`** — added "🏆 Ranking" tab (between "JD Match" and "Skill Search"). It
  ranks every CV in the database against the current JD via `src.matcher.ranker.rank_cvs()`,
  rendering Rank / Name / Match % / Semantic / Skill Overlap / Rubric Score, a "why this
  ranking" weights breakdown (semantic/skill/rubric/bm25), and a jump-to-CV select box
  that loads the chosen candidate into the main detail view.
- Ranker ignores any pre-stored match objects (recomputes fresh from raw_text + skills),
  so rankings always reflect the live JD in the input field.
- Verified end-to-end with `rank_cvs` on the DB entry shape (Python/Django CV ranked first,
  unrelated CV last) and `app.py` compiles clean. Full suite 387 passing.

---

## 2026-08-05 — NETSOL cross-check (independent matcher validation)

- **`scripts/eval_netsol_crosscheck.py`** → `models/netsol_crosscheck.json`. 849 real
  candidate-JD pairs (numeric score 0-10; resumes reconstructed from structured fields).
  Confirms the matcher-confit / learned-weight advantage is not ATS-only:
  | Embedder | pure-semantic ρ | hand 0.5 blend ρ | learned 1.0 blend ρ |
  |---|---|---|---|
  | bge-small (base) | 0.329 | 0.324 | 0.329 |
  | **matcher-confit** | **0.345** | 0.329 | **0.345** |
  Semantic-dominant (learned) weights beat the hand 0.5 blend on both embedders. Smaller
  absolute gap than ATS is expected (continuous generated score, structured resumes).
- Docs/TODO updated.

---

## 2026-08-05 — resume-job-description-fit dataset eval (second benchmark)

- **`scripts/eval_resume_jd_fit.py`** → `models/resume_jd_fit_eval.json`. Evaluates the
  matcher on `cnamuangtoun/resume-job-description-fit` (MIT) as an independent, human-labeled
  matching set beyond our ATS data. 1,759 held-out test pairs:
  | Embedder | binary-fit Spearman ρ | retrieval NDCG@10 |
  |---|---|---|
  | bge-small (base) | 0.216 | 0.296 |
  | **matcher-confit** | **0.332** (+54% rel) | **0.309** |
  Cross-confirms the ConFit fine-tune gain on an independent, human-labeled set.
- Docs/TODO updated.

---

## 2026-08-05 — App-start eager warm-up of the matcher embedder

- **`src/matcher/embedder.py`** — added `warm_up()`: loads the embedder (matcher-confit)
  and runs one trivial encode so tokenizer/weights are baked in; idempotent (reuses the
  module-level singleton, ~0.0s on repeat).
- **`app/app.py`** — added `@st.cache_resource preload_matcher()` calling `warm_up()`,
  invoked once at startup next to `load_classifier()`. Removes the ~10s model-load cold
  start from the *first JD match* (moves it to app start, cached for the session).
- Includes a "please wait" benefit: the embedder is ready before the user reaches JD
  matching, so per-match latency stays at the measured ~35 ms/pair.
- `app.py` compiles clean.

---

## 2026-08-05 — ConFit-style contrastive fine-tune of bge-small

- **`scripts/train_matcher_confit.py`** — fine-tunes bge-small-en-v1.5 with
  `MultipleNegativesRankingLoss` on the 6,241 human-labeled train pairs of
  `cnamuangtoun/resume-job-description-fit` (MIT): resume = anchor, matching JD =
  positive; unrelated in-batch JDs act as hard negatives (ConFit Siamese idea).
  Saves to `models/matcher-confit`. 1 epoch best (2 epochs slightly overfits).
- **Results (adopted as default embedder in `embedder.py`; opt-out via `CV_EMBEDDER`):**
  | Benchmark | bge-small base | matcher-confit |
  |---|---|---|
  | resume-job-description-fit held-out binary-fit ρ | 0.216 | **0.332** (+54% rel) |
  | our ATS human-label ρ (n=5043) | 0.314 | **0.436** (+39% rel) |
  | NDCG@5 (demo/benchmark) | 0.98 | **0.985** (no regression) |
  Per-pair CPU latency identical to bge-small (~35 ms/pair) — no speed cost.
  Full suite 397 passing.
- **Note:** eval used the embedder directly on CUDA; `embedder.py` still forces
  `device="cpu"` by design (Cloud/CPU-only), which is why the earlier full-ATS run
  burned ~50% CPU and no GPU (~378s vs ~20s on GPU).
- Docs/TODO updated. E (BM25) + A/B/C/D + EBC all recorded.

---

## 2026-08-05 — Matcher E: Hybrid BM25 + semantic

- **`src/matcher/bm25_scorer.py`** — hand-rolled Okapi BM25 (standard lib only, no new
  dependency). JD as query / CV as doc, stopword-filtered tokens.
  - `score(cv, jd)` → single-pair lexical relevance normalized to [0, 1]
    (smoothed-constant IDF, JD-vocabulary coverage).
  - `score_corpus(cv_texts, jd)` → real corpus-IDF BM25 across the pool (raw, relative)
    for pre-filtering / cheap ranking.
- **Ranker integration** — 4th, **opt-in** signal: default `bm25` weight 0.0 so the
  0.5/0.3/0.2 behaviour is unchanged. Enable via `weights={"bm25": ...}` or the 4-part
  `CV_RANK_WEIGHTS="sem;skill;rub;bm25"`. Result dict gains `bm25_score`; `_default_weights`
  accepts 4 parts.
- **Tests:** +10 (35 matcher / **397 total** passing).
- **Hybrid quantified (2026-08-05)** — `scripts/eval_bm25_hybrid.py` →
  `models/bm25_hybrid_eval.json`, on the human-labeled resume-JD-fit test (1,759 pairs).
  Blend `(1-w)*semantic + w*bm25` monitonically *lowers* binary-fit Spearman ρ and
  NDCG@10 vs pure semantic for every `w>0` (ρ: 0.332→0.315→0.292→0.265; NDCG@10:
  0.309→0.300→0.246→0.213 for w=0/0.1/0.3/0.5). **Conclusion: keep default `bm25` weight
  0.0.** BM25 remains opt-in (useful for pool pre-filtering via `score_corpus`, where it
  never alters the blended score of the default ranker).

---

## 2026-08-04 — Schema v2: Criteria Scores + Rationales

- **`config/default_criteria.json`** — configurable criterion list (order,
  weight, `method` tag, rationale) replacing the hardcoded 7-section loop.
- **`src/schema.py`** — added `CriterionScore` model + `criteria_scores` list
  on `CVSchema`; renamed `jd_match` → `match` (accepts legacy `jd_match` on
  load via `AliasChoices`, kept `jd_match` property alias for compat).
- **`src/scorer/scorer.py`** — scores every criterion, derives weight from the
  rubric cap (so the app's custom-weights feature stays consistent), writes
  `criteria_scores` (name/score/max_points/weight/method/rationale/
  overridden_by) plus the legacy `section_scores` dict.
- **`src/scorer/section_scorers.py`** — added per-section rationale builders
  (no LLM; objective templates restating scoring inputs).
- **`app/app.py`** — rationale shown under each section card; JD-match reads
  tolerate both `match` and legacy `jd_match` in saved DBs.
- **380 tests passing** (+10 criteria_scores/rationale tests). Benchmark mean
  56.4 unchanged; demo per-file scores unchanged.
- **Backward compat verified:** old `cv_database.json` entries with
  `jd_match` load correctly into `match`; new output serializes as `match`.

---

## 2026-08-04 — Extraction Audit Session

- Ran full per-section demo audit: mean **53.8**; lows are placeholder dates
  (template, "20XX") and sparse CVs (junior_dev), not extraction bugs.
- Corpus scan (`labeled_cvs.csv`): p5=53, median 67; bottom CVs are genuinely
  thin synthetic profiles.
- Two web research threads completed and saved: NER-efficient-use
  (`docs/research_ner_hybrid_extraction.md`) and text reformatting for broken CVs
  (`docs/research_text_reformatting.md`). Verdict: encoder token-NER is the only
  CPU-real-time family; layout repair (reading order, DOCX tables) at parser
  level is high-ROI; generic broken-text repair is not worth it.
- **Rule-hardening fixes** in `src/extractor/experience_extractor.py`:
  company-on-next-line fallback, title separator strip, duplicated-block
  truncation. priya_dwivedi extraction now clean (title/company/dedup).
  **361 tests passing**, demo mean unchanged (score is duration-driven).
- Wrote `docs/extraction_audit.md` with findings + prioritized Tier 1/Tier 2
  improvements.
- **Multi-entry experience fix (root cause):** DOCX paragraphs are joined with a
  single `\n` (no blank lines), so the old `\n\s*\n` paragraph split collapsed
  multi-job experience to 1 entry. Refactored `_parse_experience_text` to
  process the whole section as one date-anchored stream; added `_find_all_dates()`
  with overlapping-range dedup and `_looks_like_job_header()` bullet-line skip.
- **"at" false-split fix:** `_parse_title_company` tries comma/pipe split before
  the `at/@|–|—|-` regex → "Teacher's Assistant, University of Texas at Austin"
  keeps full company (was "Austin").
- **Rubric degree-key gap:** extractor emits `M.Sc`/`M.Tech`/`M.A`/`M.E`
  (and `B.A`/`B.E`) but `rubric_config.json` only had `MSc`/`MS` etc. — master's
  degrees scored **0** for education. Added missing keys to `degree_points`.
- **Benchmark CV set:** `scripts/generate_benchmark_cvs.py` →
  `demo/benchmark/` — 10 reproducible scenario CVs (one per failure mode:
  two-column PDF, table DOCX, duplicated cells, date-first, multi-degree,
  ORG-FP trigger, sparse, strong) + `manifest.json` + `_baseline.json`.
  Baseline mean **46.4 → 53.6** after fixes; original demo mean unchanged
  (53.8); entry counts now correct (01:2, 03:2, 05:3, 06:3, 08:3).
  Later hardenings (ORG-FP span repair, project titles, academic aliases,
  date-first headers) raised benchmark mean to **56.4** and demo mean to
  **56.2** (2026-08-04).
**Overall Completion:** ~99% (code), 0% (deployment)

---

## 2026-08-04 — Week 7: Metrics + Final Report

- **`scripts/week7_eval.py`** — consolidated metric aggregation →
  `models/week7_metrics.json`. Re-reads saved artifacts (classifier CSV, LLM-vs-rules
  gate) and freshly computes NER F1 + NDCG@5.
- **Classifier:** XGBoost 0.8765 acc / 0.8754 F1 > LR 0.8581 / 0.8642 > majority
  0.7346 / 0.6222.
- **LLM vs rules:** grounded Qwen3-0.6B LoRA mean 65.6 vs rules 53.6, wins 10/10 demo CVs.
- **NER (distilbert `ner-v1`):** in-domain token P/R/F1 0.998 (caveat: synthetic corpus).
- **Matcher:** Spearman ρ 0.193 (n=500, reference benchmark); NDCG@5 0.98 on benchmark set.
- **`docs/final_report.md`** written; added to `docs/README.md` index.

---

## Matcher D — embedder upgrade + learned weights (2026-08-04)

- **Embedder:** default → `BAAI/bge-small-en-v1.5` (overridable via `CV_EMBEDDER`;
  dim read from the loaded model, not assumed 384). ATS human-label ρ: MiniLM-L6 0.259
  → **bge-small 0.348**; bge-base 0.260 (slower). Full-suite semantic-vs-human ρ
  0.288 → **0.314**.
- **Ranker weights configurable:** `match_cv(weights=...)` / `rank_cvs(weights=...)`
  and `CV_RANK_WEIGHTS` env. New `scripts/learn_ranker_weights.py` fits the semantic:skill
  ratio on ATS human labels → **best_w=1.00** (pure semantic ρ=0.338 vs 0.5-blend
  0.075). Skill-overlap is noisy on unstructured ATS text; keep the current hand weights
  as the production default and use the learned ratio as a hint for tuned deployments.
- Output: `models/ranker_weights_learned.json`.

---

## Matcher measurement + bug fixes (2026-08-04, post-report)

- **Fix A (measurement):** `scripts/week7_eval.py` matcher_metrics() now recomputes
  Spearman live instead of hardcoding 0.193. Uses the ATS dataset's **human ordinal
  label** as embedder-independent ground truth → ρ=0.288 (p=5.2e-11, n=500). The jina
  `ats_score` comparison (ρ=0.267 at the time; **0.299 after matcher D** — bge-small) is
  kept labeled as reference only. NDCG@5 = 0.98
  remains the headline ranking metric. Old 0.193 was misleading (jina-derived target).
- **Fix B (bugs):**
  - `src/matcher/skill_overlap.py` — empty JD now returns 0.0 (was silent 1.0 full
    credit); non-empty JD with no taxonomy-parseable skills returns neutral 0.5.
  - `src/matcher/ranker.py` — `_cv_to_text()` builds weighted structured text
    (experience/education/projects + skills joined) instead of the flat field list.
- **Tests:** 381 passing (added JD-no-taxonomy + empty-JD cases).
- Backlog recorded in `TODO.md` (C section-level embedding, D embedder upgrade /
  learned weights, E hybrid BM25).

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

### Phase 1c — Rule-Based Extractor Fixes ✅ (Completed 2026-07-15)

All 6 priorities fixed on 9 real-world CVs (4 PDF + 3 DOCX + 2 TXT):

| # | Fix | What Changed | Impact |
|---|-----|-------------|--------|
| 1 | **DOCX table/textbox parsing** | Table cells on separate lines; `w:txbxContent` paragraphs extracted | `priya_dwivedi_repo_MathewElliot.docx`: 1→5 sections, score 12→42; `pro-cv-template-burgundy.docx`: 1→5 sections, score 0→15 |
| 2 | **Experience title/company swap** | `_looks_like_company()`, `_looks_like_job_title()` backtrack logic | 7 reversed entries fixed across 4 PDFs |
| 3 | **Education institution filtering** | `_DEGREE_INSTITUTION_FALSE_POSITIVES` allowed for legitimate inst names containing degree phrases (e.g. "Indian Institute of Technology") | Rahul: "Bachelor of Science"→"Delhi University"; Vikram: "Mechanical Engineering"→"Anna University"; Ananya: "Bachelor of Technology..."→"VIT Pune"; Barry: no regression |
| 4 | **Languages: skip tech categories** | `_TECH_CATEGORY_HEADERS` filter; paren-match validated against `_KNOWN_LANGUAGES`; removed fallback that captured any line | senior_python_dev: 4 fake lang entries removed; Vikram: 2 removed; Ananya: `js (Basic)` removed |
| 5 | **Indian phone pattern** | `\d{5}[-.\s]?\d{5}` added to `_PHONE_RE` | Rahul, Ananya, Vikram phones detected (were empty) |
| 6 | **PDF (cid:127) cleanup** | `\(cid:\d+\)` regex in `_clean_text` | All description/project text cleaned across 3 PDFs |

**Before vs After comparison:**

| CV | Before | After | Δ |
|----|--------|-------|----|
| priya_dwivedi_repo_MathewElliot.docx | score=12, 1 section, name="JavaScri" | score=42, 5 sections, name="MATHEW ELIOT", edu=Columbia, exp=Web Developer | **+30 pts** |
| pro-cv-template-burgundy.docx | score=0, 1 section, name="Email", email mangled | score=15, 5 sections, edu=PhD, 10 achievements | **+15 pts** |
| resume_02_rahul_verma.pdf | inst="Bachelor of Science...", phone=empty, titles reversed | inst="Delhi University", phone="+91 87654 32109", titles correct | **quality** |
| resume_03_ananya_patel.pdf | inst="Bachelor of Technology...", phone=empty, titles reversed, lang="js" | inst="VIT Pune", phone="+91 76543 21098", titles correct, no fake lang | **quality** |
| resume_04_vikram_singh.pdf | inst="Mechanical Engineering", phone=empty, titles reversed, lang="Flask"+"Databases:" | inst="Anna University", phone="+91 65432 10987", titles correct, no fake lang | **quality** |
| senior_python_dev.txt | titles reversed, langs="Frameworks:...", edu extra pipe | titles correct, no fake lang, edu clean | **quality** |

---

## 2. What's Implemented (Working)

### Week 6 — JD Matching & Ranking (V2) ✅ (~100%)

All 4 matcher modules built and tested:

| Module | File | Purpose |
|--------|------|---------|
| Embedder | `src/matcher/embedder.py` | Sentence-transformer wrapper (`multi-qa-MiniLM-L6-cos-v1`, CPU, 384-dim), `embed_texts()` batch method |
| Semantic Scorer | `src/matcher/semantic_scorer.py` | Cosine similarity between CV and JD embeddings |
| Skill Overlap | `src/matcher/skill_overlap.py` | JD skill extraction via `extract_skills()`, overlap ratio + missing skills list |
| Ranker | `src/matcher/ranker.py` | Weighted final score (50% sem + 30% skill overlap + 20% rubric), `_cv_to_text()` fallback for dicts without raw_text |

**App integration:** JD text area beside CV upload (wrapped in `st.form` with Match button) → "JD Match" tab shows match %, semantic similarity, skill overlap, matched skills (left column) + missing skills (right column). Auto-matches during initial processing; form submit re-matches from cache.

**Theme:** All colors now use light+dark compatible palette (brighter purple/green/amber/red). No hardcoded white/light backgrounds. Dataframe backgrounds overridden to transparent.

**Upload area:** Native `st.file_uploader` styled directly as a dashed clickable box via CSS pseudo-elements (icon + text). No overlay, no visible button/label/chips. Fixed 255px height to match JD form.

**18 tests** in `tests/test_matcher.py` — all passing.
**`notebooks/matching_eval.ipynb`** — Spearman ρ evaluation against HF `0xnbk/resume-ats-score-v1-en` (1275 CV-JD pairs).

**Total: 380 tests** across 18 files — all passing (361 → 380 after the
criteria_scores/rationale test additions).

---

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

### easyocr Bug Fixes (2026-07-17)

Two runtime crashes fixed in `src/parser/ocr_parser.py`:

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| Crash #1 | `Invalid input type` on every page → empty result | `_preprocess_for_ocr()` returned PIL Image; easyocr `readtext()` only accepts `str(path)`, `bytes`, or `np.ndarray` | Return `np.array(img, dtype=np.uint8)` |
| Crash #2 | `TypeError` → app crash | `detail=1` returns `[(bbox, text, conf), ...]` but code did `for text, conf in results` — `conf` got the text string, then `text >= 0.3` raised TypeError | `for bbox, text, conf in results` |

**Preprocessing improved**: Removed hard binarization (easyocr CNN works better on natural gradients). Softer: grayscale → autocontrast (cutoff=3) → contrast (1.3×) → sharpen.

**Post-processing**: Added digit-context patterns to `_fix_easyocr_errors()` — `O→0` between digits, `l→1` at number endings.

**Comparison on `demo/ocrtest.pdf`** (scanned PDF, 81.6 KB):

| Metric | Tesseract | EasyOCR |
|--------|-----------|---------|
| Chars extracted | 1973 | 1900 |
| Email | `vikram.singh@email.com \| +91...` | `vikram singh@emailcom` (dots/pipe lost) |
| "Present" | `Present` | `Preseni` (keyword corrupted) |
| "September" | `Septernber` | `Sepleitbei` (garble) |
| "Career Break" | `[Career Break` | `ICareer Break` (leading I injected) |
| Line structure | Pipe-separated compound lines intact | Fragmented into separate lines |
| Key data (phone, dates, names) | All correct | Phone correct; dates mostly ok |

**Verdict**: easyOCR is usable but significantly less accurate than tesseract. Line-structure collapse is the fundamental limitation — pipe-separated compound fields fragment into independent lines, confusing the rule-based section parser. ~60% of scanned CVs will extract partially.

### UI Overhaul (2026-07-17/18)

Complete Streamlit UI rewrite in `app/app.py`:

| Feature | Before | After |
|---------|--------|-------|
| Header | Plain `st.title()` | Branded doc icon + "CV-Insight" title + purple accent subtitle |
| Upload area | Raw `st.file_uploader` | Native `st.file_uploader` styled as dashed clickable box with CSS pseudo-element icon/text, fixed 255px height, hidden label/button/chips |
| JD input | Plain text area beside upload | Wrapped in `st.form` with Match button; auto-matches on initial upload, form-submit for re-matching |
| Column layout | `st.columns([3, 2])` (upload wider) | `st.columns(2)` (equal widths, symmetric) |
| Tips | None | Side-by-side tips column with checkmark list |
| Pipeline | Text markdown list | 5 horizontal steps with icons + arrow connectors |
| Section scores | Progress bars (rows) | Color-coded mini-cards with score + thin progress bar |
| Key strengths | Not shown | Auto-extracted card (skills, experience, education, projects) |
| History | Duplicate entries on every rerun | Dedup by filename, max 5 entries |
| Clear state | Sidebar button only | Top-right "Clear All" button + sidebar "Clear History" |
| Sidebar model info | Text only | Card with green checkmark + "Online" indicator + help widget |

### Deployment Status — Scanned PDF Only Issue

**App is running**: Text PDF, DOCX, TXT all work correctly on Streamlit Cloud (https://cvinsight-io.streamlit.app).

**Problem**: Scanned/image-based PDFs crash with OOM. When easyocr is triggered in the subprocess, it loads torch (CUDA, ~800MB) which exceeds the 1GB RAM limit. The subprocess gets OOM-killed, and the error propagates up.

**Attempted fixes (didn't work):**
1. **CPU-only torch pin** → Changed uv dependency tree, downgraded packages (numpy 2.5.1→2.4.4), broke app startup on different versions
2. **Removed packages.txt** → Was installing 68 apt packages wastefully, but not the root cause
3. **Full revert** → Confirmed issue is torch memory in subprocess, not app code

**Path forward**: 
- EasyOCR fallback removed from Cloud deploy (scanned PDFs get "Could not extract text" message)
- For scanned PDF support: migrate to Docker host (Render.com/Railway) with system tesseract+poppler

### Deployment to Streamlit Community Cloud (2026-07-16) (HISTORICAL)

- **URL:** https://cvinsight-io.streamlit.app
- **Working:** PDF (text-layer), DOCX, TXT files — parse → extract → score → classify → suggest
- **Scanned PDFs:** easyocr fallback active (pure Python, no system binaries). Accuracy lower than tesseract — line structure fragmented, punctuation lost, character confusions common
- **Model loaded:** XGBoost 3.3.0 (re-saved for version compatibility)
- **Subprocess isolation:** PDF parsing runs in subprocess to contain segfaults from C++ PDFium engine
- **50 MB file size limit** enforced to prevent OOM on 1 GB RAM container
- **Model files tracked:** `xgb_classifier.pkl` + `xgb_booster.model` (native format) + `xgb_vectorizer.pkl` + `xgb_labels.json`

### Week 5 — ML Text Classifier & Streamlit V1 ✅ (100%)

**Classifier Results (4,612 CVs, TF-IDF vectorization):**

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Majority Class Baseline | 73.46% | 0.6222 |
| Logistic Regression | 85.81% | 0.8642 |
| **XGBoost (best)** | **87.65%** | **0.8754** |

- Both ML models significantly outperform majority baseline
- XGBoost selected as production model
- Top features: "project", "projects", "application", "experience", "developer experience", "machine learning" — genuine quality signals
- Weak class (1.8% of data) remains challenging for both models

**Deliverables:**
- `scripts/vectorize_cvs.py` — full training pipeline with LR baseline + XGBoost + error analysis
- `scripts/build_classifier_data.py` — reconstructs resume text from structured JSON, runs extract_all + scoring
- `models/xgb_classifier.pkl` — best model (TF-IDF + XGBoost pipeline)
- `models/lr_baseline.pkl` — baseline model
- `app/app.py` — Streamlit V1 UI: upload → parse → extract → score → classify → suggest
  - Side-by-side rubric score vs ML prediction
  - Section breakdown bar chart
  - Color-coded labels (green/orange/red)
  - Improvement suggestions
  - JSON download

### Week 6 — JD Matching, Ranking & V2 App ✅ (100%)
- `src/matcher/embedder.py` — sentence-transformer embedder
- `src/matcher/semantic_scorer.py` — cosine similarity
- `src/matcher/skill_overlap.py` — skill overlap computation
- `src/matcher/ranker.py` — ranking formula engine
- Streamlit V2 with ranking tab

### Week 7 — Fine-Tuning, Evaluation & Final Report ✅ (100%)
- NER fine-tuning → `models/ner-v1` (distilbert, in-domain P/R/F1 0.998)
- Side-by-side evals: rule vs LLM, rubric classifier vs TF-IDF+XGBoost
- Full evaluation metrics suite (NDCG@5 0.98, Spearman ρ) → `models/week7_metrics.json`
- Matcher fixes A/B/C/D + `docs/final_report.md` + `docs/matcher_datasets_latency.md`
- Remaining (research/backlog, not blockers): see `TODO.md` "Matcher improvement pipeline"

### LLM Extraction — Custom LoRA Fine-Tuning (v2) — 2026-08-03
- **Data curation:** `scripts/curate_dataset.py` builds canonical, leakage-safe split →
  `data/processed/curated_{train,val,test}.jsonl` (3,928 / 345 / 345) + 8 hand-write edge cases
  (career break, academic `20XX`, `Present` on its own line, membership≠leadership, Ph.D./MBA degrees).
  Degrees canonicalized to the scorer's `degree_points` keys; spoken languages pulled out of `skills`.
- **Training:** `scripts/train_llm.py` now consumes curated split + **masked JSON-only loss**,
  rsLoRA, NEFTune, CPU fallback, and checkpoint/`--resume` for power-loss recovery.
- **Env:** moved to CUDA build `torch 2.11.0+cu128` (+`torchvision 0.26.0`) on RTX 5070 Ti;
  small peft↔transformers 5.8 lazy-import shim added.
- **Timed run:** 1 epoch = 7.5 min on GPU; `train_loss 0.492`, `eval_loss 0.399`.
- **Gate:** `scripts/gate_llm_vs_rules.py` — v2 (1-epoch) beats the rules on **10/10 demo CVs**
  (mean score 68 vs 53.6). Uses `json-repair` to handle stray quotes the decoder occasionally emits.
- Models: `models/qwen3-0.6b-cv-lora-v2/` (validated 1-epoch) with `checkpoint-246`; full run → `-v3`.
- New dependency `json-repair>=0.61` (noted in `project_plan.md`, added to `requirements.txt`).
- **Full 3-epoch run (`-v3`) does NOT help — evidence of overfitting.** Lower dev eval_loss
  (0.389 vs 0.399) but demo-CV performance drops: mean 59.9 (vs 68) and beats rules on only
  5/10 (vs 10/10). The 1-epoch `v2` model generalizes better to real (out-of-distribution) CVs.
  → Keep `qwen3-0.6b-cv-lora-v2` as the production model; `-v2-1epoch` is an explicit backup.

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
| W3 | NER extraction | ✅ ~97% | 20% | ~19.5% |
| W4 | Scoring engine + Phase 1b | ✅ ~95% | 20% | ~19% |
| P2 | Dataset adapters (cross-dataset) | ✅ 100% | — | +8% |
| P1c | Rule-based extractor fixes (9 real CVs) | ✅ 100% | — | +3% |
| P3 | Text-path rewrites | ✅ 100% | — | +5% |
| LLM | Ready-made adapter evaluation (2 models) | ✅ 100% | — | +2% |
| LLM | Custom LoRA fine-tuning (v1) | ✅ 100% | — | +2% |
| W5 | ML Classifier + Streamlit V1 | ✅ 100% | 15% | 15% |
| W6 | JD matching & ranking | ✅ 100% | 10% | 10% |
| W7 | Fine-tuning & final report | ✅ 100% | 5% | 5% |
| **Total** | | | **100%** | **~99% + backlog research** |

**Execution progress (3-phase plan):**

1. **Phase 1** — ✅ Fix 6 extractor bugs (already applied before this session) → batch-score 4500 CVs → `labeled_cvs.csv` → `borderline_review.csv` → rubric weights adjusted (24.6% Strong, 73.7% Average, 1.7% Weak)
2. **Phase 2** — ✅ Create `src/extractor/adapters.py` with 4 adapters (netsol, ner, ats, classification) + `scripts/batch_extract_all.py` for incremental batch extraction
3. **Phase 3** — ✅ Rewrite text-path extractors: experience (title + description + YYYY-YYYY/MM/YYYY dates), education (paragraph-level + cross-line association), projects (tools from description via skill_extractor), languages (name detection + `Language (Proficiency)` parsing)

**Next: Fix rule-based extractors for 9 real CVs (Phase 1c) → Test LLM-based extraction → Classifier + Streamlit**

---

## 6. LLM-Based Extraction — Exploration Phase

### Motivation
Real-world CV testing (9 CVs on 2026-07-15) exposed critical gaps in the rule-based extractors:
- Section splitter fails on DOCX with tables (2 CVs got score 0 and 12)
- Experience title/company reversed on PDFs (3/9 CVs)
- Education institution captures degree name (5/9 CVs)
- Languages extracts skill categories instead of language names
- Phone regex misses `+1-555-0198` format

### Options Evaluated
| Option | Model | Params | Speed/CPU | RAM | JSON Rel. | Setup |
|--------|-------|--------|-----------|-----|-----------|-------|
| A | `sandeeppanem/qwen3-0.6b-resume-json` (LoRA) | 0.6B | 3-6s | ~1.8GB | 95%+ | Ollama + adapter |
| B | `Qwen2.5 1.5B` (base instruct) | 1.5B | 10-15s | ~2.5GB | 95.7% | `ollama run qwen2.5:1.5b` |
| C | `SmolStruct 1.7B` + GBNF grammar | 1.7B | 10-15s | ~2.5GB | 99.5% | llama.cpp |
| D | `Gemma 2 2B` (best entity accuracy) | 2.6B | 10-15s | ~2.5GB | 95% parse | `ollama run gemma2:2b` |

**Recommended starter:** Option A — `sandeeppanem/qwen3-0.6b-resume-json` (LoRA on Qwen3-0.6B)
- Purposely fine-tuned on 4,879 resumes for structured JSON extraction
- Fastest CPU inference (3-6s/CV)
- Smallest footprint (~1.8GB)
- Already has a working HF Space demo
- Apache 2.0 license

### Decision: Custom Fine-Tuning Over Ready-Made

After evaluating options, the chosen approach is **custom LoRA fine-tuning of Qwen3-0.6B on our own labeled CV data** rather than using a ready-made adapter. Rationale:
- Teacher requires demonstrable custom contribution — training our own adapter meets this
- Full control over field definitions and output schema
- Can extend to Bangla/Bengali CVs in future phases
- Hardware available: RTX 5070 Ti (local training, no Colab needed)
- Reference pipeline: `sandeeppanem/qwen3-resume-extraction` repo

**Fine-tuning pipeline plan:**
1. Convert 4,500+ labeled CVs → JSONL (raw text + CVSchema JSON pairs)
2. Format as Qwen3 chat template with system/user/assistant messages
3. Load `Qwen/Qwen3-0.6B` base model (frozen) + PEFT LoRA adapter
4. Train on RTX 5070 Ti (est. 15-20 min for 3 epochs)
5. Export LoRA adapter (~5MB) → integrate into pipeline
6. Side-by-side comparison vs current `extract_all()` on 9 real CVs

### Bangla/Bengali CV Support — Strategy

Qwen3 supports 119 languages including Bengali natively. Three-phase approach:

**Phase 1 (Current):** Fine-tune on English CVs only — establish baseline
**Phase 2 (Bangla V1):** Collect 200-500 Bangla CVs, translate to English, fine-tune mixed model
**Phase 3 (Bangla native):** Use available Bangla NLP resources for native extraction:

| Resource | Type | Use |
|----------|------|-----|
| B-NER (Kaggle) | Bangla NER dataset | Entity recognition |
| Onneshon (Mendeley) | Hybrid Bengali resume dataset, section-labeled | Resume structure |
| AI4Bharat Sangraha | 251B tokens across 22 Indic languages | Pre-training corpus |
| celloscopeai/bangla_ner_dataset | Person name extraction | Name field |

### LLM Evaluation Results (2026-07-15)

#### Ready-Made Adapter Search

| Model | Params | Valid JSON | Fields | Verdict |
|-------|--------|-----------|--------|---------|
| `sandeeppanem/qwen3-0.6b-resume-json` | 0.6B | 9/9 | Profile summary only | Schema incompatible |
| `nimendraai/NuExtract-tiny-Resume-Data-Extractor` | 0.5B | 8/9 | Basic fields | Too shallow, hallucinations |

**Decision: Custom LoRA Fine-Tuning of Qwen3-0.6B** — no ready-made adapter matches CVSchema requirements; custom contribution required.

#### Custom LoRA Fine-Tuning ✅

**Training Pipeline** (built 2026-07-15):
- `scripts/generate_training_dataset.py` — processes datasetmaster_clean.csv (4,612 CVs), reconstructs resume text from structured JSON columns, runs `extract_all()` → Qwen3 chat format JSONL
- `scripts/train_llm.py` — PEFT LoRA training on Qwen3-0.6B
- **Hyperparams:** LoRA rank=16, lr=2e-4, batch=4, grad_accum=4, BF16, max_seq_length=2048
- **Training:** 2 epochs, 552 steps, ~2 hrs on RTX 5070 Ti
- **Best eval loss:** 0.499 (checkpoint-276, epoch 1). Epoch 2 slightly overfit (0.505)

**Evaluation on 9 Demo CVs:**
- 2/9 valid full JSON (Vikram, Barry); 7/9 truncated at 1024 tokens
- **Does extract:** name, email, phone, skills, education, experience, projects, languages
- **Win:** Got Barry Allen name correct (both rule-based + sandeeppanem adapter failed there)
- **Issues:** Experience entry duplication, skill duplicates (inherited from training data), ~42s/CV inference
- **Action Items:** Deduplicate training data, use compact JSON (saves 60% token budget), add more datasets before retraining

### Remaining Edge Cases (Post Phase 1c + Custom LoRA)

Even after fixes, some edge cases remain in our rule-based extractor:

| CV | Remaining Issue | Severity |
|----|----------------|----------|
| `pro-cv-template-burgundy.docx` | Name extracted as "Email" (first detected line is email) | Medium |
| `Rebecca_Software or Computational Roles.docx` | Company = "PROJECT HIGHLIGHTS" in 3 entries | Low (non-standard CV) |
| `srbhr_repo_barry_allen_fe.pdf` | Email contains `#` prefix noise | Low |
| All | Skill deduplication (occasional cross-section duplicates) | Low |
| All | Location leakage into experience duration (e.g. "Pune", "New York") | Low |

### Extraction Improvement Ideas (From NuExtract Analysis)

1. **Experience duration post-processing** — Filter city/state/country words from parsed duration strings
2. **Skill deduplication** — Add `deduplicate_skills()` step normalizing casing and removing near-duplicates
3. **Phone normalization** — Standardize formatting to consistent pattern
4. **Name confidence heuristic** — Reject extracted names that look like section headers or common words
5. **Cross-field consistency** — Flag if experience company is a location word (hallucination detection)

### Immediate Next Steps
1. ✅ Phase 1c completed — all 6 extractor bugs fixed
2. ✅ `test_cv_files.py` verified all 9 CVs improved; 343 tests passing
3. ✅ Ready-made adapters evaluated (sandeeppanem: profile summarizer, NuExtract: too shallow)
4. ✅ Custom LoRA pipeline designed, trained, and evaluated
5. ✅ **Week 5 — ML Text Classifier + Streamlit V1** — XGBoost (87.65% acc), Streamlit app running
6. ✅ **Week 6 — JD Matching & Ranking (V2)** — embedder, semantic scorer, skill overlap, ranker, app integration, matching_eval notebook, 18 tests
7. ✅ **Week 7 — Custom LoRA v2 improvements + Final Report** (see 2026-08-04 entries above)

### NLP Techniques Mapping

| Technique | Teacher Required | Where | Implementation |
|-----------|----------------|-------|----------------|
| Named entity recognition | ✅ | Extractor | spaCy EntityRuler + PhraseMatcher |
| Information extraction | ✅ | Extractor | Section-specific rule-based parsers |
| Keyword extraction | ✅ | Extractor | Skill PhraseMatcher on `skill_taxonomy.json` |
| Text classification | ✅ | **Week 5** | **TF-IDF + XGBoost on raw CV text** |
| Semantic similarity | ✅ | Week 6 | sentence-transformers all-MiniLM-L6-v2 |
| Ranking model | ✅ | Week 6 | Weighted formula (0.5×semantic + 0.3×skill + 0.2×rubric) |

### Progress Estimate

| Phase | Status | Est. Complete |
|-------|--------|--------------|
| Phase 1 (Parser + Extractor) | ✅ 100% | Done |
| Phase 1c (Rule-based fixes) | ✅ 100% | Done |
| Phase 2 (Dataset adapters) | ✅ 100% | Done |
| Phase 3 (Text-path rewrites) | ✅ 100% | Done |
| LLM Eval (Ready-made) | ✅ 100% | Done |
| Custom LoRA (v1) | ✅ 100% | Done |
| Custom LoRA (v2 quality) | ⏸️ Hold | After V1 |
| Week 5 (ML Classifier + Streamlit) | ✅ 100% | Done |
| Week 6 (JD Matching) | ✅ 100% | Done |
| Week 7 (Fine-tune + eval report) | ✅ 100% | Done |

**Overall:** ~99% (code complete; matcher research backlog + deployment remain)

---

### Fast "Rule + NER" Hybrid & Small-Model Research � 2026-08-04

**LLM hybrid set aside.** The fine-tuned Qwen3 LoRA v2 (mean 65.6, beats rules 10/10) is
accurate but runs ~1min/CV on CPU (~20-30s even on a 16GB GPU), and free CPU Cloud (1GB RAM)
cannot host it at all. It remains available offline (`src/extractor/hybrid.py`) but is
**unplugged from the app** � not viable for real-time per-upload scoring.

**Small-model research (2021-2026):** Reviewed encoder NER vs small generative models on CPU:
- Generative text-to-JSON SLMs (TinyLlama 1.1B, SmolLM2-360M, Qwen2.5-0.5B, NuExtract-tiny) all run
  ~10-60s per resume on CPU � same wall as our LLM. Not real-time.
- Encoder token-classification NER (DistilBERT/BERT, 65-110M) is the only CPU-real-time family
  (~15-150ms/resume). External top pick: `oksomu/resume-ner` (65M, 13 resume labels, ~15ms quantized).
- Our own fine-tune **`models/ner-v1`** (distilbert, trained in ~37s) already matches that schema.

**Implemented:** `src/extractor/ner_tag.py` � windowed span tagging (handles >512 tokens),
`predict_spans()` / `merge_skills()` / `extract_education_gaps()`. App now offers
**"Rule-based" (default) / "Rule + NER (fast)"** (CPU, no GPU, no LLM). LLM hybrid removed from UI.

**Benchmark (10 demo CVs, CPU):**
| Engine | mean score | notes |
|--------|-----------|-------|
| rule-only | 53.6 | production default |
| **rule + NER** | **54.6** | fast (~0.05-0.09s/CV) |
| LLM hybrid (set aside) | 69.0 | too slow CPU |

**Honest takeaway:** rule+NER is fast and hallucination-safe but only +1pt mean (3/10 CVs) � rules
already near-cap the skills section and the NER can't do the dates/relations that gave the LLM its
real gain. It's a safe, free upgrade, not a dramatic one. If a significant extractor gain is needed
at CPU latency, that needs a fast relation/date resolver (open problem; out of current scope).

Files: `src/extractor/ner_tag.py`, `scripts/train_ner.py`, `models/ner-v1`; docs updated in TODO.
