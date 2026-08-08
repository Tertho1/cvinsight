# CV-Insight — Complete Knowledge Document for the Presentation

> **Goal of this file:** give the person building the PPT every fact they need, in plain
> language, so they can explain the project, answer "why did you do it this way" and
> "was there a better alternative" questions, and make slides without digging through code.
>
> Everything here is taken from the actual repository (`README`, `progress.md`,
> `project_plan.md`, `TODO.md`, `docs/final_report.md`, source code, measured metrics).

---

## 1. One-line description

> **CV-Insight** is an end-to-end automatic CV evaluation, scoring and matching system:
> upload a CV (PDF / DOCX / TXT), it extracts structured information (skills, education,
> experience, projects…), scores it against a configurable rubric (0–100), classifies its
> quality (Strong / Average / Weak) with a machine-learning classifier, generates
> improvement suggestions, and optionally matches the CV against a **job description** to
> rank multiple candidates.

---

## 2. Problem statement & motivation

**Problem:** Recruiters and HR staff receive hundreds/thousands of CVs for one opening.
Reading every CV manually to score quality and match it to a job description is:

- **Time-consuming** — minutes per CV × hundreds of CVs.
- **Inconsistent** — two reviewers score the same CV differently.
- **Unstructured** — CVs are PDF/DOCX/TXT files; skills, dates, degrees are buried in free text.

**Goal of the project:** automate the repetitive parts — extract, score, classify, suggest,
and rank CVs vs a job description — with transparent, explainable, configurable rules,
plus a machine-learning quality classifier and an optional fine-tuned LLM extraction path.

**Academic framing (what to say in the thesis):** a "hybrid" NLP system — **rule‑based +
NER** for deterministic, explainable extraction; **ML** (TF-IDF + XGBoost) for quality
classification; **sentence‑embedding + fine-tuned contrastive matcher** for JD matching;
and a **fine-tuned small LLM (Qwen3‑0.6B LoRA)** as the deepest extraction path.

---

## 3. The three versions (build order)

| Version | What was added | Mandatory? |
|---|---|---|
| **V1** — Core Evaluator | Parse → Extract → Score (rubric) → Classify (ML) → Suggest. Streamlit web app. | Mandatory |
| **V2** — JD Matching & Ranking | Paste a job description → semantic similarity + skill overlap → match % → ranked list of many CVs | Target |
| **V3 — Schema v2 (delivered)** | Auditable `criteria_scores` breakdown with human-readable rationale per criterion; `jd_match` → `match` rename | stretch → delivered as Schema v2 |
| **V3 — Fine-tuned models** | Fine-tuned NER (distilbert), fine-tuned LLM (Qwen3 LoRA), fine-tuned matcher (bge-small ConFit) | stretch → done |

Note for the presentation: all three versions were **delivered**. There is an untouched
company-mode/batch/auth backlog (see §13).

---

## 4. System architecture & data flow

**The end-to-end pipeline (one diagram for the first slide set):**

```
CV file (PDF/DOCX/TXT)
   │  (1) PARSER
   ▼
raw text  (OCR fallback if scanned, < 50 chars → tesseract / easyocr)
   │  (2) SECTION SPLITTER
   ▼
sections dict (experience, skills, education, projects, …)
   │  (3) EXTRACTOR  (rule-based + spaCy NER; optional DistilBERT NER fuse;
   │                  optional Qwen3 LoRA LLM fuse)
   ▼
CVSchema (Pydantic) — structured JSON: name, email, phone, education, experience,
         skills, projects, certifications, languages, achievements, leadership
   │  (4) SCORER — rubric (config/rubric_config.json + default_criteria.json)
   ▼
criteria_scores (each with score/max/weight/method/rationale) + total_score + label
   │  (5) SUGGESTER — template tips per weak section
   ▼
suggestions list
   │  (6) Classifier — TF-IDF + XGBoost on raw text → ML label + confidence
   ▼
ML quality label (compared side-by-side with rubric label in the UI)
   │  (7) Matcher (optional JD) — semantic similarity + skill overlap + rubric → match%
   ▼
JD match %, missing skills, ranked candidate list (0.5×sem + 0.3×skill + 0.2×rubric)
```

**Schema-first convention:** every module writes to **`src/schema.py` → `CVSchema`**
(a Pydantic model). The parser produces text, the extractor fills fields, the scorer fills
scores/label, the suggester fills suggestions, the matcher fills `match`, the app displays.

---

## 5. Technology stack (for the "Tools & Technologies" slide)

| Layer | Tool | Why / role |
|---|---|---|
| Language | Python 3.10 (env actually 3.14) | Team language, rich NLP/data ecosystem |
| Parsing PDF | `pdfplumber`, `pdfminer.six`, `pypdf` | 3-strategy extraction with fallbacks; multi-column handling |
| Parsing DOCX | `python-docx` | Tables & text boxes ("w:txbxContent") as lines |
| Parsing OCR | `pytesseract` (+ system Tesseract), `easyocr` | Scanned/image PDF fallback; tesseract is much more accurate |
| NER / NLP | `spaCy` (en_core_web_sm) | Sentence splitter, PhraseMatcher for skills, NER (PERSON/ORG) for name/company, EntityRuler |
| NER (fine-tuned) | Hugging Face **Transformers** — DistilBERT token classifier `models/ner-v1` | Fast CPU-era resume NER (SKILL/DEGREE/INSTITUTION/…) fused over rules |
| Skill matching | spaCy `PhraseMatcher` (attr=»LOWER») over `config/skill_taxonomy.json` (~270 skills, 8 categories) | Case-insensitive vocabulary match |
| LLM extraction | **Qwen/Qwen3-0.6B** base + **PEFT LoRA** adapter (`models/qwen3-0.6b-cv-lora-v2`), `torch`/`transformers`/`peft`/`json-repair` | Deep, schema-aware extraction; JSON output with `json-repair` for malformed JSON |
| Scoring | `pydantic` schema, JSON config files | Config-driven weights (no hardcoded) |
| Quality classifier | deployed **hybrid v3** `models/classifier_v3_hybrid_synth.pkl` (XGBoost **regressor** → 0–100 score; features = `matcher-confit` 384-d embedding + 12 engineered `extract_all` features + ~7 organic-prose macro features); fallback `xgb_classifier.pkl` (TF-IDF + XGBoost) | ML quality label (Strong/Average/Weak) via score thresholds (Weak<50 / Average<72 / Strong≥72) |
| Embeddings matcher | `sentence-transformers`; default `models/matcher-confit` (ConFit-style fine-tune of `BAAI/bge-small-en-v1.5`) | CV ↔ JD semantic similarity (cosine) |
| Lexical matcher | hand-rolled Okapi **BM25** (`src/matcher/bm25_scorer.py`, no dependency) | Optional lexical signal / pool pre-filter |
| Blending / ranking | `numpy` weighted formula; configurable via `CV_RANK_WEIGHTS` | Final rank = 0.5·sem + 0.3·skill + 0.2·rubric (+0 ·BM25) |
| Web app | **Streamlit** | Upload → results UI; lightweight, Python-native |
| Config | JSON files (`rubric_config.json`, `default_criteria.json`, `skill_taxonomy.json`) | Change weights without touching code |
| Testing | `pytest` | 482 tests across 23 files |
| Versioning | Git + GitHub | https://github.com/Tertho1/cvinsight |
| Hardware (dev) | RTX 5070 Ti (16 GB VRAM) | Local training of LoRA + ConFit |
| Deploy | Streamlit Community Cloud (halted — 1 GB RAM OOM) → target Render/Railway | See §12 |

---

## 6. How each component works (deep dive — copy bullets to slides)

### 5.1 Parser — `src/parser/parser.py`
- `parse_cv(path)` dispatches by extension (`.pdf`, `.docx`, `.txt`).
- **PDF:** 3-strategy attempt (pdfplumber → pdfminer → pypdf) to survive broken layout PDFs.
- **OCR fallback:** if PDF text < 50 chars it is considered scanned → run pytesseract OCR;
  if tesseract unavailable → easyocr (pure Python, no binaries). Verified benchmarks:
  tesseract much more accurate than easyocr (measurable per-character differences).
- **DOCX**: table cells placed on separate lines (so section headings inside tables are
  detected) and textbox paragraphs (`w:txbxContent`) are extracted.
- **TXT**: multi-encoding reader (utf-8, latin-1, cp1252).
- **Cleaner** (`cleaner.py`): removes special chars, normalizes whitespace/encoding, cleans
  `(cid:127)` PDF artifacts.

### 5.2 Section splitter — `src/parser/section_splitter.py`
- Maps **80+ heading aliases** ("Work Experience", "Employment History", "Professional
  Experience", …) to **12 canonical sections** (header, summary, education, experience,
  skills, projects, certifications, languages, achievements, leadership, references, other).
- Line-by-line detection: short line (< 60 chars), no sentence-ending punctuation, no
  bullet prefix, matches a keyword exactly or as a boundary.
- Any content that doesn't match is kept in `other` — **no data is ever dropped**.

### 5.3 Extractor — `src/extractor/`
The master function `extract_all(text, sections)` orchestrates sub-extractors:
- **contact_extractor.py** — email/phone/LinkedIn via regex + spaCy NER for the name, with
  a large tech-term blocklist to prevent NER labeling e.g. "Java"/"React" as a PERSON.
- **skill_extractor.py** — PhraseMatcher over taxonomy; multi-source: structured
  `skills.languages` routing, summary block, and full-text fallback.
- **education_extractor** — degree level (Ph.D/Master/Bachelor/Diploma + aliases M.Sc,
  M.Tech, M.Eng, B.E, …), institution (with filtering so a degree name is not taken as an
  ORG), year, GPA, field; multi-line paragraph parsing.
- **experience_extractor** — title / company / dates / duration. Flexible date regexes
  (`Jan 2021 – Present`, `2021-06` / `Till Date`). Whole-section, date-anchored parser so
  multiple jobs in one DOCX paragraph are not collapsed. Title/company reversal detection
  (common on PDFs) with backtracking; ORG false-positive guards.
- **misc_extractor.py** — projects (name, tools from description, GitHub link), certificates,
  languages (name + B2/C1 proficiency), leadership (including "Led team of 5…" bullets found
  inside work-experience text), achievements.
- **Dataset adapters** (`adapters.py`) — normalize 4 external dataset formats (NETSOL, NER,
  ATS, classification) so batch extraction works on all of them.
- **Optional NER tagger** (`ner_tag.py`) — a fine-tuned DistilBERT framing
  (`models/ner-v1`) that emits only spans present in text (~15–60ms per resume), fused into
  rules: `merge_skills()` + `extract_education_gaps()`. A `_skill_parts()` cleaner splits
  chained spans and drops URL/email/geo junk while preserving true `.js` skills.
- **Optional LLM** (`hybrid.py`, `extract_with_llm`) — runs the fine-tuned Qwen3 LoRA,
  builds a grounded dict (skills only if present in text — hallucination filter), then
  `fuse()` merges it with the rule output field-by-field:
  - skills = union (both grounded; dedup)
  - experience/education = prefer the source with ≥ number of dated entries
  - projects/certs/languages = whichever found more
  - leadership/achievements = rule-based only
  - On any failure → empty dict → graceful rule-based fallback.
  - **App skips this step for Bengali CVs** (the LLM is English-only; garbage output would
    overwrite the clean rule+NER result via `fuse()`).
- **Bangla native route** (`bangla_extractor.py` + `bangla_section.py`) — `extract_all()`
  detects Bengali script (`is_bangla`, U+0980–U+09FF ≥10% + ≥3 chars) and routes to
  `extract_bangla()`:
  - Transliterates Bengali digits (০-৯→0-9), months (জানুয়ারি→January), date markers
    (বর্তমান→present), degree words (Master/B.Sc/HSC, dotted B.Sc/B.A), job titles
    (প্রকৌশলী→Engineer, IT Support Executive, Senior Backend Engineer), technical skills
    (Terraform/FastAPI/PostgreSQL/AWS/Linux/Windows), institutions (বিশ্ববিদ্যালয়→University,
    Polytechnic, College, Institute), section headings (দক্ষতা→TECHNICAL SKILLS,
    কর্মসংস্থান ও অভিজ্ঞতা→experience, সার্টিফিকেশন ও প্রজেক্ট→certifications,
    পেশাগত সারাংশ→summary) and dash-format language pairs (বাংলা — মাতৃভাষা→Bengali – native).
    Latin tech terms, emails and phones pass through untouched.
  - BD phone normalization: `+880 1712-345678` → `+8801712345678`.
  - Sets `language: "bangla"` on the schema; uses `bangla_section.py` (Onneshon-trained
    classifier) as the sectioning fallback when headings are missing.
  - App shows a "Language: Bangla" badge and skips English-only steps (DistilBERT-NER,
    hybrid-ML classifier, Qwen3 LLM).
  - Hardened on real CVs (2026-08-09): `demo/banglacv1.txt` 22→54, `demo/banglacv2.txt` 6→26.

### 5.4 Schema — `src/schema.py`
- Pydantic `CVSchema` with nested models `Education, Experience, Project, Certification,
  Language, SectionScores, CriterionScore, JDMatch`.
- Fields: cv_id (MD5 of bytes), name, email, phone, education, experience, skills,
  projects, certifications, languages, achievements, leadership, `criteria_scores`,
  section_scores, total_score, label, suggestions, `match` (legacy alias `jd_match`),
  `language` (default "en"; "bangla" when the CV is Bengali).

### 5.5 Scorer — `src/scorer/scorer.py`
- Reads weights from `config/rubric_config.json` and the criterion list from
  `config/default_criteria.json`.
- Each criterion computes a score via `section_scorers.py`, clamps to its cap, and emits a
  `criteria_scores` row with `{name, score, max_points, weight, method, rationale, overridden_by}`.
- Total = weighted mean of `(score/max)*weight`, normalized to 0–100. With default config this
  equals the plain sum of section scores.
- Label thresholds: **Strong ≥ 72, Average 50–71, Weak < 50** (config-driven).
- Rationale builders produce explainable strings **without** an LLM (e.g. "2 roles totalling 5.0
  years of experience").

### 5.6 Scoring rubric (the actual table for a slide)

| Criterion | Max | How it's scored |
|---|---|---|
| Experience | 25 | Total years → bands (0.5=2, 1=5, 2=9, 4=14, 6=18, 8=22, 8+=25) |
| Projects | 20 | 8 pts per project (count ×8, cap 20) + 1 pt per GitHub/live link (max 5) |
| Skills | 20 | # matched skills / target (10) capped × 20 |
| Education | 15 | Ph.D.=15, Master=13, Bachelor=10, Diploma/Assoc=6, +2 GPA ≥ 3.5 bonus |
| Certifications | 10 | 2 pts each, cap |
| Languages | 5 | 1 lang=2, 2 lang=4, 3+ lang=5 |
| Leadership & extras | 5 | 2 pts per role, cap |

### 5.7 Suggester — `src/suggester/suggester.py`
- For each section scoring below 60% of its cap, emits a specific tip from a template dict
  (e.g. "Add your GPA", "Quantify achievements in your work experience"), **max 5 tips**.
- Config thresholds — no hardcoded values.

### 5.8 Quality Classifier — `src/extractor/quality_features.py` + `scripts/build_hybrid_classifier.py`
- **(v1/v2 era):** TF-IDF vectorizer on raw reconstructed CV text → XGBoost. ~4,500 CVs,
  labels = rubric Strong/Average/Weak (pseudo-labels). XGBoost beat LR baseline
  (87.65% vs 85.81%). But it was trained on **reconstructed** dataset text, so on
  **organic prose** (demo/benchmark) the agronly matched the rubric label **4/10** — it had
  learned the training distribution, not CV quality.
- **(v3 hybrid, deployed 2026-08-08)** `models/classifier_v3_hybrid_synth.pkl` — reframed
  as **regression to total_score** (ordinal by construction), then label = thresholds
  (Weak<50 / Average<72 / Strong≥72):
  1. `matcher-confit` embedding of raw text (384-d, same embedder as matcher) — transfers
     from prose to prose;
  2. **12 engineered features** from a real `extract_all` pass (`feature_builder.build_features`:
     highest degree, total years, skill/project/cert/language/leadership/achievement counts,
     github link, avg GPA, entry counts);
  3. **~7 organic-prose macro features** (no NER: section_presence, word_count, line_count,
     mean_sentence_len, date_count, Flesch reading ease);
  4. GPU XGBoost regressor (`reg:squarederror`) predicts total_score.
  - Benchmark agreement **7/10** vs 4/10, score-level Spearman **+0.758** vs ~0.
  - `load_classifier()` prefers the hybrid ("Hybrid (v3 synth)") with `xgb_classifier.pkl`
    fallback. Warm per-CV classify ~120–180 ms.
  - Fixed a `predict_proba` bug: bucket centers flipped at 74/42.5 ≠ rubric thresholds;
    rewrote as normal-CDF (`scipy.special.erf`) so argmax ≡ `predict()` for every score.

### 5.9 Matcher / JD matching — `src/matcher/`
- **embedder.py**: `sentence-transformers` singleton, default `models/matcher-confit`
  (bge-small fine-tune), lazy load, `warm_up()` for app-start warming. `CV_EMBEDDER` overrides.
- **semantic_scorer.py**: cosine similarity between CV embeddings and JD embeddings.
  2 modes — `whole` (default, one vector each) and `section` (each CV section embed vs JD,
  weighted: skills 0.30, experience 0.25, summary 0.15, projects 0.15, education 0.10,
  achievements 0.03, cert 0.02).
- **skill_overlap.py**: extract JD skills with the taxonomy, `matched/JD_skills` ratio +
  sorted `missing_skills`. Empty JD = 0.0; non-empty JD without parseable skills = neutral 0.5.
- **bm25_scorer.py**: hand-rolled Okapi BM25, JD=query, CV=doc. `score()` is single-pair
  [0,1]; `score_corpus()` is corpus-IDF for pre-filtering.
- **ranker.py**: final match = `0.5·semantic + 0.3·skill_overlap + 0.2·rubric` (+ optional
  `bm25` weight, default 0). `rank_cvs()` sorts all CVs vs the same JD.

### 5.10 Streamlit app — `app/app.py`
- Upper: custom dashed upload box (multi-file, PDF/DOCX/TXT, ≤ 50 MB), JD paste box.
- Sidebar: extraction engine selector **("spaCy + NER" default | "spaCy + Qwen3 LoRA LLM")**,
  LLM device selector (auto/gpu/cpu), **adjustable rubric weight sliders** (sum-100 check),
  model status, "Clear Database". Bengali CVs get a "Language: Bangla" badge and skip the
  English-only NER/classifier/LLM steps.
- Results view: KPI cards (Rubric Score + label, ML classification + confidence, JD match %,
  key strengths), per-section colored score cards with editable rationale, tabs:
  Extracted Data / Suggestions / Raw Text / History / **JD Match** / **Ranking** /
  **Skill Search**.
- History tab: sortable comparison table of all CVs (name, score, label, skill count…),
  CSV + JSON export, filtered search.
- Ranking tab: ranks every stored CV vs the current JD, show weights, click-to-view.
- Skill Search tab: AND/OR skill query across all stored CVs.
- Persistence: `data/processed/cv_database.json`; subprocess-based PDF parsing to isolate
  segfaults; NER/LLM/matcher models cached with `@st.cache_resource`.

---

## 7. Job-description matching — step by step (the "how for JD" question)

1. **User pastes a job description** (text area) and uploads CV(s).
2. **Skill extraction from JD** — same PhraseMatcher taxonomy finds the JD's required skills.
3. **Embed both** — CV (whole text or sections) and JD are embedded with the sentence-transformer
   embedder (`matcher-confit`, CPU).
4. **Semantic similarity** — cosine between CV and JD vectors → `sem` ∈ [0,1].
5. **Skill overlap** — `#(JD skills ∩ CV skills) / #(JD skills)` → `skill_ratio` + sorted
   `missing_skills`.
6. **Rubric normalization** — `rubric_score/100`.
7. **Final match** = 0.5·sem + 0.3·skill + 0.2·rubric (default weights; configurable via
   the UI and `CV_RANK_WEIGHTS`). Optional BM25 lexical signal (default weight 0).
8. **Ranking across candidates** — `rank_cvs()` computes this for every stored CV against the
   same JD and sorts descending; the UI shows Rank / Name / Match % / Semantic / Skill / Rubric.
9. The **"JD Match" tab** shows the % bar, semantic, skill overlap, missing-skill count, and
   two lists: ✔ matched skills and ❌ missing skills.

**Why this design:** pure semantic embeddings miss exact keywords (frameworks, acronyms),
so skill overlap catches hard requirements; rubric score adds overall CV strength so the ranking
is "best fit for the JD AND strong CV", not just relative mention similarity.

---

## 8. Datasets — the "for the dataset slide"

| Dataset | Approx. size | Used for | Where from | Why chosen |
|---|---|---|---|---|
| `MehyaarD/Annotated_NER_PDF_Resumes` | ~5,029 resumes (used clean ~3,300+) | NER fine-tuning (skills, entities) | Hugging Face | Resume-domain NER; clean labels |
| `datasetmaster/resumes` | ~4,800 (4,612 usable) | Parser dev, structured JSON, main extraction corpus; **LoRA training data source** | Hugging Face (MIT) | Real + synthetic; has structured columns (skills.languages etc.) |
| `noran-mohamed/Resume-Classification-Dataset` | ~12,000 (raw 13k) | Quality / category classification corpus | GitHub/Kaggle | Large labeled corpus for classifier |
| `0xnbk/resume-ats-score-v1-en` | ~5,000 pairs (raw ~6,300) | JD matching eval (human ordinal label + jina-derived reference) | Hugging Face | The standard CV↔JD score benchmark |
| `netsol/resume-score-details` | ~850 clean pairs (raw ~1,031) | JD classifier build; independent matcher cross-check (numeric 0–10 scores) | Hugging Face | Additional scored CV–JD examples (GPT-4o) |
| `cnamuangtoun/resume-job-description-fit` | 6,241 pairs (1,759 test) | ConFit-style contrastive fine-tune of the embedder; second independent matching eval | Hugging Face (MIT) | Human-labeled fit labels; the standard HF resume–JD matching set |
| `Onneshon` (Mendeley) | 1,739 Bangla segments | Bangla section classifier training (skill/experience/education/objective) | Mendeley | The only public Bangla resume-in-like dataset (section-level) |

**Key caveat to know (teacher question):** public resume datasets are labeled by **job
category**, not by **quality**. A "Data Science" label says nothing about Strong/Average/Weak.
We therefore **generated our own quality labels** using the rubric itself — called
**pseudo-labeling** (the rubric score → Strong/Average/Weak label). 1,382 borderline CVs
(45–51 and 68–74) were flagged for manual review.

**How long did it take** (numbers that answer "how" in the presentation):
- Batch scoring/extraction ~4,500–5,000 CVs: scripted overnight (incremental save every 500).
- Classifier training (4,612 CVs, TF-IDF + XGBoost): minutes on CPU.
- LoRA fine-tune Qwen3-0.6B: **1 epoch ≈ 7.5 minutes on GPU** (older v1 run ~2 h / 2 epochs).
- distilbert NER fine-tune: **~13–37 s**.
- Contrastive matcher fine-tune: ~ minutes on 6,241 pairs (1 epoch).
- Bangla section classifier: quick (1,392 segs).

---

## 9. Models used — what, how trained, results

### 8.1 Quality classifier — `models/xgb_classifier.pkl` → deployed `models/classifier_v3_hybrid_synth.pkl`
- **v1 deployed XGB** (`xgb_classifier.pkl`): TF-IDF (title-case corpus) + **XGBoost**
  (gradient-boosted trees), baseline LR. Training: 4,612 CVs; pseudo-labels from rubric;
  80/20 stratified split; 5-fold CV.
- Results: **XGBoost 87.65% acc / 0.8754 F1** vs LR 85.81% / 0.8642 vs majority 73.46% / 0.6222.
  Why XGBoost: best accuracy, handles sparse TF-IDF features, feature-importance for top
  terms ("project(s)", "experience", "developer", "machine learning", …).
- **v2 lessons (why it plateaued):** benchmark rubric agreement stuck at **4/10** for every
  model — the training text was *reconstructed from dataset JSON* while benchmark/demo CVs are
  *real organic prose* (train/test distribution mismatch), oversampling didn't rescue it, and
  the Weak class is only ~1.8% of the corpus.
- **v3 hybrid (deployed 2026-08-08)** — details in §5.8; headline: reframed as **regression to
  total_score** on 384-d `matcher-confit` embedding + 12 engineered features + ~7 macro features
  → benchmark agreement **7/10**, score-level Spearman **+0.758**. `load_classifier()` prefers
  it; the XGBoost pipeline stays as CPU fallback.

### 8.2 Fine-tuned NER — `models/ner-v1` (distilbert-base)
- Method: token classification (encoder-only — the only family that is CPU real-time,
  ~15–60 ms/CV). Labels: skills, degree, institution, title, company, project, cert, language, person.
- Training: generated labels from dataset resumes (NER tags JSONL, ~37 s GPU).
- Results: in-domain token **P/R/F1 = 0.998**; seqeval-equivalent **span (entity) F1 = 0.988**
  (skill 0.978). On real resumes it only emits spans present in text — contamination-safe.
- Caveat for the presentation: synthetic/train-consistent corpus → **in-domain** F1, not
  cross-domain; real-resume entity F1 is homegrown at ~7/183 span-join artifacts.

### 8.3 LLM extraction — `models/qwen3-0.6b-cv-lora-v2` (Qwen3-0.6B + PEFT LoRA)
- Method: custom **LoRA fine-tune** of the 0.6B base on our curated 4,612-CV training data
  (3,928 train / 345 val / 345 test). Chat-templated JSONL, **masked JSON-only loss**,
  **rsLoRA** + **NEFTune**, BF16, batch 4 × grad_accum 4, seq 2048.
- Training: 1 epoch = 7.5 min. v1 2 epochs ≈ 2 h, eval loss 0.499. v2 1 epoch eval loss 0.399.
  **v3 (3 epochs) overfits** — demo mean 60 vs v2 68; so keep 1-epoch `-v2`.
- Results (grounded skills): demo mean **65.6** vs rules 53.6, wins **10/10** demo CVs.
- Latency: **~27 s/CV GPU** (~1 min CPU) → optional, not default, in the app.
- Why custom LoRA: ready-made resume-LLMs available were 'schema-incompatible or shallow'
  (e.g. sandeeppanem = only profile summary; NuExtract-tiny hallucinates). Custom gives full
  control of the field schema + a demonstrable custom contribution (good for the thesis).

### 8.4 Matcher embedder — `BAAI/bge-small-en-v1.5` → ConFit fine-tune `models/matcher-confit`
- Method: **MultipleNegativesRankingLoss** contrastive Siamese fine-tune on 6,241
  human-labeled resume↔JD pairs (resume anchor, matching JD positive, in-batch JDs as hard
  negatives). 1 epoch (2 epochs overfit).
- Results: test fit-ρ 0.216 → **0.332**; our ATS human-label ρ **0.314 → 0.436**;
  NDCG@5 0.98 → 0.985. Same CPU latency as base (~35 ms/pair).
- Why not bigger: bge-base slower (worse) on our data.

### 8.5 Bangla section classifier (research) — `models/bangla_section_classifier.pkl`
- char-ngram TF-IDF + LogisticRegression on Onneshon segments; 5-fold CV acc **0.9454**,
  dup-leak-safe dedup; held-out 0.922. Supports native Bangla sectioning and is **wired
  into the Bangla route** as the sectioning fallback when heading transliteration misses.

*Rejected models (will say in presentation):*
- Ready-made resume LoRA (sandeeppanem) — schema incompatible (profile-summary only).
- NuExtract-tiny — hallucination, too shallow.
- **Learning-to-rank** (XGBoost rank:ndcg) — NDCG@10 0.68 vs pure semantic 0.78; auxiliary
  features dilute the embedding; **not adopted**.
- **BM25-semantic hybrid** — any nonzero weight lowers ρ/NDCG; settle default weight 0 (BM25
  kept only as an opt-in pool pre-filter).
- Bigger embedders (bge-base): slower and lower ρ than bge-small.

---

## 10. Evaluation metrics (the "results" slide)

Reproduced in `models/week7_metrics.json`.

| Module | Metric | Result | Target |
|---|---|---|---|
| Classifier (XGB, held-out primary) | accuracy / weighted F1 | 0.8765 / 0.8754 | ≥0.80 |
| Classifier (hybrid v3, 2026-08-08) | benchmark rubric agreement | **7/10** (vs 4/10 XGB) | ≥7/10 |
| Classifier (hybrid v3) | score-level Spearman vs rubric | **+0.758** (vs ~0 XGB) | ↑ |
| Classifier (hybrid v3) | held-out accuracy / F1 | 0.885 / 0.881 | ≥0.88 |
| NER (token, in-domain) | F1 | 0.998 | ≥0.75 |
| NER (entity/span, in-domain) | F1 | 0.988 | — |
| Extraction quality | demo mean score | 56.2 | — |
| Benchmark set | mean score (baseline → after fixes) | 46.4 → 56.4 | ↑ |
| LLM vs rules | LLM demo mean (10/10 wins) | 65.6 vs 53.6 | — |
| Bangla route (real CVs hardset 2026-08-09) | score before → after | 22 → 54 (sr. SWE) and 6 → 26 (jr.) | ↑ |
| JD matching | Spearman ρ vs human label (ATS, n=500) | 0.436 (conf-tuned embed) | ≥0.65 (aspirational) |
| JD matching | NDCG@5 (benchmark) | 0.985 | ≥0.75 |
| JD matching | NDCG@10 (resume-JD-fit held-out) | 0.309 | reference |
| Suggestions | manual relevance | ≥4/5 in test CVs | spot check |

**Test suite:** 482 unit tests across 23 files (parser 133, extractor ~285, scorer+suggester 61,
bangla 33, hybrid/classifier/ner-skill ~28) — all passing. Command `pytest tests/`.

**Deployment status:** Streamlit Community Cloud down (OOM, torch + easyOCR exceed 1 GB RAM);
migration to Render.com/Railway (Docker) deferred — documented in `docs/` as the final project step.

---

## 11. Key design decisions & alternatives (probably a whole Q&A slide)

| Decision | Chose | Why not the alternative |
|---|---|---|
| Extraction: rules+NER vs LLM | Rule+DistilBERT NER default; LLM optional mode | Rules: deterministic, 0 hallucination risk, ~milliseconds. LLM is more accurate (65.6 vs 53.6) but ~27 s/CV and needs torch memory — so it's a selectable "deep" mode (GPU). A rule + fine-tuned encoder gives testable accuracy without huge latency. |
| Classifier | hybrid v3: embedding + engineered features + score regression (XGB) | TF-IDF XGB was 87.65% on held-out primary but only **4/10** on real prose (distribution mismatch). Hybrid regressor → 7/10 + Spearman +0.758. |
| Embedding | bge-small-en + ConFit fine-tune | MiniLM ρ=0.259 < bge-small 0.348 < ConFit 0.436; bge-base slower *and* worse. |
| Ranker | hand-tuned weighted mix | LTR probe trained → overfit/skip (0.68 vs 0.78 NDCG); automatic-weight fitting says pure semantic is best on ATS; keep a configurable blend for the demo (0.5/0.3/0.2). |
| JSON config for rubric | `rubric_config.json` + `default_criteria.json` | No hardcoded weights — teacher can change scoring without code (this was a project requirement). |
| Pseudo-labels | Rubric score → Strong/Average/Weak | Public datasets don't label quality; labeling only 200–500 manually would be too few; rubric gives the whole training set + borderline band for manual review. |
| Data adapters | one adapter per dataset (`adapters.py`) | External formats all differ in key names (title vs name, degree.title vs degree.level, "Till Date", …). |
| OCR | rule-based fallback to tesseract | EasyOCR lost line structure and punctuation (measured on ocrtest PDF); usable only as a secondary fallback. |
| Environment | CPU-only for default models | Cloud free tier is CPU-only 1 GB — hence DistilBERT NER, no big generative models by default; a GPU path exists for the LoRA on the dev machine. |

---

## 12. Known limitations (be honest — the teachers appreciate honesty)

1. **Deployment OOM**: Streamlit Community Cloud (1 GB RAM) cannot hold torch + easyOCR —
   scanned-PDF OCR disabled on the Cloud today; text PDF/DOCX/TXT work. Migration planned to
   Render.com / Railway (dockerized).
2. **NER F1 is in-domain** (synthetic corpus) — true cross-domain entity accuracy unknown.
3. **Matcher ρ (0.436)** still below the aspirational 0.65 target — because the reference
   dataset is noisy (human ordinal labels); we report NDCG@5/10 as the primary ranking metric.
4. **LLM slow** on CPU (~1 min/CV); only viable on GPU/batch.
5. **Bangla CVs** — native route shipped and hardened (2026-08-09): transliteration + Onneshon
   section classifier get real CVs to 54/26; LLM is English-only so it's skipped for Bengali.
6. **Scoring** rewards presence/quantity (projects, skills, content) — it is not deep expert
   judgement; e.g. it counts matched skills but doesn't judge the quality of a project.
7. JDs with no parseable skills get a neutral 0.5 skill credit (handled deliberately).
8. Scanned PDFs rely on Tesseract + EasyOCR on images.

---

## 13. Future scope (the "future work" slide)

- **Deploy** on Render.com/Railway via Docker so scanned-PDF OCR works in production.
- **Bangla (next steps)** — the native route works on demo CVs but is a transliteration +
  classifier route: extend coverage with more phrase tables, a real Bangla CV corpus, native
  Bangla NER (bangla-bert / Onneshon / BanNERD labels), and LLM support (English-only today).
- **Company / Team hiring mode** — batch of 100+ CVs, criteria builder with editable weights,
  results board, CSV exports (archived design ready).
- **Async batch processing** and a real database/user auth.
- **Improve suggestions**: optional LLM scorer for qualitative criteria (achievement quality,
  leadership signal) — currently transparent template-based rationales.
- **Retrain NER for out-domain**: collect a real held-out resume set for spanning eval.
- **Feedback loop / user manual overrides** stored with `overridden_by` tag already in schema.

---

## 14. Suggested PPT outline — 28 slides (title per slide, with what to put on it)

*Target: a 25-30 slide deck that walks through **each component** and the **methodology** and the
**dataset** story clearly. Every "data point" below is a measured number from §10.*

### Block A — Context (slides 1–5)
1. **Title slide** — "CV-Insight: End-to-end CV Evaluation & Job-Matching System"; subtitle
   "Automated extraction, rubric scoring, ML classification, LLM & JD ranking"; your name +
   date. Put the one-sentence pipeline on the footer line.
2. **Problem statement** — recruiters, hundreds of CVs per role; three bullets: time (
   minutes/CV × hundreds), inconsistency between reviewers, unstructured PDF/DOCX/TXT data.
3. **Motivation & goal** — automate the repetitive 5 steps (extract → score → classify →
   suggest → match); make it transparent, explainable, configurable. One diagram of the 5
   boxes (reuse for slide 6).
4. **Three versions V1/V2/V3** — table: V1 Core (parse→score→classify→suggest), V2 JD matching
   & ranking, V3 Schema-v2 criteria + fine-tuned models (NER/LoRA/ConFit). All delivered.
5. **Scope & audience** — who uses it (HR/recruiters/researchers); English + Bangla; what it is
   not (no HR analytics, no auth/company mode — listed honestly in §13).

### Block B — Architecture & pipeline (slides 6–9)
6. **System architecture diagram** — the end-to-end diagram from §4: CV file → parser → sections
   → extract_all → CVSchema → scorer (criteria) → suggester → classifier → matcher. Annotate
   each arrow with the module name.
7. **Schema-first design** — CVSchema (Pydantic) as the contract; every module reads/writes it
   (parser→text, extractor→fields, scorer→scores/label, suggester→tips, matcher→match). One
   hands-on screenshot of a real extracted JSON.
8. **Tech stack** — the table in §5 (Parser, NER, matcher, web, config, testing). One line per
   tool + one-line "why".
9. **Repository tour / reproducibility** — folder layout (`src/`, `config/`, `scripts/`, `tests/`,
   `models/`), tests command, `demo/benchmark` + baseline JSON; commit discipline.

### Block C — Core components (slides 10–17)
10. **Parser & section splitter** — 3-strategy PDF fallback, DOCX table/textbox handling, OCR
    fallback; the 80+ heading alias → 12 canonical sections map. Screenshot of `parse_cv`.
11. **Extractor (rule + NER)** — contact/skill/education/experience/misc extractors; flex date &
    multi-entry experience parser; PhraseMatcher vs taxonomy; adapters for 4 external formats.
    Before/after extraction example on one CV.
12. **Schema v2 / criteria scores** — how each criterion computes score/max/weight/method,
    rationale without an LLM; the config-driven weights (no hardcoded). Screenshot of the app
    section cards with rationale shown.
13. **Scorer & rubric** — the actual rubric table (experience 25 / projects 20 / skills 20 /
    education 15 / cert 10 / languages 5 / leadership 5), label thresholds Strong≥72/
    Average 50-71/Weak<50 (keep 0-100 scoring). Editable weights via sliders; costs come from
    `criteria_scores`.
14. **Suggester** — template-based tips for sections below 60% cap; max 5 tips; repo config
    thresholds not hardcoded. Show 2-3 example tips next to their sections.
15. **Classifier v3 (hybrid)** — historical note = TF-IDF XGB (87.65% on primary) then the
    failure: only 4/10 on real prose (train/test mismatch). v3 reframes as **regression to 1-100
    score** with 384-d `matcher-confit` embedding + 12 engineered features + ~7 macro features
    → **7/10 benchmark** and Spearman **+0.758**. The two numbers side-by-side make the "why".
16. **LLM path (optional)** — the fine-tuned Qwen3-0.6B LoRA (why custom, latency ~27 s GPU),
    grounding/hallucination filter, `json-repair`, 1-epoch v2 (65.6 vs 53.6 on demo). Show the
    app's "spaCy+NER / spaCy+LLM" selector.
17. **NER fine-tune** — DistilBERT token classifier `models/ner-v1`, in-domain F1 0.998/0.988,
    skill merge hardening (span splits, junk filtering); real-resume safety (spans in text only).

### Block D — JD matching & Bangla (slides 18–21)
18. **JD matching — methodology** — the formula `0.5·sem + 0.3·skill + 0.2·rubric`; semantic
    embeddings (whole vs section modes), skill overlap with missing_skills, optional BM25.
19. **Matcher fine-tune (ConFit)** — contrastive Siamese bge-small → NDCG@5 0.985, ATS ρ 0.436;
    why not LTR/BM25. Ranker: final match sorting.
20. **JD match & Ranking tabs** — app screenshots: match % bar, matched/missing skill lists,
    missing-skill count; ranking table Rank / Match / Semantic / Skill / Rubric.
21. **Bangla support** — script detection → transliteration maps (digits, months, degrees,
    headings, institutions, phone normalization) → existing extractors; Onneshon-trained section
    classifier as fallback; "Language: Bangla" badge + LLM skip; demo scores 22→54 / 6→26.

### Block E — Data & evaluation (slides 22–27)
22. **Dataset master slide** — the §8 table (NER 5k, resumes 4.6k, classification 12k, ATS 5k+,
    NETSOL 850, RJF 6,241, Onneshon 1.7k) + the "no quality labels exist" caveat.
23. **Pseudo-labeling** — why (no public quality labels): rubric score → Strong/Average/Weak;
    1,382 borderline CVs flagged for manual review; how that unlocks classifier training.
24. **Training pipelines** — how each model was trained: LoRA data prep, NER label generation,
    the synthetic corpus for the v3 hybrid (primary-reconstructed vs synthetic), matcher negative
    setup; each with its runtime (~7.5 min LoRA epoch / ~13-37 s NER / minutes matcher).
25. **Evaluation matrix** — §10 table (classifier acc/F1, 7/10, +0.758, NER span F1 0.988, demo
    56.2, benchmark 46.4→56.4, LLM 65.6, ρ, NDCG@5/10) + 482 tests. Highlight what hits and what
    is below target (honesty slide if you want to split it into two).
26. **Where it underperforms / limitations** — §12 list (deployment OOM, in-domain NER F1,
    matcher ρ noise, LLM CPU latency, presence-based scoring). Be honest: teachers appreciate it.

### Block F — Why, future, close (slide 27–28)
27. **Design decisions & alternatives** — the §11 table ("why not GPT/LTR/BM25/big-embed / why
    pseudo-labels") as a single "why" slide.
28. **Future work & thank-you** — §13 (Docker deploy, Bangla NER, company mode, batch, real
    DB/auth) + "Thank you / Questions" with the cheat-sheet numbers on the corner.

> Tip: every "data point" used on a slide has its full evidence in §6-12 of this file; exact
> screenshots exist in `docs/final_report.md`, `docs/extraction_audit.md`, `docs/classifier_v3_hybrid.md`,
> `docs/research_bangla_cv_support.md`, and the app's UI tabs (History / JD Match / Ranking).
> Tune block C/D/E to your grading focus: if it's a model-heavy class, give slides 15-19 extra
> time; if it's an industry/project course, spend more on 2-3 and 18-20.

---

## 15. Quick answers for interview-style questions (sneak cheat sheet)

- **Q: Which NLP techniques did you use?** NER (spaCy + fine-tuned DistilBERT), keyword/phrase
  matching (PhraseMatcher), regex, rule-based IE, TF-IDF + XGBoost text classification,
  sentence-embedding cosine similarity, contrastive fine-tuning, Okapi BM25 lexical
  ranking, generative LLM (LoRA) JSON extraction.
- **Q: Why not just GPT-4/other API?** We chose a self-hosted GPU LoRA fine-tune: no per-CV
  cost, no privacy leaks of CVs, teacher-demonstrable custom training, CPU-path also deployed.
- **Q: Is it deterministic?** Rule spelling yes (deterministic, explainable). LLM results
  are sanitized via grounding (only in-text tokens) + `json-repair`.
- **Q: How reproducible?** Benchmark CVs = generated, baseline in `demo/benchmark/_baseline.json`,
  all tests in `pytest tests/`, metrics re-computable `scripts/week7_eval.py`.
- **Q: Best improvement you made?** Data-point: the multi-entry experience fix (DOCX had
  no blank lines) → 46.4 → 53.6 → 56.4 benchmark; and ConFit embedder → ρ 0.314→0.436.
- **Q: Latency?** default extraction ~0.5–2 s/CV; NER ~40–90 ms; LLM ~27 s/CV; match ~35–60 ms per
  original pair + one 10 s warm-up.
- **Q: What would you do better?** collect cross-domain real NER eval set, deploy to Docker,
  maybe try ChatGPT large models. Fine-tune for out-domain NER.

---

*File location: `presentation_knowledge.md` (repo root). Source facts verified against repo
docs/metrics as of 2026-08-09.*