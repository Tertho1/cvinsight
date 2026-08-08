# Bangla / Bengali CV Support — Research Review
### Feasibility scan: datasets, models, and a recommended route for cvinsight (2026-08-05)

**Context this answers:** `progress.md:631` outlines a three-phase Bangla plan (Phase 1 English fine-tune → Phase 2 translate 200-500 Bangla CVs to English → Phase 3 native via B-NER / Onneshon / Sangraha / celloscopeai). This report verifies each claimed resource actually exists, adds what's missing, checks the toolchain against cvinsight's architecture, and gives a concrete recommendation. §8 records an empirical build of an Onneshon-trained section classifier.

---

## 0. Bottom line / recommendation

cvinsight is a **rule-first hybrid** system (`regex + EntityRuler` as primary, one small fine-tuned distilbert `ner-v1` as support). Bangla is a different writing system with weak positional cues — the English regexes (title/company/date patterns, degree aliases) will largely **not transfer**. A native Bangla path is therefore more rewrite than "add a language," and it buys limited value unless the target user base is actually Bengali-language CVs.

**Recommended (most value per effort, in order):**

1. **Phase 2 shortcut — translate-to-English is the pragmatic first cut.** The English pipeline is battle-tested and the matcher already embeds multilingual text (bge/matcher-confit handle Bengali decently). Feeding an LLM-translated Bangla CV into the existing extractor works immediately with near-zero code risk. Use `IndicTransv2` (free, local, 22 Indic languages incl. `ben`) rather than a cloud API.
2. **Phase 3 native, only if needed** — and don't build a Bangla resume parser from scratch. Freely available Bangla NER + parsing resources exist (below). The realistic native units are: (a) Bangla NER for entity spans, (b) `bnlp-toolkit` for Bangla tokenization/lemmatization/number normalization, (c) Bangla OCR only for scanned images.
3. **Watch Banglish (Latin-script Bengali).** Many "Bangla" CVs are actually written in Latin letters (e.g. "Shohoz Ltd", "Kono Software"). Detect Banglish and normalize to Bangla script (or keep Latin) before extraction — this is a distinct sub-problem from Devanagari-script Bangla.
4. **Defer.** If there is no confirmed Bengali-language CV demand, the 3-phase plan is currently "designed but not built" (`docs/final_report.md:157`) and should stay that way. Everything below is enablement, not a requirement.

---

## 1. Verifying the resources named in the plan

| Claimed resource | Verdict | Reality |
|---|---|---|
| **Onneshon** (Mendeley, hybrid Bengali resume, section-labeled) | ✅ exists | Mendeley DER DOI `10.17632/4md7bx6fd7.1`; 1,739 segments / 100 resumes (50 synthetic + 50 manually labelled); labels Experience 823 / Skill 446 / Education 370 / Objective 100; CSV format. Directly resume-relevant and is the closest thing to a "Bangla resume structure" dataset. |
| **B-NER** (Kaggle) | ✅ exists | IEEE 10103464. **Largest general Bangla NER dataset**, but covers the standard PER / ORG / LOC / TIM / EVT / MAT set — **not** resume-specific entities (no SKILL/DEGREE/TITLE labels). Good for a generic Bangla NER baseline, not for CV fields directly. ~Kaggle mirror: `mdzahidulhaquealvi/b-ner`. |
| **AI4Bharat Sangraha** | ✅ exists, but **pre-training only** | `ai4bharat/sangraha`, 251B tokens over 22 Indic langs, CC-BY-4.0. Bengali (`ben`) has Verified 10.6B + Synthetic 13.8B + Unverified 5.6B ≈ **30B tokens**. It is a *pretraining corpus* for language models — useless for a small extractor unless we train/fine-tune a Bangla LM from it. Not a labeled CV resource. Its realistic use is pretraining (or more cheaply, as evidence the ecosystem is healthy). |
| **celloscopeai/bangla_ner_dataset** | ✅ exists, but **person-name only** | HF `celloscopeai/bangla_ner_dataset`; a *person-name extraction* dataset (PER only) — matches the "Name field" annotation. Code repo `VirusProton/Bangla-Person-Name-Extractor`. A related `celloscopeai/celloscope_28000_bangla_ner_dataset` is the broader 28k-sample NER set. |

**Plan correction:** B-NER gives *generic* entities, celloscopeai gives *names only*; **neither labels CV fields (SKILL / DEGREE / COMPANY / TITLE)**. If we want a native resume-NER, we must label Bangla CVs ourselves or reuse Onneshon segments. This is the single most important finding: the plan's "native entity recognition" step is not turnkey.

---

## 2. Bangla NER datasets beyond the plan (all public)

- **BanNERD** (ACL Findings NAACL 2025, `eblict-gigatech/BanNERD`) — curated benchmark, 10 NER classes, high annotation quality (IAA 0.88), cross-dataset eval showed it generalizes best among Bangla NER sets. Best-quality generic Bangla NER source. 10 classes ≈ substantially richer than B-NER.
- **ANCHOLIK-NER** (arXiv 2502.11198) — 17,405 sentences across 5 regional dialects (Barishal, Chittagong, Mymensingh, Noakhali, Sylhet). Useful *only* for dialect robustness, and findings show mBERT does best (F1 ~82.6% on Mymensingh). Niche.
- **Naamapadam** (`ai4bharat/naamapadam`) — Indic NER covering 11 languages incl. `bn`, CC-BY-SA. Another generic PER/ORG/LOC baseline.
- **BNLP POS** (`banglanlp/bnlp-resources`) — POS/wordlist; supplies the tokenizer/lemmatizer.
- **Stanza Bangla NER tutorial** (Stanford) — documents an end-to-end Bangla NER train path (BIO → JSON → train) using `sagorsarker/bangla-bert-base`; a ready-made recipe if we later train native. https://stanfordnlp.github.io/stanza/new_language_ner.html

**No public Bangla *resume-NER* (SKILL/DEGREE/TITLE/COMPANY) dataset exists** — confirmed by absence across all the above. Onneshon is section-level, not token/span-level for these fields. So a clean native resume-NER dataset must be **created** (annotate or adapt). See §5 cost.

---

## 3. Bangla models on the shelf (HuggingFace)

| Model | Notes | Fit |
|---|---|---|
| **sagorsarker/bangla-bert-base** | Pretrained Bangla BERT; used in Stanza's NER recipe. | generic token classification |
| **csebuetnlp/banglabert** (ELECTRA-style, ~110M) | State-of-the-art on BLUB; over multilingual baselines e.g. mBERT/XLM-R. Includes token-classification fine-tune code + `bnlp-toolkit` (sentence segmentation, **normalization**, lemmatizer). | token NER + text normalization |
| **csebuetnlp/banglishbert** | For **Banglish** (Latin-script Bengali) — bows directly to the Banglish sub-problem above. | Banglish handling |
| **sagorsarker/multilingual-e5 / xlm-r / mBERT** | General multilingual embedders. Already what `src/matcher/embedder.py`'s `_DEFAULT_MODEL` (`models/matcher-confit`, bge-based) leans on — these take Bengali text today with no code change. | matcher side, zero-effort |
| **IndicTransv2** (AI4Bharat) | 22-language translation EN↔BN, free/local. | Phase-2 translate route |
| **Qwen3 (0.6B fine-tuned)** | cvinsight already fine-tuned this for extraction (English); Qwen3 natively speaks Bengali. | LLM-route both phases |

**Key gap:** cvinsight's *extraction* side is regex-first and English-shaped; its *matcher* side is multilingual-capable already. So the cheap win (translate) touches only the file-parsing/extraction input, while the expensive win (native) requires rebuilding extraction regexes + training a Bangla resume-NER.

---

## 4. Bangla OCR / document layer (for scanned Bangla CVs)

- cvinsight already ships `easyocr` in the parser (`models/ner-v1`, torch+easyocr are the cause of the Streamlit OOM — AGENTS.md). easyocr supports **Bengali** natively, so scanned-image Bangla OCR is *already available* with no new dependency.
- Dedicated Bangla OCR corpora: **BN-HTRd** (Mendeley `743k6dm543/4`, document-level handwritten), **Bengali handwritten grapheme** (Kaggle CV-19), Bangla text detection/recognition (Kaggle). Useful only if handwriting becomes in scope — almost certainly not for CVs.
- AI4Bharat `Indic-OCR` is "coming soon" (still in-progress). easyocr is the pragmatic choice.

---

## 5. Effort / risk assessment for the three real options

**A. Translate-to-English (Phase 2, recommended first).**
- Effort: low–moderate. Add IndicTransv2 (or an LLM) call into the parse path when detected Bangla; feed English text to existing `extract_all()`.
- Risk: low. No changes to extractor/matcher. Translation loss on company/skill names that should stay Bangla (mitigate by keeping named-entity spans untranslated / using language detection).
- Value: immediately usable ranking/scoring for Bengali candidates with the already-proven English metrics.

**B. Native Bangla extraction (Phase 3).**
- Effort: **high**. Needs (1) a language detector, (2) Bangla equivalents of every section header + date/degree regex, (3) a Bangla resume-NER — and the honest constraint from §2 is that **no labeled resume-NER dataset exists**, so we must annotate (or adapt Onneshon/BanNERD with a label mapping + resume text). Recommend `banglabert` + `bnlp-toolkit` + training a head, or an LLM (Qwen3) as the extractor for Bangla only.
- Risk: high, especially date/title/company parsing and Banglish. Banglish normalization needed before anything.
- Value: native UX and no-FP English-only bias; correctness = best only after right datasets are built.

**C. Defer (current reality).**
- Matches `docs/final_report.md:157-158` ("designed but not built"). Publish and validate the English product first; keep §1-§4 as the enablement map if demand materializes.

---

## 6. Concrete next actions (if we proceed)

1. Demo demand: confirm Bengali-language CVs are in scope (language distribution of target users).
2. If yes → **Option A**: add `bangla-detector` (e.g. `langcodes`/`lingua` or a Bangla-script Unicode range check) at parse time; wire IndicTransv2 translation into the Bangla path; keep original Bangla appended for reference. Measure demo/benchmark means to confirm parity.
3. If native required → build a small labeled Bangla resume set (Onneshon segments + hand labels into SKILL/DEGREE/TITLE/COMPANY), train a `banglabert` token-classification head (recipe in Stanza guide), and add Banglish→Bangla normalization via `bnb-bert`/`bnlp-toolkit`. Reuse easyocr for scanned images (already present).

---

## 7. Source index

- Onneshon — Mendeley DOI `10.17632/4md7bx6fd7.1` (1,739 segments, 100 resumes, CSV).
- B-NER — IEEE 10103464; Kaggle `mdzahidulhaquealvi/b-ner`.
- Sangraha — `ai4bharat/sangraha`, arXiv 2403.06350, CC-BY-4.0 (ben ≈ 30B tokens).
- celloscopeai person-name — `celloscopeai/bangla_ner_dataset`; code `VirusProton/Bangla-Person-Name-Extractor`; broader `celloscopeai/celloscope_28000_bangla_ner_dataset`.
- BanNERD — ACL Findings NAACL 2025; `eblict-gigatech/BanNERD`; macro-F1 81.85% / 10 classes.
- ANCHOLIK-NER — arXiv 2502.11198; 17,405 sentences / 5 dialects.
- BanglaBERT — arXiv 2101.00204, `csebuetnlp/banglabert`, BLUB, `bnlp-toolkit`.
- BanglishBERT — `csebuetnlp/banglishbert`.
- Sangraha ecosystem / Indic resources — `sabbirhossainujjal/Awesome_Bangla_Datasets` (comprehensive index of public Bangla NLP datasets).
- Stanza Bangla NER training recipe — stanfordnlp.github.io/stanza/new_language_ner.html.
- OCR: BN-HTRd (Mendeley `743k6dm543/4`); easyocr (already a cvinsight dependency, Bengali-capable).

---

## 8. Empirical build — Onneshon section classifier (2026-08-05)

Onneshon's CSV was obtained (`data/raw/onneshon_raw.csv`) and analyzed, then a
classifier built + tested. Findings:

**Data.** 1,739 segments / 2 cols (`text`, `label`), no nulls/no empty texts, matching
the paper's composition exactly (Experience 823 / Skill 446 / Education 370 / Objective
100). **347 exact-duplicate text rows → 1,392 unique**, and every duplicate text carries a
single consistent label (no label conflict), so dedup is leakage-safe — this dedup is
required before any train/val split or the same skill/date fragment leaks across folds and
inflates accuracy. Text quirks: 176 leftover `|` pipe chars (concatenation artifacts that
are actually a strong Experience cue), 123 bare slashes, 14.9% Latin-script tech terms.

**Model.** `scripts/train_bangla_section_classifier.py` → `models/bangla_section_classifier.pkl`
(char n-gram TF-IDF, `char_wb (1,3)` + LR, `class_weight="balanced"`). No BanglaBERT needed —
the classes are textually very distinct:

| Metric | Value |
|---|---|
| 5-fold CV accuracy | **0.9454** (0.931–0.961) |
| 5-fold CV macro-F1 | **0.952** |
| Held-out accuracy (25%) | **0.922** vs majority baseline 0.520 |
| Per-class held-out F1 | Education 0.949, Experience 0.928, Objective 1.000, Skill 0.874 |

Skill↔Experience is the only real confusion (short tech fragments); the rest are
near-clean. Metrics: `models/bangla_section_eval.json`.

**Module.** `src/extractor/bangla_section.py` — lazy-loaded `BanglaSectionClassifier`
(mirrors `src/matcher/embedder.py`), singleton `get_bangla_section_classifier()` and
`classify_section()` convenience. Maps Onneshon labels → CVSchema canonical sections
(`Objective`→`summary`, `Experience`→`experience`, `Skill`→`skills`, `Education`→`education`).
Returns `None` gracefully when the model file or text is missing. Tests: 13 in
`tests/test_bangla_section.py`; full suite 410 passing (397 + 13).

**Scope confirmed.** This classifier does **section detection only** — it maps a Bangla
fragment to one of four sections and does not extract company/date/degree spans, so it is a
building block for native Bangla sectioning (beside `section_splitter.py`), not end-to-end
Bangla extraction. Entity extraction + scoring still need their own Bangla handling (§5-B).

**Status (updated 2026-08-08).** The Bangla route shipped — `extract_all()` now detects Bangla
script (`is_bangla` in `src/extractor/bangla_extractor.py`) and routes to `extract_bangla()`:
it transliterates Bengali digits, months, date markers, degree words, spoken languages and
section headings (with the Onneshon classifier as the sectioning fallback) so the existing
English extractors fire; Latin tech terms, emails and phones pass through untouched. The app
shows a "Language: Bangla" badge and skips English NER/ML for Bengali CVs. This is the 
*section-structure + transliteration* route ("Option 1" in the session's native rule-based
decision), which sits between the research doc's §5 Option A (translate) and Option B (native NER).
Residual polish for full §5-B native extraction (Bangla company/title regexes, Banglish
normalization, Bangla resume-NER) is tracked in `TODO.md` Phase 3.

---

## 9. Ready-made Bangla NER models on HuggingFace (2026-08-05)

Checked the model hub for off-the-shelf Bangla NER that could supply the entity layer a
native (Phase-3) path still lacks. The short answer: several exist, but **all are trained on
news/wiki domains, none is resume-domain** — no SKILL/DEGREE/COMPANY labels anywhere. They are
usable as generic entity support (name/location/org/date spans) inside a Bangla path, not as
a turnkey CV extractor.

| Model (HF id) | Base / task | Labels | Reported F1 | Params | Fit for cvinsight |
|---|---|---|---|---|---|
| **arafatfahim/BanglaTag** | csebuetnlp/banglabert (ELECTRA), B-NER (22,144 news sentences) | PER, LOC, ORG, POL, DATE, TIME, EVENT, CRIME, TITLE, NUM, SYMBOL, CONSTITUENCY, INST (BIO) | 0.749 overall; PER 0.745, DATE 0.764, ORG 0.565, TITLE 0.959, INST 0.737 | 0.1B | **Best domain fit** of the ready-mades — has DATE/ORG/INST/TITLE. But TITLE = news designations (মহাপরিদর্শক "Inspector General"), not job titles; ORG F1 0.565 is weak; still no SKILL/DEGREE. apache-2.0. |
| **sagorsarker/mbert-bengali-ner** | bert-base-multilingual-uncased, wikiann `bn` | PER, ORG, LOC | 0.971 | 0.2B | Highest-F1 generic NER for Bangla names/orgs/locations. Only 3 labels; no dates/titles. MIT. 462 downloads/mo — the community standard. |
| **Suchandra/bengali_language_NER** | bert-base-multilingual-cased, wikiann `bn` | PER, ORG, LOC | 0.967 | 0.2B | Same 3-label shape as above; slightly lower F1. Good for a quick name/org sanity signal. |
| **Davlan/xlm-roberta-base-wikiann-ner** | XLM-R, 20 langs incl. `bn` | PER, ORG, LOC | high on wikiann | 0.3B | Multilingual; covers bn but tuned for cross-lingual transfer, not Bangla-first. Heaviest of the group. |
| **saiful9379/BanglaNER** | spaCy transformer pipeline, PER only | PER | ~0.81 | spaCy | Person-name only (like celloscopeai dataset). Marginal — our name field is already regex+rule handled; an extra 0.8-F1 model adds little. |
| **csebuetnlp/banglabert** / **sagorsarker/bangla-bert-base** | Pretrained ELECTRA / BERT bases (MLM/RTD) | — (not NER-tuned) | — | 0.11B / 0.11B | **Base models for fine-tuning**, not off-the-shelf NER. These are the right starting points if we train our own resume-NER head (§5-B), not drop-in extractors. |

**Assessment for CV support:**
- **Name / organization / location:** `sagorsarker/mbert-bengali-ner` (F1 0.971) or
  `Suchandra/bengali_language_NER` — useful to bootstrap a Bangla name/company candidate
  layer cheaply. Modest gain over regex because Bangla CVs often write these in Latin anyway.
- **Dates:** `arafatfahim/BanglaTag` is the only ready-made with a DATE tag (F1 0.764). A
  Bangla CV date-range finder (e.g. `জানুয়ারি ২০২০ - বর্তমান`) would need Bangla month names
  regardless; the model could *support* a regex-first date layer, mirroring how `ner-v1`
  supports the English extractor.
- **Skills / degrees / job titles:** **no ready-made model covers these.** Resume TITLE ≠ news
  TITLE; ORG F1 0.565 in BanglaTag is too weak to trust as company. This confirms §2 again:
  the high-value Bangla CV fields still require either the translate route (Phase 2) or a
  custom-labeled Bangla resume-NER (Phase 3). Ready-made NER only shrinks the custom part.
- **Runtime constraint:** all of these are 0.1–0.3B transformers. cvinsight already OOMs on
  Streamlit Cloud (1GB) with torch+easyocr and keeps its only model on CPU. A Bangla
  token-classifier adds the same weight class as the LLM hybrid that was dropped for being too
  slow on CPU (~27 s/CV). If we use one, it must be lazy-loaded and limited to a candidate-
  generation role, not a per-CV mandatory pass.

**Bottom line:** ready-made Bangla NER exists but is generic (news/wiki PER/ORG/LOC + a weak
DATE/ORG in BanglaTag). It cannot deliver the CV entity layer alone. Recommended order stands:
Phase-2 translate for real scores now; Phase-3 native = our own Bangla resume-NER fine-tuned
from `csebuetnlp/banglabert` or `sagorsarker/bangla-bert-base`, with a ready-made
wikiann/mBERT model only as a bootstrap labeler or a name/org candidate layer.