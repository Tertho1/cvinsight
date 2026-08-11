---
title: CV-Insight
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: ">=1.44"
app_file: streamlit_app.py
pinned: false
---

# 📄 CV-Insight

**Turn raw resumes into ranked, explainable decisions.**

CV-Insight is an end-to-end CV evaluation engine: upload a CV, and it parses,
extracts structured entities, scores against a configurable rubric, classifies
quality with an ML hybrid, and suggests targeted improvements — then optionally
matches candidates against a job description and ranks them.

> English and **Bengali** CVs. Text, DOCX, and scanned PDFs (OCR).

---

## ✨ Highlights

- **Full pipeline in one flow** — parse → extract → score → classify → suggest → match → rank
- **Two extraction engines** — fast spaCy + DistilBERT NER, or deep Qwen3-0.6B LoRA LLM fusion
- **Native Bangla route** — script detection, digit/month/degree transliteration, Bangla NER + section classifier
- **ML quality classification** — hybrid classifier (engineered features + semantic embeddings + XGBoost), trained on a synthetic rubric corpus
- **JD matching & ranking** — semantic similarity (ConFit fine-tuned embedder) + skill overlap; rank whole candidate pools
- **Explainable scoring** — every section carries a criterion score, weight, method, and a plain-English rationale
- **Live skill search** — filter a stored candidate pool by skill, sorted by total score

---

## 🗺️ How it works

```
CV file (PDF/DOCX/TXT) ──▶ parser ──▶ section splitter ──▶ extract_all()
    │                                                                 │
    └── OCR for scanned pages          CVSchema (Pydantic) ◀──────────┘
                                          │
              ┌───────────────┬───────────┴──────────┬──────────────┐
              ▼               ▼                      ▼              ▼
          Scorer         ML classifier          Suggester       Matcher (optional)
      rubric weights     quality label        section tips      JD similarity + ranking
              │               │                      │              │
              └───────▶ Total score + label + rationale + ranked candidates ◀────┘
```

Everything reads and writes one **CVSchema** (Pydantic), so each stage is
independently testable and swappable.

## 🧩 Features

### Parsing
- PDF text extraction, **scanned PDF OCR** (EasyOCR), DOCX (table-aware), and plain TXT
- Robust text cleanup: `(cid:127)` markers, layout-broken lines, phone number normalization

### Extraction
- **Entity extraction** — name, email, phone, education, experience (multi-job, date-anchored), skills, projects, certifications, languages, achievements, leadership
- **Two model tiers** (selectable in the app):
  - ⚡ **Fast:** spaCy + DistilBERT NER (`models/ner-v1`), ~40–90 ms/CV
  - 🧠 **Deep:** spaCy + fine-tuned Qwen3-0.6B LoRA LLM fusion, ~27–32 s/CV (GPU)
- **Bangla route** — automatic Bengali-script detection; transliterates digits, months, degrees, and section headings so English extractors fire; labels the CV `language: "bangla"`
- Models auto-download from HuggingFace when local artifacts are absent

### Scoring
- Configurable rubric (`config/rubric_config.json`) across 7+ sections (experience, projects, skills, education, …)
- Per-criterion **score / max / weight / method / rationale** (`config/default_criteria.json`)
- Custom rubric weights adjustable live in the app

### Quality classification
- Hybrid classifier: TF-IDF + engineered features + DistilBERT semantic embeddings → XGBoost
- Trained on a synthetic rubric-labeled corpus; **7/10 benchmark rubric agreement**; beat plain TF-IDF baselines

### Suggestions
- Actionable, section-level improvement tips generated from extraction gaps

### Matching & ranking (V2)
- Semantic embedder (ConFit contrastive fine-tune, `models/matcher-confit`) + BM25 pre-filter + skill overlap
- Multi-CV ranking with explainable per-candidate JD match scores
- Ranker benchmarked with NDCG and Spearman metrics

### App UX
- 7 workspace tabs: Extracted Data, Suggestions, Raw Text, History, JD Match, Ranking, Skill Search
- Session-scoped CV database (safe on hosted deploys — no cross-user leaks), JSON/CSV export, one-click delete-all

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py        # production entry point (Streamlit Cloud)
# or
python -m streamlit run app/app.py    # dev entry point
```

Dependencies: `requirements.txt` (Streamlit, pandas, spaCy, torch, scikit-learn, xgboost, EasyOCR, HuggingFace Transformers).

> **Heads-up:** the first load downloads the NER / embedder / LLM checkpoints
> from HuggingFace into `models/` (gitignored). Requires ~2 GB free disk.

## ✅ Testing

```bash
python -m pytest tests/        # 482 tests — parser, extractor, scorer, suggester, matcher, hybrid, Bangla
```

## 🧪 Benchmark

`demo/benchmark/` holds 10 reproducible scenario CVs (two-column, table DOCX,
date-first, multi-degree, sparse, senior, …) with a baseline rubric —
`scripts/generate_benchmark_cvs.py` regenerates them.

## 📊 Datasets

The models were trained and evaluated on five public resume/JD datasets
(~26,000 CVs total), plus a synthetic corpus for the classifier.

### Download links & storage paths

Place the raw files under `data/raw/` (the folder is gitignored), then run
`python scripts/clean_datasets.py` to produce the cleaned `data/processed/*.csv`
versions the training/eval scripts expect.

| Dataset | CVs | Source | Raw path |
|---|---|---|---|
| DatasetMaster | ~4,800 | [datasetmaster/resumes](https://huggingface.co/datasets/datasetmaster/resumes) (HF) | `data/raw/datasetmaster_raw.csv` |
| NER resumes | ~3,300 | [Mehyaar/Annotated_NER_PDF_Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes) (HF) | `data/raw/ner_resumes_raw.csv` |
| Classification | ~12,100 | [noran-mohamed/Resume-Classification-Dataset](https://github.com/noran-mohamed/Resume-Classification-Dataset) (GitHub) · fallback [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) (HF) | `data/raw/classification_raw.csv` |
| ATS scores | ~5,000 | [0xnbk/resume-ats-score-v1-en](https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en) (HF) | `data/raw/ats_scores_raw.csv` |
| NETSOL | ~850 | [netsol/resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details) (HF) | `data/raw/netsol_raw.csv` |
| JD fit pairs | ~8,000 | [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) (HF) | loaded on demand by eval/training scripts |
| Synthetic rubric corpus | 1,500 | Generated by `scripts/generate_synthetic_corpus.py` | `data/curated/corpus_synth_v1.csv` |

Helper downloaders exist for the trickier sources: `scripts/load_netsol.py`,
`scripts/load_classification.py`, and `scripts/load_ner.py`.

## 🗂️ Repository layout

```
cvinsight/
├── app/           Streamlit UI
├── src/
│   ├── parser/    PDF/DOCX/TXT/OCR → text
│   ├── extractor/ rule + NER + Bangla + LLM → CVSchema
│   ├── scorer/    rubric scoring + criteria rationales
│   ├── matcher/   JD similarity, BM25, skill overlap, ranking
│   ├── classifier/ hybrid ML quality classifier
│   └── suggester/ improvement tips
├── config/        rubric_config.json, skill_taxonomy.json (~300 skills, 8 categories)
├── models/        trained .pkl artifacts + HuggingFace checkpoints
├── scripts/       training, eval, benchmark, and gating scripts
├── tests/         482 unit tests
└── demo/          sample CVs + benchmark scenario set
```

## 🏗️ Project status

- **V1** — extract → score → classify → suggest ✅
- **V2** — JD matching & multi-CV ranking ✅
- **V3** — NER fine-tuning, Bangla support, hybrid classifier, LLM extraction ✅
- **Deployment** — live on **Streamlit Cloud** (cvinsight1.streamlit.app). Plans to migrate to **Render.com / Railway** for a dedicated environment (more RAM than the 1 GB Streamlit Cloud tier).

---

Built for the CV Evaluator & Ranking capstone — structured, testable, and
designed to explain *why* a score is what it is.
