---
title: CV Insights
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: ">=1.44"
app_file: streamlit_app.py
pinned: false
---

# CV Evaluator & Quality Classifier

Upload a CV (PDF/DOCX/TXT) → extract information → score against rubric → classify quality (ML) → get improvement suggestions. Optional job-description matching & multi-CV ranking.

## Features

- **Parsing** — PDF (text + scanned/OCR), DOCX, TXT
- **Extraction** — rule-based + spaCy NER, optional fine-tuned DistilBERT (fast) and Qwen3-0.6B LoRA LLM (deep) fusion; native Bengali-script CV route (transliteration + Bangla NER)
- **Scoring** — configurable rubric (rubric_config.json) with per-criterion rationales
- **ML Classification** — hybrid classifier (engineered features + embeddings + XGBoost, trained on a synthetic corpus; 7/10 benchmark rubric agreement)
- **Suggestions** — actionable improvement tips per section
- **Matching & Ranking** — JD similarity (semantic + skill overlap), multi-CV ranking

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py     # or python -m streamlit run app/app.py
```

## Testing

```bash
python -m pytest tests/            # 482 tests
```