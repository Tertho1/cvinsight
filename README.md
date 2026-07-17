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

Upload a CV (PDF/DOCX/TXT) → extract information → score against rubric → classify quality (ML) → get improvement suggestions.

## Features

- **Parsing** — PDF (text + scanned/OCR), DOCX, TXT
- **Extraction** — NER + rule-based for education, experience, skills, projects, certifications, languages
- **Scoring** — Rubric-based scoring with configurable weights
- **ML Classification** — TF-IDF + XGBoost (87.65% accuracy)
- **Suggestions** — Actionable improvement tips per section
