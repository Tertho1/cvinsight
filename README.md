---
title: CV Evaluator & Quality Classifier
emoji: 📄
colorFrom: indigo
colorTo: emerald
sdk: streamlit
sdk_version: 1.44.1
app_file: streamlit_app.py
pinned: false
license: mit
---

# CV Evaluator & Ranking System

Upload a CV (PDF/DOCX/TXT) → extract structured info → score against rubric → classify quality → get improvement tips.

## NLP Techniques Used

| Technique | Implementation |
|-----------|---------------|
| Named Entity Recognition | spaCy EntityRuler + PhraseMatcher |
| Information Extraction | Rule-based section parsers |
| Keyword Extraction | Skill taxonomy PhraseMatcher |
| Text Classification | TF-IDF + XGBoost (87.65% accuracy) |
| Semantic Similarity | sentence-transformers (Week 6) |

## Pipeline

```
CV → parser → section_splitter → extract_all() → score_cv() → label + suggestions
                                                                  ↓
                                                          TF-IDF + XGBoost → ML label
```

## Project Structure

```
├── app/              # Streamlit UI
├── config/           # Rubric & skill taxonomy JSON
├── data/processed/   # Cleaned datasets + training data
├── demo/             # Sample CVs
├── models/           # Trained .pkl files (XGBoost, LR)
├── scripts/          # Pipeline scripts
├── src/              # Source code
│   ├── parser/       # PDF/DOCX/TXT/OCR parsing
│   ├── extractor/    # NER + rule-based extraction
│   ├── scorer/       # Rubric scoring engine
│   ├── matcher/      # JD similarity (Week 6)
│   └── suggester/    # Suggestion templates
└── tests/            # 343 unit tests
```

Built with Python 3.14, spaCy, XGBoost, sentence-transformers, Streamlit.
