# CV Evaluator & Ranking System

An automated CV evaluation, scoring, and ranking tool built with **Python 3.14**, **spaCy**, **XGBoost**, **Sentence-Transformers**, and **Streamlit**.

Upload a CV (PDF, DOCX, or TXT) and get a structured evaluation: extracted entities, section scores (0-100), a quality label (Strong/Average/Weak), and actionable improvement suggestions. Optionally add a job description to compute match percentage and rank multiple candidates.

**Version architecture:**
- **V1** — Upload → Extract → Score → Suggest
- **V2** — Add JD → Compute match % → Rank CVs
- **V3** — Fine-tune NER on self-labeled data (stretch)

---

## Project Status

| Week | Area | Status |
|------|------|--------|
| W1 | Foundation & datasets | ✅ Complete |
| W2 | PDF/DOCX/TXT parser + OCR | ✅ Complete |
| W3 | NER extraction (3000 CVs, 285 tests) | ✅ ~90% |
| W4 | Scoring engine & suggestion generator | ❌ Not started |
| W5 | Classifier training & Streamlit V1 | ❌ Not started |
| W6 | JD matching & ranking (V2) | ❌ Not started |
| W7 | Fine-tuning & final report | ❌ Not started |

See `progress.md` for detailed status and `project_plan.md` for the full roadmap.

---

## Features

- **Multi-format parsing** — PDF (with 3-strategy fallback + OCR for scanned docs), DOCX, and TXT
- **Entity extraction** — Name, email, phone, education, experience, skills, projects, certifications, languages, achievements, leadership
- **Scoring rubric** — Configurable weights via `config/rubric_config.json` for 7 sections
- **Quality classification** — Strong / Average / Weak based on total score
- **JD matching** — Semantic similarity + skill overlap + rubric score → final match %
- **CV ranking** — Rank multiple candidates against a single job description
- **Improvement suggestions** — Section-specific tips for underperforming areas

---

## Setup

```bash
# Clone and enter
git clone <repo-url>
cd cvinsight

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Datasets

The project uses 5 public datasets stored in `data/raw/`:

| Dataset | Size | Purpose |
|---------|------|---------|
| Mehyaar/Annotated_NER_PDF_Resumes | 5,029 CVs | NER model training |
| datasetmaster/resumes | ~3,000 CVs | Parser dev & structured JSON |
| noran-mohamed/Resume-Classification-Dataset | ~13,000 CVs | Category classification |
| 0xnbk/resume-ats-score-v1-en | 6,374 pairs | JD matching |
| netsol/resume-score-details | 1,031 pairs | Scored resume-JD examples |

---

## Run Tests

```bash
# Run all 285 unit tests
pytest tests/
```

---

## Run the App

```bash
# When Streamlit V1 is built:
streamlit run app/app.py
```

---

## Project Structure

```
cvinsight/
├── app/                  # Streamlit UI (coming in Week 5)
├── config/               # rubric_config.json, skill_taxonomy.json
├── data/
│   ├── raw/              # Original datasets (do not edit)
│   └── processed/        # Cleaned CSV/JSON files
├── demo/                 # Sample CVs & demo script (coming in Week 7)
├── models/               # Trained .pkl files (coming in Week 5)
├── notebooks/            # EDA & evaluation notebooks
├── scripts/              # Utility scripts
├── src/
│   ├── parser/           # PDF, DOCX, TXT, OCR parsing
│   ├── extractor/        # NER + rule-based entity extraction
│   ├── scorer/           # Rubric scoring engine (coming in Week 4)
│   ├── matcher/          # JD similarity & ranking (coming in Week 6)
│   └── suggester/        # Feedback generation (coming in Week 4)
├── tests/                # 285 unit tests
├── progress.md           # Weekly progress tracking
├── project_plan.md       # Full week-by-week roadmap
└── requirements.txt      # Python dependencies
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| PDF parsing | pdfplumber, pdfminer, pypdf |
| DOCX parsing | python-docx |
| OCR fallback | pytesseract |
| NER | spaCy (en_core_web_sm), Hugging Face transformers |
| Scoring | scikit-learn, XGBoost |
| Sentence embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Similarity | sklearn cosine_similarity |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

---

## License

MIT
