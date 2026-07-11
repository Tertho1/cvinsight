"""Quick smoke test of adapters on small samples."""
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.adapters import adapt_netsol
from src.extractor.extractor import extract_all
from src.parser.section_splitter import split_sections

# Test NETSOL with 5 rows
print("=== NETSOL (5 rows) ===")
df = pd.read_csv('data/processed/netsol_clean.csv').head(5)
for i, row in df.iterrows():
    sections, text = adapt_netsol(row.to_dict())
    cv = extract_all(text, sections=sections)
    edu_years = [e.get('year') for e in cv.get('education', [])]
    print(f"  [{i}] {cv['name']}: edu={len(cv['education'])} skills={len(cv['skills'])} years={edu_years}")

# Test NER with 3 rows
print("\n=== NER (3 rows) ===")
ner = pd.read_csv('data/processed/ner_resumes_clean.csv').head(3)
for i, row in ner.iterrows():
    cv = extract_all(str(row['text']), sections=split_sections(str(row['text'])))
    print(f"  [{i}] name={cv['name']} skills={len(cv['skills'])} edu={len(cv['education'])} exp={len(cv['experience'])}")

# Test ATS with 3 rows
print("\n=== ATS (3 rows) ===")
ats = pd.read_csv('data/processed/ats_scores_clean.csv').head(3)
for i, row in ats.iterrows():
    cv = extract_all(str(row['text']), sections=split_sections(str(row['text'])))
    print(f"  [{i}] name={cv['name']} skills={len(cv['skills'])} edu={len(cv['education'])} exp={len(cv['experience'])}")
