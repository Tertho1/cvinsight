"""Test Phase 2 adapters work correctly."""
import sys, os, json
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.extractor.extractor import extract_all

# --- Test NETSOL adapter ---
print("=== NETSOL adapter ===")
netsol = pd.read_csv('data/processed/netsol_clean.csv')
row = netsol.iloc[0].to_dict()
sections, text = adapt_netsol(row)
print(f"Sections keys: {list(sections.keys())}")
for k, v in sections.items():
    print(f"  {k}: {str(v)[:100]}")
print(f"Text: {text[:150]}")

cv = extract_all(text, sections=sections)
print(f"Name: {cv['name']}")
print(f"Skills count: {len(cv['skills'])}")
print(f"Education count: {len(cv['education'])}")
if cv["education"]:
    print(f"  First edu: {cv['education'][0]}")
print()

# --- Test NER adapter ---
print("=== NER adapter ===")
ner = pd.read_csv('data/processed/ner_resumes_clean.csv')
row2 = ner.iloc[0].to_dict()
sections2, text2 = adapt_ner(row2)
print(f"Text length: {len(text2)}")
print(f"Sections: {sections2}")
print()

# --- Test ATS adapter ---
print("=== ATS adapter ===")
ats = pd.read_csv('data/processed/ats_scores_clean.csv')
row3 = ats.iloc[0].to_dict()
sections3, text3 = adapt_ats(row3)
print(f"Text length: {len(text3)}")
print(f"Sections: {sections3}")
print()

# --- Test Classification adapter ---
print("=== Classification adapter ===")
cls = pd.read_csv('data/processed/classification_clean.csv')
row4 = cls.iloc[0].to_dict()
sections4, text4 = adapt_classification(row4)
print(f"Text length: {len(text4)}")
print(f"Sections: {sections4}")

print("\nAll adapters OK")
