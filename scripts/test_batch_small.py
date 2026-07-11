"""Test batch extraction on small samples from each dataset."""
import sys, os, json, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore", category=SyntaxWarning)
import pandas as pd
from src.extractor.extractor import extract_all
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.parser.section_splitter import split_sections

PROCESSED = "data/processed"

datasets = [
    ("netsol", "netsol_clean.csv", adapt_netsol),
    ("ner", "ner_resumes_clean.csv", adapt_ner),
    ("ats", "ats_scores_clean.csv", adapt_ats),
    ("classification", "classification_clean.csv", adapt_classification),
]

samples = []
for name, fname, adapter_fn in datasets:
    filepath = f"{PROCESSED}/{fname}"
    if not os.path.exists(filepath):
        print(f"SKIP {name}: {filepath} not found")
        continue
    df = pd.read_csv(filepath).head(20)
    for _, row in df.iterrows():
        try:
            sections, text = adapter_fn(row.to_dict())
            if not sections and text:
                sections = split_sections(text)
            cv = extract_all(text, sections=sections)
            cv["_dataset"] = name
            samples.append(cv)
        except Exception as e:
            pass

print(f"Extracted {len(samples)} sample CVs")
ds = {}
for cv in samples:
    ds[cv.get("_dataset", "?")] = ds.get(cv.get("_dataset", "?"), 0) + 1
print(f"By dataset: {ds}")

for name in [d[0] for d in datasets]:
    subset = [c for c in samples if c.get("_dataset") == name]
    if not subset:
        continue
    names_found = sum(1 for c in subset if c.get("name"))
    skills_avg = sum(len(c.get("skills", [])) for c in subset) / len(subset)
    edu_avg = sum(len(c.get("education", [])) for c in subset) / len(subset)
    exp_avg = sum(len(c.get("experience", [])) for c in subset) / len(subset)
    lang_avg = sum(len(c.get("languages", [])) for c in subset) / len(subset)
    print(f"  {name}: name={names_found}/{len(subset)} skills={skills_avg:.1f} edu={edu_avg:.1f} exp={exp_avg:.1f} lang={lang_avg:.1f}")
