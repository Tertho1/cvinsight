# scripts/clean_datasets.py
#
# Cleans all 5 raw datasets and saves processed versions.
#
# Input  : data/raw/*.csv
# Output : data/processed/*.csv
#
# Cleaning operations applied to every dataset:
#   1. Strip HTML tags
#   2. Normalize whitespace (tabs, newlines → single space)
#   3. Remove non-printable / control characters
#   4. Drop duplicate rows (based on text content)
#   5. Drop rows where CV text is under 100 characters
#   6. Reset index after dropping rows

import os
import re
import json
import pandas as pd

os.makedirs('data/processed', exist_ok=True)


# ── Cleaning utilities ─────────────────────────────────────────

def strip_html(text: str) -> str:
    """
    Removes HTML tags like <b>, <br/>, <p> etc.
    Some CVs were scraped from web pages and carry HTML.
    We only want the readable text content.
    """
    if not isinstance(text, str):
        return ''
    return re.sub(r'<[^>]+>', ' ', text)


def normalize_whitespace(text: str) -> str:
    """
    Collapses all whitespace sequences (spaces, tabs, newlines)
    into a single space, then strips leading/trailing whitespace.
    
    Why? Because 'Python  \t  Developer' and 'Python Developer'
    are the same thing, but string matching would treat them
    differently without this step.
    """
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', text).strip()


def remove_control_chars(text: str) -> str:
    """
    Removes non-printable control characters (ASCII 0-31 except
    newline and tab which we handle via normalize_whitespace).
    These appear in PDFs and cause silent downstream errors.
    """
    if not isinstance(text, str):
        return ''
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def clean_text(text: str) -> str:
    """Master cleaning function — applies all steps in order."""
    text = strip_html(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    return text


def report(name: str, before: int, after: int, cols: list):
    """Prints a clean before/after summary for each dataset."""
    dropped = before - after
    print(f'\n{"─"*55}')
    print(f'  {name}')
    print(f'{"─"*55}')
    print(f'  Rows before : {before:,}')
    print(f'  Rows after  : {after:,}  (dropped {dropped:,})')
    print(f'  Columns     : {cols}')


# ══════════════════════════════════════════════════════════════
# DATASET 1 — NER Resumes
# Primary use: extractor training and NER model fine-tuning
# Key column: 'text' (raw CV text), 'annotations' (NER labels)
# ══════════════════════════════════════════════════════════════
print('\n[1/5] Cleaning NER Resumes...')

ner = pd.read_csv('data/raw/ner_resumes_raw.csv')
before = len(ner)

# Clean the text column
ner['text'] = ner['text'].apply(clean_text)

# Drop rows with empty or very short text
# A CV under 100 chars is either a parsing failure or a blank file
ner = ner[ner['text'].str.len() >= 100].copy()

# Drop exact duplicates based on text content
ner = ner.drop_duplicates(subset=['text']).reset_index(drop=True)

# Keep only the columns we need downstream
ner_clean = ner[['text', 'annotations', 'text_length']].copy()
ner_clean['text_length'] = ner_clean['text'].str.len()

ner_clean.to_csv('data/processed/ner_resumes_clean.csv',
                 index=False, encoding='utf-8')
report('NER Resumes', before, len(ner_clean),
       list(ner_clean.columns))
print(f'  Sample text : {ner_clean["text"].iloc[0][:100]}...')


# ══════════════════════════════════════════════════════════════
# DATASET 2 — Structured Resumes (datasetmaster)
# Primary use: extractor testing — already has structured fields
# Key columns: skills, experience, education, projects etc.
# We also build a 'text' column by flattening the JSON fields
# so the parser can also treat it as raw text if needed.
# ══════════════════════════════════════════════════════════════
print('\n[2/5] Cleaning Structured Resumes...')

res = pd.read_csv('data/raw/datasetmaster_raw.csv')
before = len(res)


def flatten_resume(row: pd.Series) -> str:
    """
    Converts the structured columns of a datasetmaster row
    into a single text string that looks like a CV.
    This lets us use it as raw text input for parser testing.
    """
    parts = []
    for col in ['personal_info', 'experience', 'education',
                'skills', 'projects', 'certifications',
                'achievements']:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ('', '[]', '{}', 'nan'):
            parts.append(str(val))
    return ' '.join(parts)


res['text'] = res.apply(flatten_resume, axis=1)
res['text'] = res['text'].apply(clean_text)

# Drop short rows
res = res[res['text'].str.len() >= 100].copy()
res = res.drop_duplicates(subset=['text']).reset_index(drop=True)
res['text_length'] = res['text'].str.len()

# Keep structured columns + synthesized text
keep_cols = ['text', 'text_length', 'personal_info', 'experience',
             'education', 'skills', 'projects', 'certifications',
             'achievements']
res_clean = res[keep_cols].copy()

res_clean.to_csv('data/processed/structured_resumes_clean.csv',
                 index=False, encoding='utf-8')
report('Structured Resumes', before, len(res_clean),
       list(res_clean.columns))
print(f'  Sample text : {res_clean["text"].iloc[0][:100]}...')


# ══════════════════════════════════════════════════════════════
# DATASET 3 — Classification Dataset (ResumeAtlas)
# Primary use: category labels for quality label generation (Week 4)
# Key columns: 'Category', 'Text'
# ══════════════════════════════════════════════════════════════
print('\n[3/5] Cleaning Classification Dataset...')

clf = pd.read_csv('data/raw/classification_raw.csv')
before = len(clf)

# Rename to standard names
clf = clf.rename(columns={'Text': 'text', 'Category': 'category'})
clf['text'] = clf['text'].apply(clean_text)
clf = clf[clf['text'].str.len() >= 100].copy()
clf = clf.drop_duplicates(subset=['text']).reset_index(drop=True)
clf['text_length'] = clf['text'].str.len()

clf_clean = clf[['text', 'category', 'text_length']].copy()

clf_clean.to_csv('data/processed/classification_clean.csv',
                 index=False, encoding='utf-8')
report('Classification Dataset', before, len(clf_clean),
       list(clf_clean.columns))
print(f'  Categories  : {clf_clean["category"].nunique()} unique')
print(f'  Sample text : {clf_clean["text"].iloc[0][:100]}...')


# ══════════════════════════════════════════════════════════════
# DATASET 4 — ATS Scores (0xnbk)
# Primary use: JD matching evaluation in Week 6
# Key columns: 'text' (CV+JD combined), 'ats_score', 'original_label'
# ══════════════════════════════════════════════════════════════
print('\n[4/5] Cleaning ATS Scores...')

ats = pd.read_csv('data/raw/ats_scores_raw.csv')
before = len(ats)

ats['text'] = ats['text'].apply(clean_text)
ats = ats[ats['text'].str.len() >= 100].copy()
ats = ats.drop_duplicates(subset=['text']).reset_index(drop=True)
ats['text_length'] = ats['text'].str.len()

# Normalize ATS score to 0-1 range if it isn't already
if ats['ats_score'].max() > 1.5:
    print(f'  Score range before normalization: '
          f'{ats["ats_score"].min():.1f} – {ats["ats_score"].max():.1f}')
    ats['ats_score'] = ats['ats_score'] / ats['ats_score'].max()
    print(f'  Score range after normalization : '
          f'{ats["ats_score"].min():.3f} – {ats["ats_score"].max():.3f}')

ats_clean = ats[['text', 'ats_score', 'original_label',
                  'text_length']].copy()

ats_clean.to_csv('data/processed/ats_scores_clean.csv',
                 index=False, encoding='utf-8')
report('ATS Scores', before, len(ats_clean),
       list(ats_clean.columns))
print(f'  Score stats : mean={ats_clean["ats_score"].mean():.3f}, '
      f'min={ats_clean["ats_score"].min():.3f}, '
      f'max={ats_clean["ats_score"].max():.3f}')


# ══════════════════════════════════════════════════════════════
# DATASET 5 — Netsol Resume Scores
# Primary use: ranking calibration in Week 6
# We keep only 'match' and 'mismatch' records (drop empty/invalid)
# ══════════════════════════════════════════════════════════════
print('\n[5/5] Cleaning Netsol Resume Scores...')

net = pd.read_csv('data/raw/netsol_raw.csv')
before = len(net)

# Keep only meaningful record types
net = net[net['file_type'].isin(['match', 'mismatch'])].copy()
print(f'  Kept match+mismatch rows: {len(net):,} '
      f'(dropped {before - len(net):,} empty/invalid)')

# Clean text fields
net['job_description'] = net['job_description'].apply(clean_text)
net['skills']          = net['skills'].apply(
    lambda x: clean_text(str(x)) if pd.notna(x) else '')

# Drop rows with no job description
net = net[net['job_description'].str.len() >= 50].copy()

# Drop rows with no score
net = net[net['score'].notna()].copy()

net = net.reset_index(drop=True)

net_clean = net[['file_type', 'candidate_name', 'job_description',
                  'skills', 'education', 'experience',
                  'score', 'justification']].copy()

net_clean.to_csv('data/processed/netsol_clean.csv',
                 index=False, encoding='utf-8')
report('Netsol Resume Scores', before, len(net_clean),
       list(net_clean.columns))
print(f'  Score stats : mean={net_clean["score"].mean():.2f}, '
      f'min={net_clean["score"].min():.2f}, '
      f'max={net_clean["score"].max():.2f}')
print(f'  File types  : {net_clean["file_type"].value_counts().to_dict()}')


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f'\n{"═"*55}')
print('  CLEANING COMPLETE — data/processed/ contents:')
print(f'{"═"*55}')

processed_files = [
    'data/processed/ner_resumes_clean.csv',
    'data/processed/structured_resumes_clean.csv',
    'data/processed/classification_clean.csv',
    'data/processed/ats_scores_clean.csv',
    'data/processed/netsol_clean.csv',
]

for path in processed_files:
    if os.path.exists(path):
        df = pd.read_csv(path)
        size_kb = os.path.getsize(path) / 1024
        print(f'  ✓ {os.path.basename(path):<40} '
              f'{len(df):>6,} rows  {size_kb:>8.1f} KB')
    else:
        print(f'  ✗ {os.path.basename(path)} — MISSING')

print(f'{"═"*55}')
print('  Day 5 complete. Ready for Day 6.')
print(f'{"═"*55}')