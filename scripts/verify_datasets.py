# scripts/verify_datasets.py
# Confirms all 5 raw CSVs exist, have the right row counts,
# and can be loaded without errors.
# Run this at the end of Day 4 as the official sign-off check.

import os
import pandas as pd

datasets = [
    {
        'name':     'NER Resumes (Mehyaar)',
        'path':     'data/raw/ner_resumes_raw.csv',
        'min_rows': 5000,
        'purpose':  'NER model training — Week 3 & 7',
    },
    {
        'name':     'Structured Resumes (datasetmaster)',
        'path':     'data/raw/datasetmaster_raw.csv',
        'min_rows': 4000,
        'purpose':  'Parser testing — Week 2 & 3',
    },
    {
        'name':     'Classification Dataset (noran-mohamed)',
        'path':     'data/raw/classification_raw.csv',
        'min_rows': 1000,
        'purpose':  'Category labels — Week 4',
    },
    {
        'name':     'ATS Scores (0xnbk)',
        'path':     'data/raw/ats_scores_raw.csv',
        'min_rows': 500,
        'purpose':  'JD matching evaluation — Week 6',
    },
    {
        'name':     'Resume Score Details (netsol)',
        'path':     'data/raw/netsol_raw.csv',
        'min_rows': 1000,
        'purpose':  'Ranking calibration — Week 6',
    },
]

print('=' * 65)
print('  DATASET VERIFICATION REPORT — Week 1 Days 3 & 4')
print('=' * 65)

all_passed = True

for ds in datasets:
    print(f"\n▶ {ds['name']}")
    print(f"  Purpose : {ds['purpose']}")
    print(f"  File    : {ds['path']}")

    # Check file exists
    if not os.path.exists(ds['path']):
        print(f"  Status  : ✗ MISSING — file not found")
        all_passed = False
        continue

    # Check file size
    size_kb = os.path.getsize(ds['path']) / 1024
    print(f"  Size    : {size_kb:.1f} KB")

    # Try loading
    try:
        df = pd.read_csv(ds['path'])
        rows, cols = df.shape
        nulls = df.isnull().sum().sum()

        row_ok = rows >= ds['min_rows']
        status = '✓ PASS' if row_ok else '✗ FAIL (too few rows)'

        print(f"  Rows    : {rows:,}  (minimum expected: {ds['min_rows']:,})")
        print(f"  Columns : {cols}  → {list(df.columns)}")
        print(f"  Nulls   : {nulls:,} total across all cells")
        print(f"  Status  : {status}")

        if not row_ok:
            all_passed = False

    except Exception as e:
        print(f"  Status  : ✗ ERROR loading file — {e}")
        all_passed = False

print()
print('=' * 65)
if all_passed:
    print('  OVERALL: ALL DATASETS VERIFIED ✓')
    print('  Days 3 & 4 complete. Ready for Day 5 (cleaning).')
else:
    print('  OVERALL: SOME DATASETS FAILED ✗ — fix before proceeding')
print('=' * 65)