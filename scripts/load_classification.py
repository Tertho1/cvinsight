# scripts/load_classification.py
#
# The noran-mohamed dataset uses Git LFS (Large File Storage).
# Downloading the repo zip gives a pointer file, not real data.
# We must fetch the raw file directly via the GitHub raw URL
# or use the Kaggle version of the same dataset as a fallback.
#
# Strategy:
#   1. Try direct GitHub raw download via LFS URL
#   2. If that fails, use the Kaggle resume dataset (same content)

import os
import urllib.request
import pandas as pd

os.makedirs('data/raw/classification_resumes', exist_ok=True)

# ── Attempt 1: direct GitHub raw download ─────────────────────
# GitHub LFS serves real files through media.githubusercontent.com
urls_to_try = [
    'https://media.githubusercontent.com/media/noran-mohamed/Resume-Classification-Dataset/main/Resume.csv',
    'https://raw.githubusercontent.com/noran-mohamed/Resume-Classification-Dataset/main/Resume.csv',
]

dest = 'data/raw/classification_resumes/Resume.csv'
downloaded = False

for url in urls_to_try:
    try:
        print(f'Trying: {url}')
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f'Downloaded: {size_kb:.1f} KB')

        # Verify it's real data, not another pointer
        if size_kb > 100:
            df = pd.read_csv(dest)
            print(f'Rows: {len(df)}, Columns: {list(df.columns)}')
            if len(df) > 100:
                downloaded = True
                print('SUCCESS — real data confirmed')
                break
            else:
                print('Too few rows — still a pointer file, trying next URL')
        else:
            print('File too small — still a pointer, trying next URL')
    except Exception as e:
        print(f'Failed: {e}')

# ── Attempt 2: Kaggle API fallback ────────────────────────────
if not downloaded:
    print()
    print('GitHub URLs failed. Trying Kaggle...')
    try:
        import subprocess
        result = subprocess.run(
            ['kaggle', 'datasets', 'download',
             '-d', 'gauravduttakiit/resume-dataset',
             '-p', 'data/raw/classification_resumes',
             '--unzip'],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            # Find the downloaded CSV
            import glob
            csvs = glob.glob('data/raw/classification_resumes/**/*.csv',
                             recursive=True)
            print(f'CSV files found: {csvs}')
            if csvs:
                df = pd.read_csv(csvs[0])
                print(f'Rows: {len(df)}, Columns: {list(df.columns)}')
                # Standardize to our expected format
                df.to_csv(dest, index=False)
                downloaded = True
                print('SUCCESS via Kaggle')
        else:
            print('Kaggle error:', result.stderr)
    except FileNotFoundError:
        print('Kaggle CLI not installed.')

# ── Attempt 3: HuggingFace fallback ───────────────────────────
if not downloaded:
    print()
    print('Trying HuggingFace fallback dataset...')
    try:
        from datasets import load_dataset
        # This dataset on HuggingFace contains the same resume classification data
        ds = load_dataset('ahmedheakl/resume-atlas', trust_remote_code=True)
        print('Splits:', ds)
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f'Rows: {len(df)}, Columns: {list(df.columns)}')
        df.to_csv(dest, index=False)
        downloaded = True
        print('SUCCESS via HuggingFace ResumeAtlas')
    except Exception as e:
        print(f'HuggingFace also failed: {e}')

# ── Final status ───────────────────────────────────────────────
if downloaded:
    df = pd.read_csv(dest)
    df.to_csv('data/raw/classification_raw.csv', index=False)
    print()
    print(f'Final file: data/raw/classification_raw.csv')
    print(f'Rows      : {len(df):,}')
    print(f'Columns   : {list(df.columns)}')
    print()
    print('Sample categories:')
    # Find the category column
    cat_col = None
    for name in ['Category', 'category', 'label', 'Label', 'job_category']:
        if name in df.columns:
            cat_col = name
            break
    if cat_col:
        print(df[cat_col].value_counts().head(10))
    else:
        print('(category column not identified — check columns above)')
else:
    print()
    print('ALL ATTEMPTS FAILED.')
    print('Manual steps to follow — see instructions below.')