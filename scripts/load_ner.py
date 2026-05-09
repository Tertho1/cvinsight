# scripts/load_ner.py

import json
import glob
import pandas as pd


def deep_clean(obj):
    """
    Recursively walks any Python object (dict, list, str)
    and removes surrogate unicode characters from every string.
    
    Why recursive? Because json.loads() can reconstruct surrogates
    from JSON escape sequences like \\uD83D AFTER our byte-level
    cleaning. The only guaranteed fix is to clean the parsed object
    itself, walking into every nested string.
    """
    if isinstance(obj, str):
        # Encode to bytes dropping surrogates, decode back to clean str
        return obj.encode('utf-8', errors='ignore').decode('utf-8')
    elif isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean(item) for item in obj]
    else:
        return obj  # int, float, bool, None — leave untouched


files = glob.glob('data/raw/ner_extracted/**/*.json', recursive=True)
print(f'Found {len(files)} JSON files')

records = []
errors = 0

for i, path in enumerate(files):
    try:
        # Step 1: read raw bytes
        with open(path, 'rb') as f:
            raw = f.read()

        # Step 2: decode bytes — replace bad byte sequences with ?
        text_raw = raw.decode('utf-8', errors='replace')

        # Step 3: parse JSON
        data = json.loads(text_raw)

        # Step 4: deep clean the entire parsed object
        data = deep_clean(data)

        # Step 5: extract fields
        text        = data.get('text', '')
        annotations = str(data.get('annotations', []))

        records.append({
            'file':        path,
            'text':        text,
            'annotations': annotations,
            'text_length': len(text)
        })

    except Exception as e:
        errors += 1
        print(f'  Skipped [{i}] {path} — {e}')

print(f'Building DataFrame from {len(records)} records...')

# Build column by column to avoid pandas arrow encoding issues
df = pd.DataFrame({
    'file':        [r['file']        for r in records],
    'text':        [r['text']        for r in records],
    'annotations': [r['annotations'] for r in records],
    'text_length': [r['text_length'] for r in records],
})

print(f'Saving to CSV...')
df.to_csv('data/raw/ner_resumes_raw.csv', index=False, encoding='utf-8')

print(f'Successfully loaded : {len(df)} CVs')
print(f'Errors skipped      : {errors}')
print(f'Avg text length     : {df["text_length"].mean():.0f} chars')
print(f'Min text length     : {df["text_length"].min()}')
print(f'Max text length     : {df["text_length"].max()}')
print()
print('Sample CV text (first 300 chars):')
print(df['text'].iloc[0][:300])
print()
print('Saved to data/raw/ner_resumes_raw.csv')