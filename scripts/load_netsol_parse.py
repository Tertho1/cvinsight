# scripts/load_netsol_parse.py

import json
import glob
import os
import pandas as pd


def safe_str(val) -> str:
    if val is None:
        return ''
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val).encode('utf-8', errors='ignore').decode('utf-8')


def extract_score(output: dict) -> float | None:
    """
    Pull the final numeric score from the output block.
    The netsol dataset stores scores inside output -> scores -> aggregated_scores
    or directly as output -> final_score. We try both.
    """
    if not isinstance(output, dict):
        return None
    # Try direct final_score first
    if 'final_score' in output:
        try:
            return float(output['final_score'])
        except (TypeError, ValueError):
            pass
    # Try aggregated_scores
    agg = output.get('scores', {}).get('aggregated_scores', {})
    if isinstance(agg, dict) and agg:
        # Take the first numeric value found
        for v in agg.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def classify_file(filename: str) -> str:
    base = os.path.basename(filename)
    if base.startswith('match_'):           return 'match'
    elif base.startswith('mismatch_'):      return 'mismatch'
    elif base.startswith('empty_'):         return 'empty'
    elif base.startswith('invalid_'):       return 'invalid'
    return 'unknown'


files = glob.glob('data/raw/netsol_raw_files/**/*.json', recursive=True)
print(f'Total JSON files: {len(files)}')

records = []
errors = 0

for path in files:
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        clean = raw.decode('utf-8', errors='surrogateescape')
        clean = clean.encode('utf-8', errors='ignore').decode('utf-8')
        data = json.loads(clean)

        # ── Extract the three blocks ──────────────────────────
        inp     = data.get('input',   {}) or {}
        output  = data.get('output',  {}) or {}
        details = data.get('details', {}) or {}

        # ── Job description (inside input) ────────────────────
        jd_text = safe_str(inp.get('job_description', ''))

        # ── CV data (inside details) ──────────────────────────
        candidate_name  = safe_str(details.get('name', ''))
        skills          = safe_str(details.get('skills', []))
        education       = safe_str(details.get('education', []))
        experience      = safe_str(details.get('experience', []))
        projects        = safe_str(details.get('projects', []))
        achievements    = safe_str(details.get('achievements', []))

        # ── Score and justification (inside output) ───────────
        score           = extract_score(output)
        justification   = safe_str(output.get('justification', ''))

        records.append({
            'file_type':        classify_file(path),
            'filename':         os.path.basename(path),
            'candidate_name':   candidate_name,
            'job_description':  jd_text,
            'skills':           skills,
            'education':        education,
            'experience':       experience,
            'projects':         projects,
            'achievements':     achievements,
            'score':            score,
            'justification':    justification,
        })

    except Exception as e:
        errors += 1
        print(f'  Error on {os.path.basename(path)}: {e}')

df = pd.DataFrame(records)

print(f'Total records : {len(df)}')
print(f'Errors        : {errors}')
print()
print('File type distribution:')
print(df['file_type'].value_counts())
print()
print('Score — non-null :', df['score'].notna().sum())
print('JD    — non-empty:', (df['job_description'] != '').sum())
print('Skills— non-empty:', (df['skills'] != '[]').sum())
print()

# Show one clean match record
match_rows = df[df['file_type'] == 'match']
if len(match_rows):
    s = match_rows.iloc[0]
    print('=== Sample match record ===')
    print(f'Candidate : {s["candidate_name"]}')
    print(f'Score     : {s["score"]}')
    print(f'Skills    : {s["skills"][:150]}')
    print(f'JD snippet: {s["job_description"][:200]}')
    print(f'Reasoning : {s["justification"][:200]}')

df.to_csv('data/raw/netsol_raw.csv', index=False, encoding='utf-8')
print()
print(f'Saved {len(df)} records → data/raw/netsol_raw.csv')