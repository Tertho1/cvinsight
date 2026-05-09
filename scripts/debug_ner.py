# scripts/debug_ner.py
# Finds exactly which files contain surrogate characters
# so we know the scope of the problem

import json
import glob

files = glob.glob('data/raw/ner_extracted/**/*.json', recursive=True)
print(f'Scanning {len(files)} files for surrogate characters...')

problem_files = []

for path in files:
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        # Try the strictest possible decode
        text = raw.decode('utf-8', errors='surrogateescape')
        # Now try to re-encode — this is where surrogates explode
        text.encode('utf-8', errors='strict')
    except UnicodeEncodeError as e:
        problem_files.append((path, str(e)))

print(f'Files with surrogate characters: {len(problem_files)}')
print()
for path, err in problem_files[:10]:
    print(f'  {path}')
    print(f'  Error: {err}')
    print()