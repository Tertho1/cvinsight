# scripts/load_netsol.py
# Downloads netsol dataset by fetching raw JSON files directly
# instead of using load_dataset() which crashes on schema conflicts

import os
import json
import glob
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

os.makedirs("data/raw/netsol_raw_files", exist_ok=True)

print("Listing all files in netsol/resume-score-details repo...")

# Get list of all files in the repo
all_files = list(list_repo_files("netsol/resume-score-details", repo_type="dataset"))

print(f"Total files in repo: {len(all_files)}")

# Filter for JSON data files only
json_files = [f for f in all_files if f.endswith(".json") and "data/" in f]
print(f"JSON data files found: {len(json_files)}")

if not json_files:
    # Try without the data/ filter
    json_files = [f for f in all_files if f.endswith(".json")]
    print(f"All JSON files: {len(json_files)}")

print("First 5 files:", json_files[:5])
print()

# Download each file individually
downloaded = 0
errors = 0

for file_path in json_files:
    try:
        local_path = hf_hub_download(
            repo_id="netsol/resume-score-details",
            filename=file_path,
            repo_type="dataset",
            local_dir="data/raw/netsol_raw_files",
        )
        downloaded += 1
        if downloaded % 50 == 0:
            print(f"  Downloaded {downloaded}/{len(json_files)} files...")
    except Exception as e:
        errors += 1
        print(f"  Error on {file_path}: {e}")

print(f"Downloaded: {downloaded}, Errors: {errors}")
