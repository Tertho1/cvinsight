"""
Debug data format - check if ast.literal_eval works on the CSV columns.
"""
import ast
import json
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd

df = pd.read_csv("data/processed/datasetmaster_clean.csv").head(1)

for col in ["experience", "education", "projects", "skills", "personal_info"]:
    raw = str(df.iloc[0][col])
    print(f"--- {col} (repr) ---")
    print(repr(raw[:200]))
    
    try:
        parsed = ast.literal_eval(raw)
        print(f"  ast.literal_eval OK -> type={type(parsed).__name__}")
        if isinstance(parsed, list):
            print(f"  list of {len(parsed)} items")
            for i, item in enumerate(parsed[:2]):
                if isinstance(item, str):
                    try:
                        obj = json.loads(item)
                        print(f"    [{i}] JSON decoded: keys={list(obj.keys())[:5]}")
                    except:
                        print(f"    [{i}] not JSON (str len={len(item)})")
                else:
                    print(f"    [{i}] type={type(item).__name__}")
        elif isinstance(parsed, dict):
            print(f"  dict keys={list(parsed.keys())[:8]}")
    except Exception as e:
        print(f"  ast.literal_eval FAILED: {e}")
    print()
