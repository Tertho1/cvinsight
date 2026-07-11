"""
Debug education format.
"""
import ast
import json
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd

df = pd.read_csv("data/processed/datasetmaster_clean.csv").head(1)

raw = str(df.iloc[0]["education"])
print("Raw repr:")
print(repr(raw[:400]))

parsed = ast.literal_eval(raw)
print(f"\nParsed type={type(parsed).__name__}, len={len(parsed)}")

for i, item in enumerate(parsed):
    print(f"\n[{i}] type={type(item).__name__}")
    if isinstance(item, str):
        print(f"  str len={len(item)}")
        print(f"  starts with: {item[:100]}")
        try:
            obj = json.loads(item)
            print(f"  JSON OK: keys={list(obj.keys())}")
        except json.JSONDecodeError as e:
            print(f"  JSON fail: {e}")
            # Try cleaning: the string might have escaped quotes inside
            clean = item.replace("\\'", "'")
            try:
                obj = json.loads(clean)
                print(f"  JSON after clean OK: keys={list(obj.keys())}")
            except:
                pass
