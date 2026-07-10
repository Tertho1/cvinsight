"""Debug why Cloud Foundry is still being extracted as a name."""
import sys, os
sys.path.insert(0, os.getcwd())
import importlib
import src.extractor.contact_extractor
importlib.reload(src.extractor.contact_extractor)
from src.extractor.contact_extractor import _NAME_STOPWORDS, extract_contacts

print("cloud foundry in stopwords:", "cloud foundry" in _NAME_STOPWORDS)
print("Cloud Foundry in stopwords:", "Cloud Foundry".lower() in _NAME_STOPWORDS)

# Test with text containing "Cloud Foundry"
text = '{"name":"Unknown","email":"Unknown","summary":"Worked with Cloud Foundry for 2 years"}'
result = extract_contacts(text, contacts={})
print(f"Result with Cloud Foundry in JSON: {result}")

# Test with actual row 24 data
import pandas as pd
df = pd.read_csv("data/processed/structured_resumes_clean.csv")
row24 = df.iloc[24]
text24 = str(row24.get("text", ""))
sections = {"personal_info": str(row24.get("personal_info", ""))}
result24 = extract_contacts(text24, contacts=sections)
print(f"Row 24 result: {result24}")
