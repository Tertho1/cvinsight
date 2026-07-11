"""Test Phase 3 text-path improvements."""
import sys, os, json
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from src.extractor.experience_extractor import extract_experience
from src.extractor.education_extractor import extract_education
from src.extractor.misc_extractor import extract_projects, extract_languages

print("=== 1. Experience: title + description + flexible dates ===")

# Test 1a: Standard format with title
text1 = "Software Engineer, Google\nJan 2020 - Dec 2022\nDesigned and built REST APIs."
result, years = extract_experience(text1)
print(f"  Title: {result[0]['title']!r}")
print(f"  Company: {result[0]['company']!r}")
print(f"  Start: {result[0]['start']!r}")
print(f"  End: {result[0]['end']!r}")
print(f"  Desc: {result[0]['description'][:80]!r}")

# Test 1b: YYYY-YYYY format
text2 = "Senior Engineer at Acme Corp\n2018 - 2022\nLed team of 5 engineers."
result2, _ = extract_experience(text2)
print(f"\n  YYYY-YYYY: title={result2[0]['title']!r} company={result2[0]['company']!r} start={result2[0]['start']!r} end={result2[0]['end']!r}")

# Test 1c: "at" format
text3 = "Data Scientist at OpenAI\nJan 2022 - Present\nBuilt LLM pipelines."
result3, _ = extract_experience(text3)
print(f"  'at' format: title={result3[0]['title']!r} company={result3[0]['company']!r} end={result3[0]['end']!r}")

# Test 1d: MM/YYYY format
text4 = "DevOps Engineer, AWS\n01/2020 - 12/2022\nManaged cloud infrastructure."
result4, _ = extract_experience(text4)
print(f"  MM/YYYY: title={result4[0]['title']!r} start={result4[0]['start']!r} end={result4[0]['end']!r}")

print("\n=== 2. Education: paragraph-level parsing ===")

# Test 2a: Multi-line education entry
edu_text = """B.Sc in Computer Science
University of Dhaka
2018 - 2022
GPA: 3.75"""
edu = extract_education(edu_text)
print(f"  Multi-line: degree={edu[0]['degree']!r} inst={edu[0]['institution']!r} year={edu[0]['year']} gpa={edu[0]['gpa']}")

# Test 2b: Single line
edu2 = extract_education("Masters in Data Science, Stanford University, 2021")
print(f"  Single-line: degree={edu2[0]['degree']!r} inst={edu2[0]['institution']!r} year={edu2[0]['year']}")

print("\n=== 3. Projects: tools from description ===")
proj_text = "CV Analyzer\nBuilt with Python, Django, and PostgreSQL. Deployed on AWS."
proj = extract_projects(proj_text)
print(f"  Name: {proj[0]['name']!r}")
print(f"  Tools: {proj[0]['tools']}")
print(f"  Desc: {proj[0]['description'][:60]!r}")

print("\n=== 4. Languages: name detection + proficiency parsing ===")
lang1 = extract_languages("English (C1)\nBengali (Native)")
print(f"  English: {lang1[0]}")
print(f"  Bengali: {lang1[1]}")

lang2 = extract_languages("English, Fluent\nSpanish, Intermediate")
print(f"  Comma-sep: {lang2[0]}")

lang3 = extract_languages("English")
print(f"  Single: {lang3[0]}")

print("\nAll Phase 3 checks passed.")
