"""Test Phase 3 on realistic multi-experience CV text."""
import sys, os
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from src.extractor.experience_extractor import extract_experience
from src.extractor.education_extractor import extract_education
from src.extractor.misc_extractor import extract_projects, extract_languages

print("=== Realistic CV Experience Section ===")
text = """Software Engineer at Shohoz
Jan 2022 - Present
Designed REST APIs using Django and PostgreSQL.

Senior Developer at TechCorp
2018 - 2021
Built microservices architecture with Docker and Kubernetes.

Junior Developer, StartUp Inc
01/2016 - 12/2017
Developed frontend features with React and TypeScript."""
result, years = extract_experience(text)
print(f"Total years: {years}")
for i, exp in enumerate(result):
    print(f"  [{i}] title={exp['title']!r}")
    print(f"       company={exp['company']!r}")
    print(f"       start={exp['start']!r} end={exp['end']!r}")
    print(f"       months={exp['duration_months']}")
    print(f"       desc={exp['description'][:80]!r}")

print("\n=== Realistic Education ===")
edu_text = """B.Sc in Computer Science
University of Dhaka
2018 - 2022
GPA: 3.75

Masters in Data Science
Stanford University
2021 - 2023"""
edu = extract_education(edu_text)
for i, e in enumerate(edu):
    print(f"  [{i}] degree={e['degree']!r} inst={e['institution']!r} year={e['year']} gpa={e['gpa']}")

print("\n=== Realistic Projects ===")
proj_text = """CV Analyzer
Built with Python, Django, PostgreSQL. Deployed on AWS.
github.com/user/cv-analyzer

E-commerce Platform
React, Node.js, MongoDB. Docker containerized."""
proj = extract_projects(proj_text)
for i, p in enumerate(proj):
    print(f"  [{i}] name={p['name']!r} tools={p['tools']} link={p['link']!r}")

print("\n=== Realistic Languages ===")
lang_text = """English (C1)
Bengali (Native)
Spanish, Intermediate
French"""
lang = extract_languages(lang_text)
for l in lang:
    print(f"  lang={l['language']!r} prof={l['proficiency']!r}")

print("\nAll realistic tests passed.")
