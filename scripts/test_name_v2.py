"""
Comprehensive name extraction test against the known false positives.
"""
import sys, os
sys.path.insert(0, os.getcwd())

# Clear any cached imports
for mod in list(sys.modules.keys()):
    if "contact_extractor" in mod:
        del sys.modules[mod]

import importlib
import src.extractor.contact_extractor
importlib.reload(src.extractor.contact_extractor)
from src.extractor.contact_extractor import (
    extract_contacts, _is_tech_term, _contains_tech_word,
    _looks_like_real_name
)

print("=== 1. Tech term detection ===")
false_positives = [
    "Spring Boot", "Spring Cloud", "Google Cloud", "Machine Learning",
    "Apache Spark", "React Native", "Computer Vision", "Deep Learning",
    "Docker Compose", "Power BI", "Visual Studio", "Data Science",
    "Full Stack", "Amazon Web", "Kubernetes Engine", "Restful API",
    "Routing Protocols", "Problem Solving", "Login Registration",
    "Data Scientist", "Secret Key", "Siemens S7-300", "Arya Kanya",
    "Problem Solving", "Restful API", "Machine Learning Engineer",
    "Critical Thinking", "Project Management",
]
all_pass = True
for term in false_positives:
    result = _is_tech_term(term)
    if not result:
        print(f"  FAIL: {term!r} NOT detected as tech term")
        all_pass = False
if all_pass:
    print(f"  All {len(false_positives)} false positives correctly rejected")

print("\n=== 2. Real names should NOT be flagged as tech ===")
real_names = [
    "John Smith", "Jane Doe", "Mohammed Ali", "Sarah Johnson",
    "Chen Wei", "Maria Garcia", "Fahed", "Artem Sliusarenko",
    "David Miller", "Priya Patel", "Ahmed Hassan", "James Thompson",
    "Emily Chen", "Aditya Rathore", "Yogesh Tikhat", "Sherri Elliott",
    "Jason Green", "Lisa Spencer", "Jack Rodgers", "Cynthia Decker",
    "Allison Powell", "Jeffrey Rasmussen", "Manuel Ramirez",
    "Jonathan Dean", "Vanessa English", "Johnny Tucker",
    "Vincent Jordan", "Nicole Smith", "Richard Ross", "Dorothy Lester",
]
all_pass = True
for name in real_names:
    result = _is_tech_term(name)
    if result:
        print(f"  FAIL: {name!r} FLAGGED as tech term")
        all_pass = False
if all_pass:
    print(f"  All {len(real_names)} real names correctly pass")

print("\n=== 3. looks_like_real_name checks ===")
for name in real_names[:10]:
    result = _looks_like_real_name(name)
    status = "PASS" if result else "FAIL"
    print(f"  {name:30s} -> {status}")

bad_names = [
    "a Secret Key", "Spring Boot", "Problem Solving", "Restful API",
    "Login Registration", "Data Scientist", "Siemens S7-300",
    "PRINCE2 Practitioner", "Cloud Foundry",
]
for name in bad_names:
    result = _looks_like_real_name(name)
    status = "REJECTED" if not result else "PASS (bad!)"
    print(f"  {name:30s} -> {status}")

print("\n=== 4. End-to-end extraction tests ===")

tests = [
    # (text, contacts_dict, expected_name_contains)
    ("Spring Boot Developer\nJohn Smith\njohn@email.com\n+1-555-1234", {}, "John Smith"),
    ("Machine Learning Engineer\njane@email.com\nJane Doe", {}, "Jane Doe"),
    ("Python Developer at Google\nWorked on Spring Boot\nDocker and Kubernetes", {}, ""),
    ("Spring Boot\nDeveloper Profile\ncontact@email.com", {}, ""),
    ("Dr. Sarah Johnson\nChief Scientist\nsarah@lab.com", {}, "Sarah Johnson"),
    ("Sherri Elliott\njlara@example.com\n123 Main St", {}, "Sherri Elliott"),
    ("Jason Green\narnoldthomas@example.com\n456 Oak Ave", {}, "Jason Green"),
    ("Emily Chen\nemily@email.com\n(408) 555-0123", {}, "Emily Chen"),
    ("James Thompson\nSoftware Architect\njames@company.com", {}, "James Thompson"),
    ('{"name":"","email":""}\nJane Wilson\nDevOps Engineer', {}, "Jane Wilson"),
    ("Cloud Foundry Developer\nWorked with Cloud Foundry platform", {}, ""),
    # "Fahed" comes from personal_info JSON, not plain text
    ("Fahed\nPython Developer\nfahed@email.com", {"personal_info": '{"name":"Fahed"}'}, "Fahed"),
    ("Adam Smith\nadam@email.com\nSkills: Python, Docker, AWS", {}, "Adam Smith"),
    ("Skills: Problem Solving, Critical Thinking, Teamwork", {}, ""),
    ("Restful API\nLogin Registration\na Secret Key\nRouting Protocols", {}, ""),
]

all_pass = True
for text, contacts, expected in tests:
    result = extract_contacts(text, contacts=contacts)
    name = result["name"]
    if expected:
        ok = expected.lower() in name.lower()
    else:
        ok = not name
    if not ok:
        print(f"  FAIL: text={text!r}")
        print(f"        expected={expected!r}, got={name!r}")
        all_pass = False

if all_pass:
    print(f"  All {len(tests)} end-to-end tests PASS")
else:
    print(f"\n  Some tests FAILED - see above")

print("\n=== DONE ===")
