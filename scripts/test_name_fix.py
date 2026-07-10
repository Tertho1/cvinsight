"""
Test that the name extractor correctly rejects tech terms like "Spring Boot",
"Google Cloud", etc. while keeping real human names.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import importlib
import src.extractor.contact_extractor
importlib.reload(src.extractor.contact_extractor)
from src.extractor.contact_extractor import extract_contacts, _is_tech_term

print("=== Tech term detection ===")
for term in [
    "Spring Boot", "Spring Cloud", "Google Cloud", "Machine Learning",
    "Apache Spark", "React Native", "Computer Vision", "Deep Learning",
    "Docker Compose", "Power BI", "Visual Studio", "Data Science",
    "Full Stack", "Amazon Web", "Kubernetes Engine",
]:
    print(f"  {term:25s} -> tech={_is_tech_term(term)}")

print("\n=== Real names should NOT be flagged as tech ===")
for name in [
    "John Smith", "Jane Doe", "Mohammed Ali", "Sarah Johnson",
    "Chen Wei", "Maria Garcia", "Fahed", "Artem Sliusarenko",
    "David Miller", "Priya Patel", "Ahmed Hassan",
]:
    print(f"  {name:25s} -> tech={_is_tech_term(name)}")

print("\n=== Resume text extraction tests ===")

# Test 1: Real name preceded by tech terms
text1 = "Spring Boot Developer\nJohn Smith\njohn@email.com"
r1 = extract_contacts(text1, contacts={})
print(f"  Input: {text1!r}")
print(f"  Result: name={r1['name']!r}")
print(f"  Expected: John Smith - {'PASS' if 'John' in r1['name'] else 'FAIL'}")

# Test 2: Only tech terms, no real name
text2 = "Python Developer at Google\nWorked on Spring Boot projects\nDocker and Kubernetes"
r2 = extract_contacts(text2, contacts={})
print(f"  Input: {text2!r}")
print(f"  Result: name={r2['name']!r}")
print(f"  Expected: '' - {'PASS' if not r2['name'] else 'FAIL'}")

# Test 3: "Spring Boot" as the only two-word capitalized phrase
text3 = "Spring Boot\nDeveloper Profile\ncontact@email.com"
r3 = extract_contacts(text3, contacts={})
print(f"  Input: {text3!r}")
print(f"  Result: name={r3['name']!r}")
print(f"  Expected: '' - {'PASS' if not r3['name'] else 'FAIL'}")

# Test 4: Real name with salutation
text4 = "Dr. Sarah Johnson\nChief Scientist\nsarah@lab.com\n+1-555-1234"
r4 = extract_contacts(text4, contacts={})
print(f"  Input: {text4!r}")
print(f"  Result: name={r4['name']!r}")
print(f"  Expected: Sarah Johnson - {'PASS' if 'Sarah' in r4['name'] else 'FAIL'}")

# Test 5: Real name in raw text (no personal_info)
text5 = "Emily Chen\nemily.chen@email.com\n(408) 555-0123\nMachine Learning Engineer"
r5 = extract_contacts(text5, contacts={})
print(f"  Input: {text5!r}")
print(f"  Result: name={r5['name']!r}")
print(f"  Expected: Emily Chen - {'PASS' if 'Emily' in r5['name'] else 'FAIL'}")

# Test 6: Tech term followed by real name on same line pattern
text6 = '{"name":"","email":""}\nJane Wilson\nDevOps Engineer\nDocker Kubernetes AWS'
r6 = extract_contacts(text6, contacts={})
print(f"  Input: {text6!r}")
print(f"  Result: name={r6['name']!r}")
print(f"  Expected: Jane Wilson - {'PASS' if 'Jane' in r6['name'] else 'FAIL'}")

# Test 7: Common false positive - Cloud Foundry
text7 = "Cloud Foundry Developer\nWorked with Cloud Foundry platform"
r7 = extract_contacts(text7, contacts={})
print(f"  Input: {text7!r}")
print(f"  Result: name={r7['name']!r}")
print(f"  Expected: '' - {'PASS' if not r7['name'] else 'FAIL'}")

# Test 8: Name with a common first name (strong positive signal)
text8 = "James Thompson\nSoftware Architect\njames@company.com"
r8 = extract_contacts(text8, contacts={})
print(f"  Input: {text8!r}")
print(f"  Result: name={r8['name']!r}")
print(f"  Expected: James Thompson - {'PASS' if 'James' in r8['name'] else 'FAIL'}")

print("\n=== DONE ===")
