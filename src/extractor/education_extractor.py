# src/extractor/education_extractor.py
import re, spacy

nlp = spacy.load("en_core_web_sm")

DEGREE_KEYWORDS = [
    "phd",
    "ph.d",
    "bachelor",
    "b.sc",
    "b.s.",
    "master",
    "m.sc",
    "m.s.",
    "mba",
    "diploma",
    "associate",
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
]


def extract_education(section_text: str) -> list:
    results = []
    lines = section_text.split("\n")
    doc = nlp(section_text)

    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    years = re.findall(r"\b(19|20)\d{2}\b", section_text)
    gpa = re.findall(r"(?:gpa|cgpa)[:\s]*([\d\.]+)", section_text, re.IGNORECASE)

    degree_found = ""
    for line in lines:
        for kw in DEGREE_KEYWORDS:
            if kw in line.lower():
                degree_found = line.strip()
                break

    results.append(
        {
            "degree": degree_found,
            "institution": orgs[0] if orgs else "",
            "year": int(years[-1]) if years else None,
            "gpa": float(gpa[0]) if gpa else None,
            "field": "",  # can be enriched later
        }
    )
    return results
