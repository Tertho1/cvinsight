# src/extractor/experience_extractor.py
import re, spacy
from dateparser import parse as dparse
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

DATE_PATTERN = re.compile(
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[\s,]*(\d{4})'
    r'\s*[-–to]+\s*'
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present)?[\s,]*(\d{4})?',
    re.IGNORECASE
)

def compute_months(start_str, end_str) -> int:
    try:
        start = dparse(start_str)
        end = dparse(end_str) if end_str and "present" not in end_str.lower() else datetime.now()
        if start and end:
            return max(0, (end.year - start.year) * 12 + (end.month - start.month))
    except:
        pass
    return 0

def extract_experience(section_text: str) -> tuple[list, float]:
    doc = nlp(section_text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    
    experiences = []
    date_matches = DATE_PATTERN.findall(section_text)
    
    for i, match in enumerate(date_matches):
        start_str = f"{match[0]} {match[1]}".strip()
        end_str = f"{match[2]} {match[3]}".strip() if match[2] or match[3] else "Present"
        months = compute_months(start_str, end_str)
        experiences.append({
            "title": "",           # enrich with job-title NER or keyword rules
            "company": orgs[i] if i < len(orgs) else "",
            "start": match[1],
            "end": match[3] or "Present",
            "duration_months": months,
            "description": ""
        })
    
    total_years = sum(e["duration_months"] for e in experiences) / 12
    return experiences, round(total_years, 1)