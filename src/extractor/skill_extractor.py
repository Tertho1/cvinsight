# src/extractor/skill_extractor.py
import json, spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def load_skills(taxonomy_path="config/skill_taxonomy.json") -> list:
    """
    Load skills from taxonomy JSON.
    Expected structure: {"categories": {"cat1": [skill1, skill2], "cat2": [...]}}
    Returns a flattened list of all skills across all categories.
    """
    with open(taxonomy_path) as f:
        data = json.load(f)

    # Flatten all categories into a single list
    all_skills = []
    for cat_skills in data.get("categories", {}).values():
        all_skills.extend(cat_skills)
    return all_skills


def extract_skills(text: str, taxonomy_path="config/skill_taxonomy.json") -> list:
    skills = load_skills(taxonomy_path)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s.lower()) for s in skills]
    matcher.add("SKILLS", patterns)

    doc = nlp(text.lower())
    matches = matcher(doc)
    found = list({doc[start:end].text for _, start, end in matches})
    return found