# src/extractor/skill_extractor.py
import json, os, re, spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def load_skills(taxonomy_path="config/skill_taxonomy.json") -> list:
    """
    Load skills from taxonomy JSON.
    Expected structure: {"categories": {"cat1": [skill1, skill2], "cat2": [...]}}
    Returns a flattened list of all skills across all categories.
    """
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Skill taxonomy not found at {taxonomy_path}. Ensure config/skill_taxonomy.json exists.")
    with open(taxonomy_path, encoding="utf-8") as f:
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


def expand_skill_set(skills):
    """Expand raw skill entries into a set of matchable search tokens.

    Skills can be stored as chained spans (e.g. "React.js, Next.js, Vue.js" or
    "Python, Django, PostgreSQL,") that are shown as one line in the UI. A
    naive exact-match search then fails to find the individual skills even
    though they are visibly present. This returns the full entry plus every
    comma/slash/&/|/"and"/"or"-separated individual skill.

    "+" is deliberately not a separator so "C++" stays a single skill.
    """
    expanded = set()
    for s in skills or []:
        low = str(s).strip().lower().strip(".")
        if not low:
            continue
        expanded.add(low)
        for part in re.split(r"\s*(?:,|、|/|&|\||\band\b|\bor\b)\s*", low):
            part = part.strip(".").strip()
            if part:
                expanded.add(part)
    return expanded