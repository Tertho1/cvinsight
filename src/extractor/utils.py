"""
Utility functions for parsing structured dataset columns.
The datasetmaster/resumes CSV stores data as Python repr strings:
  - Simple dicts: '{"key": "val"}'  -> JSON
  - Lists of dicts: "['{...}', '{...}']" -> Python list repr with JSON strings inside
"""

import ast
import json
import re


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def try_parse_structured(raw: str) -> list[dict] | dict | None:
    if not raw or raw.strip() in ("", "nan", "[]", "{}"):
        return None

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError):
        # Fallback: try direct JSON
        try:
            parsed = json.loads(raw)
            return parsed
        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, tuple):
        parsed = list(parsed)

    if isinstance(parsed, list):
        result = []
        for item in parsed:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                item_stripped = item.strip()
                try:
                    obj = json.loads(item_stripped)
                    if isinstance(obj, dict):
                        result.append(obj)
                    elif isinstance(obj, list):
                        result.extend(o for o in obj if isinstance(o, dict))
                except (json.JSONDecodeError, TypeError):
                    objs = _extract_json_objects(item_stripped)
                    if objs:
                        result.extend(objs)
                    else:
                        result.append(item_stripped)
        return result

    return None


def _extract_json_objects(text: str) -> list[dict]:
    """Extract individual JSON objects from a string that may contain multiple concatenated JSONs."""
    results = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        results.append(obj)
                except (json.JSONDecodeError, TypeError):
                    pass
                start = -1
    return results


def parse_json_field(raw: str) -> list[dict] | dict | None:
    """Parse a structured field from the dataset into Python objects."""
    return try_parse_structured(raw)


_MODEL_CACHE = {}


def load_spacy_model():
    """Load the best available spaCy model with fallback chain: trf → md → sm."""
    if "nlp" in _MODEL_CACHE:
        return _MODEL_CACHE["nlp"]
    try:
        import spacy
    except ImportError:
        raise ImportError("spacy is required")
    for model in ("en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(model)
            _MODEL_CACHE["nlp"] = nlp
            _MODEL_CACHE["name"] = model
            return nlp
        except OSError:
            continue
    raise OSError("No spaCy model found (tried en_core_web_md, en_core_web_sm)")
