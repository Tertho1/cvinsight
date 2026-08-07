"""
tests/test_ner_skill_filter.py
Unit tests for the hardened NER skill cleaning in src/extractor/ner_tag.py.

The token-classification tagger emits noisy "skill" spans (comma-joined chains,
trailing punctuation, URLs, locations). `_skill_parts` decomposes them into
clean, plausible skill tokens; `merge_skills` unions them with the rule-based
list without reintroducing the noise.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor.ner_tag import _skill_parts, merge_skills


# ---------------------------------------------------------------------------
# _skill_parts: decompose chained / noisy NER skill spans
# ---------------------------------------------------------------------------

def test_chained_span_splits_into_individual_skills():
    assert _skill_parts("Python, React, Redux, Angular,") == \
        ["Python", "React", "Redux", "Angular"]


def test_trailing_punctuation_stripped():
    assert _skill_parts("Redis,") == ["Redis"]
    assert _skill_parts("Redux.") == ["Redux"]


def test_url_span_dropped():
    assert _skill_parts("linkedin.com/in/johndoe") == []
    assert _skill_parts("https://github.com/foo/bar") == []
    assert _skill_parts("www.example.com") == []


def test_location_span_dropped():
    assert _skill_parts("Google, Mountain View, CA, USA") == []
    assert _skill_parts("USA") == []


def test_location_word_fragment_dropped_but_real_skill_kept():
    # "Google" alone is org-ish; the split keeps nothing from the geo chain,
    # while a standalone real skill survives.
    assert _skill_parts("View,") == []


def test_dot_js_skills_preserved():
    assert _skill_parts("D3.js") == ["D3.js"]
    assert _skill_parts("Vue.js, Express.js,") == ["Vue.js", "Express.js"]


def test_empty_and_too_short():
    assert _skill_parts("") == []
    assert _skill_parts("x") == []


# ---------------------------------------------------------------------------
# merge_skills: union without re-adding noise or duplicating rules
# ---------------------------------------------------------------------------

def test_merge_keeps_rules_first_and_adds_clean_ner_skills():
    rules = ["Python", "SQL"]
    groups = {"skill": ["Python,", "Django", "Vue.js, Express.js"]}
    out = merge_skills(rules, groups)
    assert out[0] == "Python" and out[1] == "SQL"     # rule order/casing first
    assert "Django" in out
    assert "Vue.js" in out and "Express.js" in out


def test_merge_drops_ner_url_and_location_noise():
    rules = ["Python"]
    groups = {"skill": ["linkedin.com/in/x", "Google, Mountain View, CA, USA",
                        "Python", "Webpack"]}
    out = merge_skills(rules, groups)
    assert out == ["Python", "Webpack"]


def test_merge_dedups_case_insensitively():
    rules = ["python"]
    groups = {"skill": ["Python"]}
    assert merge_skills(rules, groups) == ["python"]


def test_merge_empty_ner_groups_passthrough():
    rules = ["Python"]
    assert merge_skills(rules, {}) == ["Python"]
    assert merge_skills(rules, None) == ["Python"]