"""
tests/test_hybrid.py
Tests for src/extractor/hybrid.py -- the optional Rule + LLM extraction backend.

We do NOT load the 2GB Qwen LoRA in tests. Instead we exercise the pieces that
must work without a model:
  * extract_with_llm() degrades gracefully (empty text, model-load / inference
    failure) and never blocks the rule-based path,
  * fuse() merges a rule-based CVSchema dict with an LLM dict per the documented
    per-field policy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor import hybrid


# ---------------------------------------------------------------------------
# extract_with_llm: graceful degradation without a model
# ---------------------------------------------------------------------------

def test_extract_with_llm_empty_text_returns_empty():
    assert hybrid.extract_with_llm("") == {}
    assert hybrid.extract_with_llm("   ") == {}


def test_extract_with_llm_no_model_and_failing_load_returns_empty():
    # Passing a sentinel model/tokenizer that raises on use must yield {} (the
    # caller then falls back to rule-based), not propagate an exception.
    class _Broken:
        def encode(self, *a, **k):
            raise RuntimeError("model died")

    out = hybrid.extract_with_llm(
        "some resume text", model=_Broken(), tokenizer=_Broken())
    assert out == {}


# ---------------------------------------------------------------------------
# fuse: per-field policy
# ---------------------------------------------------------------------------

def test_fuse_prefers_llm_experience_when_more_dated():
    rules = {
        "name": "A", "email": "", "phone": "", "skills": ["python"],
        "experience": [{"title": "Dev", "company": "Co", "start": "2020"}],
        "education": [], "projects": [], "certifications": [], "languages": [],
        "achievements": ["pub"], "leadership": [],
    }
    llm = {
        "name": "A", "email": "a@b.c", "phone": "123", "skills": ["python", "SQL"],
        "experience": [
            {"title": "Dev", "company": "Co", "start": "2020"},
            {"title": "Senior Dev", "company": "Co2", "start": "2018"},
        ],
        "education": [], "projects": [], "certifications": [], "languages": [],
        "achievements": [], "leadership": [],
    }
    out = hybrid.fuse(rules, llm)
    assert len(out["experience"]) == 2          # LLM has more dated entries
    assert "SQL" in out["skills"]
    assert "pub" in out["achievements"]          # rule-only field preserved


def test_fuse_prefers_rules_experience_when_more_dated():
    rules = {
        "name": "A", "email": "", "phone": "", "skills": ["python"],
        "experience": [
            {"title": "D1", "company": "C1", "start": "2019"},
            {"title": "D2", "company": "C2", "start": "2017"},
        ],
        "education": [], "projects": [], "certifications": [], "languages": [],
        "achievements": [], "leadership": [],
    }
    llm = {
        "name": "A", "email": "a@b.c", "phone": "123", "skills": ["python"],
        "experience": [{"title": "D1", "company": "C1", "start": "2019"}],
        "education": [], "projects": [], "certifications": [], "languages": [],
        "achievements": [], "leadership": [],
    }
    out = hybrid.fuse(rules, llm)
    assert len(out["experience"]) == 2           # rules win on count


def test_fuse_dedups_shared_experience():
    rules = {
        "name": "A", "email": "", "phone": "", "skills": [],
        "experience": [{"title": "Dev", "company": "Co", "start": "2020"}],
        "education": [], "projects": [], "certifications": [], "languages": [],
        "achievements": [], "leadership": [],
    }
    llm = dict(rules)
    out = hybrid.fuse(rules, llm)
    assert len(out["experience"]) == 1           # same entry not doubled


def test_build_cv_grounds_skills_outside_text():
    text = "I know Python and SQL."
    raw = {"name": "A", "email": "", "phone": "", "skills": ["python", "go"]}
    cv = hybrid.build_cv(raw, text)
    assert "python" in cv["skills"]
    assert "go" not in cv["skills"]              # invented skill dropped