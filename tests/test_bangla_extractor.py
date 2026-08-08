"""Tests for the Bengali transliteration extraction path
(src/extractor/bangla_extractor.py).

Covers script detection, Bengali digit/term/date transliteration, heading
splitting, full extraction routed through extract_all(), and the additive
Bangla NER fusion (exercised with a fake NER so no model is loaded in tests).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("CV_BANGLA_NER", "0")

from src.extractor.bangla_extractor import (  # noqa: E402
    is_bangla,
    _translate_numerals_and_terms,
    _transliterate_headings,
    split_bangla_sections,
    extract_bangla,
    _bangla_name,
    _extract_bangla_languages,
    _fuse_bangla_ner,
)

BN_CV = """A Software Engineer
rahul.sharma@email.com
+8801712345678

প্রফেশনাল সামারি
Python developer, ৫ বছরের অভিজ্ঞতা।

কর্ম অভিজ্ঞতা
সফটওয়্যার ইঞ্জিনিয়ার, ব্রাইটপাথ ডিজাইন
জানুয়ারি ২০২০ - বর্তমান
Python, Django, React

শিক্ষাগত যোগ্যতা
বিএসসি ইন কম্পিউটার সায়েন্স, ঢাকা বিশ্ববিদ্যালয়, ২০১৯

দক্ষতা
Python, JavaScript, Django, React, AWS, Docker

ভাষা
বাংলা (নেটিভ), ইংরেজি (ফ্লুয়েন্ট)
"""


class TestIsBangla:

    def test_bangla_text_detected(self):
        assert is_bangla(BN_CV) is True

    def test_english_text_not_detected(self):
        eng = ("This is a resume for John Doe, software engineer with 10 years "
               "experience in Python and Django. University of Toronto, 2020.")
        assert is_bangla(eng) is False

    def test_empty_text_false(self):
        assert is_bangla("") is False
        assert is_bangla(None) is False

    def test_low_bangla_proportion_false(self):
        assert is_bangla("English text with এক loner bengali word") is False


class TestTransliteration:

    def test_digits_converted(self):
        assert "2020" in _translate_numerals_and_terms("২০২০")
        assert "2019" in _translate_numerals_and_terms("২০১৯")

    def test_months_converted(self):
        out = _translate_numerals_and_terms("জানুয়ারি ২০২০ - বর্তমান")
        assert "January" in out
        assert "2020" in out
        assert "present" in out.lower()

    def test_degree_words_converted(self):
        assert "B.Sc" in _translate_numerals_and_terms("বিএসসি")
        assert "Master" in _translate_numerals_and_terms("স্নাতকোত্তর")

    def test_dotted_degree_words_converted(self):
        assert "B.Sc" in _translate_numerals_and_terms("বি.এসসি ইন কম্পিউটার সায়েন্স")
        assert "HSC" in _translate_numerals_and_terms("এইচএসসি — বিজ্ঞান")
        assert "Associate" in _translate_numerals_and_terms("অ্যাসোসিয়েট সার্টিফিকেট")

    def test_languages_converted(self):
        assert "Bengali" in _translate_numerals_and_terms("বাংলা")
        assert "English" in _translate_numerals_and_terms("ইংরেজি")

    def test_skill_heading_not_corrupted(self):
        # "দক্ষতা" (skills heading) must survive, not become "Proficiencyতা".
        assert "Proficiencyতা" not in _translate_numerals_and_terms("দক্ষতা")

    def test_proficiency_parenthesized(self):
        assert "(Native)" in _translate_numerals_and_terms("বাংলা (নেটিভ)")

    def test_job_titles_converted(self):
        out = _translate_numerals_and_terms("সিনিয়র সফটওয়্যার ইঞ্জিনিয়ার")
        assert "Senior Software Engineer" in out

    def test_skill_terms_converted(self):
        out = _translate_numerals_and_terms(
            "পাইথন, জাভাস্ক্রিপ্ট, রিয়্যাক্ট, মাইএসকিউএল, ডকার, গিট")
        for en in ("Python", "JavaScript", "React", "MySQL", "Docker", "Git"):
            assert en in out

    def test_short_skill_key_does_not_break_words(self):
        # "গো" (Go) must not match inside "যোগোদান" etc.
        out = _translate_numerals_and_terms("যোগোদান এবং গো প্রোগ্রামিং")
        assert "Go" in out
        assert "যোগোদান" in out

    def test_company_terms_converted(self):
        out = _translate_numerals_and_terms("টেক সলিউশনস লিমিটেড")
        assert "Limited" in out
        assert "Solutions" in out

    def test_date_connector_converted(self):
        out = _translate_numerals_and_terms("জানুয়ারি 2020 থেকে ডিসেম্বর 2023")
        assert "January 2020 to December 2023" in out

    def test_present_after_connector(self):
        out = _translate_numerals_and_terms("জানুয়ারি 2020 থেকে বর্তমান")
        assert "present" in out.lower()


class TestSplitting:

    def test_headings_transliterated(self):
        eng = _transliterate_headings(_translate_numerals_and_terms(BN_CV))
        assert "WORK EXPERIENCE" in eng
        assert "EDUCATION" in eng
        assert "TECHNICAL SKILLS" in eng

    def test_real_world_headings_transliterated(self):
        eng = _transliterate_headings(
            _translate_numerals_and_terms(
                "কর্মসংস্থান ও অভিজ্ঞতা\nকারিগরি দক্ষতা\nপ্রধান দক্ষতা\n"
                "সার্টিফিকেশন ও প্রজেক্ট\nভাষাগত দক্ষতা\nপেশাগত সারাংশ"
            )
        )
        assert "WORK EXPERIENCE" in eng
        assert "TECHNICAL SKILLS" in eng
        assert "CERTIFICATIONS" in eng
        assert "LANGUAGES" in eng
        assert "PROFESSIONAL SUMMARY" in eng

    def test_institution_and_phone_transliterated(self):
        out = _translate_numerals_and_terms(
            "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়\nফোন: +৮৮০ ১৭১২-৩৪৫৬৭৮")
        assert "Engineering University" in out
        assert "8801712345678" in out

    def test_sections_split(self):
        eng = _transliterate_headings(_translate_numerals_and_terms(BN_CV))
        sections = split_bangla_sections(eng)
        assert "experience" in sections
        assert "education" in sections
        assert "skills" in sections
        assert "languages" in sections


class TestExtractBangla:

    def test_full_extract(self):
        cv = extract_bangla(BN_CV)
        assert cv["language"] == "bangla"
        assert cv["experience"], cv["experience"]
        assert cv["experience"][0]["duration_months"] > 0
        assert any("B.Sc" in str(e.get("degree", "")) for e in cv["education"])
        langs = [l["language"] for l in cv["languages"]]
        assert "Bengali" in langs and "English" in langs

    def test_name_fallback(self):
        # Transliterated heading must not become the name.
        assert _bangla_name("WORK EXPERIENCE\nইংরেজি\nরাহুল শর্মা\nকিছু নয়") == "রাহুল শর্মা"

    def test_even_text_is_path(self):
        assert _bangla_name("Rahul Sharma") in ("Rahul Sharma", "")

    def test_multiple_languages_single_line(self):
        langs = _extract_bangla_languages("Bengali (Native), English (Fluent)")
        assert [l["language"] for l in langs] == ["Bengali", "English"]

    def test_extract_via_extract_all(self):
        from src.extractor.extractor import extract_all
        from src.parser.section_splitter import split_sections
        sections = split_sections(BN_CV)
        cv = extract_all(BN_CV, sections=sections)
        assert cv["language"] == "bangla"
        assert cv["skills"]
        assert cv["experience"]


class FakeBanglaNer:
    """Stand-in for the real models/bangla-ner-v1 loader; fixed span dict."""

    loaded = True

    def __init__(self, spans):
        self._spans = spans

    def predict_spans(self, text):
        return self._spans


class TestBanglaNerFusion:

    def test_ner_disabled_by_env(self):
        # CV_BANGLA_NER is "0" in this suite's import, so fusion must no-op.
        cv = {"skills": [], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট")
        assert out == cv

    def test_ner_skill_mapping(self, monkeypatch):
        monkeypatch.delenv("CV_BANGLA_NER")
        spans = {"skill": ["পাইথন"], "degree": []}
        cv = {"skills": [], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট", ner=FakeBanglaNer(spans))
        assert "Python" in out["skills"]

    def test_ner_skill_dedup(self, monkeypatch):
        monkeypatch.delenv("CV_BANGLA_NER")
        spans = {"skill": ["পাইথন"], "degree": []}
        cv = {"skills": ["python"], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট", ner=FakeBanglaNer(spans))
        assert out["skills"] == ["python"]

    def test_ner_education_gap_fill(self, monkeypatch):
        monkeypatch.delenv("CV_BANGLA_NER")
        spans = {"skill": [], "degree": ["বিএসসি"]}
        cv = {"skills": [], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট", ner=FakeBanglaNer(spans))
        assert out["education"][0]["degree"] == "B.Sc"

    def test_ner_unmapped_skill_ignored(self, monkeypatch):
        monkeypatch.delenv("CV_BANGLA_NER")
        spans = {"skill": ["ডেটা স্ট্রাকচার"], "degree": []}
        cv = {"skills": ["python"], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট", ner=FakeBanglaNer(spans))
        assert out["skills"] == ["python"]

    def test_ner_spans_recorded(self, monkeypatch):
        monkeypatch.delenv("CV_BANGLA_NER")
        spans = {"company": ["টেক সলিউশন"]}
        cv = {"skills": [], "education": []}
        out = _fuse_bangla_ner(cv, "বাংলা টেক্সট", ner=FakeBanglaNer(spans))
        assert out["ner_spans"]["company"] == ["টেক সলিউশন"]