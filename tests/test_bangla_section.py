import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor.bangla_section import (
    BanglaSectionClassifier,
    get_bangla_section_classifier,
    classify_section,
    _LABEL_TO_SECTION,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                          "bangla_section_classifier.pkl")

# Bangla sample segments (from Onneshon's four classes)
BN_EXPERIENCE = "সিনিয়র সফ্টওয়্যার ইঞ্জিনিয়ার"
BN_EDUCATION = "বিএসসি ইন কম্পিউটার সায়েন্স অ্যান্ড ইঞ্জিনিয়ারিং"
BN_OBJECTIVE = ("অভিজ্ঞতা সম্পন্ন সফ্টওয়্যার ইঞ্জিনিয়ার হিসেবে একটি ডায়নামিক টিমে "
                "যোগদান করা যেখানে আমার জাভা এবং স্প্রিং ফ্রেমওয়ার্কের দক্ষতা কাজে "
                "লাগিয়ে প্রতিষ্ঠানের সাফল্যে অবদান রাখতে পারি")
BN_SKILL = "প্রোগ্রামিং ভাষা: জাভা, পাইথন, সি++"


@pytest.fixture(scope="module")
def clf():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("models/bangla_section_classifier.pkl not trained; run "
                    "scripts/train_bangla_section_classifier.py first")
    return BanglaSectionClassifier(MODEL_PATH)


class TestBanglaSectionClassifier:

    def test_loaded_true_when_model_exists(self, clf):
        assert clf.loaded is True

    def test_predict_experience_label(self, clf):
        assert clf.predict(BN_EXPERIENCE) == "Experience"

    def test_predict_education_label(self, clf):
        assert clf.predict(BN_EDUCATION) == "Education"

    def test_predict_objective_label(self, clf):
        assert clf.predict(BN_OBJECTIVE) == "Objective"

    def test_predict_skill_label(self, clf):
        assert clf.predict(BN_SKILL) == "Skill"

    def test_predict_blank_returns_none(self, clf):
        assert clf.predict("") is None
        assert clf.predict("   ") is None
        assert clf.predict(None) is None

    def test_predict_section_mapping(self, clf):
        assert clf.predict_section(BN_EXPERIENCE) == "experience"
        assert clf.predict_section(BN_EDUCATION) == "education"
        assert clf.predict_section(BN_OBJECTIVE) == "summary"
        assert clf.predict_section(BN_SKILL) == "skills"

    def test_predict_section_blank_returns_none(self, clf):
        assert clf.predict_section("") is None
        assert clf.predict_section(None) is None

    def test_missing_model_returns_none(self):
        c = BanglaSectionClassifier("models/does_not_exist.pkl")
        assert c.predict(BN_SKILL) is None
        assert c.predict_section(BN_SKILL) is None
        assert c.loaded is False


class TestConvenience:

    def test_singleton_is_same_instance(self):
        assert get_bangla_section_classifier() is get_bangla_section_classifier()

    def test_classify_section_experience(self):
        if os.path.exists(MODEL_PATH):
            assert classify_section(BN_EXPERIENCE) == "experience"

    def test_classify_section_blank(self):
        assert classify_section("") is None

    def test_label_map_covers_all_canonical_sections(self):
        assert set(_LABEL_TO_SECTION.values()) == {
            "summary", "experience", "skills", "education"
        }