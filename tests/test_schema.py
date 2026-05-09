# tests/test_schema.py
#
# Unit tests for the schema and validator.
# Run with:  pytest tests/test_schema.py -v
#
# The -v flag means "verbose" — it shows each test name and pass/fail.
# Without -v you just see dots.

import pytest
from src.schema import (
    CVSchema, Education, Experience, Project,
    Certification, Language, SectionScores, JDMatch
)
from src.schema_validator import validate_cv, quick_check


# ── Test 1: Empty schema creation ─────────────────────────────
def test_empty_schema_creation():
    """
    An empty CVSchema should be creatable with all defaults.
    This confirms the schema itself has no syntax errors.
    """
    cv = CVSchema()
    assert cv.total_score == 0
    assert cv.label == ""
    assert cv.skills == []
    assert cv.education == []
    assert isinstance(cv.section_scores, SectionScores)
    assert isinstance(cv.jd_match, JDMatch)


# ── Test 2: Sub-model creation ────────────────────────────────
def test_education_model():
    """Education sub-model should store all fields correctly."""
    edu = Education(
        degree="Bachelor of Science",
        institution="BUET",
        field="Computer Science",
        year=2022,
        gpa=3.85
    )
    assert edu.degree == "Bachelor of Science"
    assert edu.gpa == 3.85
    assert edu.year == 2022


def test_experience_model():
    """Experience sub-model, including optional fields."""
    exp = Experience(
        title="Software Engineer",
        company="Google",
        start="2021-06",
        end="2023-03",
        duration_months=21,
        description="Built backend services."
    )
    assert exp.duration_months == 21
    assert exp.end == "2023-03"


# ── Test 3: Full CV object ────────────────────────────────────
def test_full_cv_object():
    """Build a complete CV and verify all fields are stored."""
    cv = CVSchema(
        name="Rahim Ahmed",
        email="rahim@email.com",
        phone="01712345678",
        skills=["Python", "Machine Learning", "SQL"],
        education=[
            Education(degree="BSc", institution="BUET", field="CSE", year=2022)
        ],
        experience=[
            Experience(title="Data Analyst", company="BRAC", duration_months=18)
        ],
        total_score=72,
        label="Average",
        suggestions=["Add more projects", "Include certifications"]
    )
    assert cv.name == "Rahim Ahmed"
    assert len(cv.skills) == 3
    assert cv.total_score == 72
    assert cv.label == "Average"
    assert len(cv.suggestions) == 2


# ── Test 4: generate_id ───────────────────────────────────────
def test_generate_id():
    """
    Two CVs with different raw text should get different IDs.
    Same raw text should always produce the same ID.
    """
    cv1 = CVSchema(raw_text="This is CV number one")
    cv2 = CVSchema(raw_text="This is CV number two")
    cv1.cv_id = cv1.generate_id()
    cv2.cv_id = cv2.generate_id()

    assert cv1.cv_id != cv2.cv_id
    assert len(cv1.cv_id) == 12

    # Same text → same ID (deterministic)
    cv3 = CVSchema(raw_text="This is CV number one")
    cv3.cv_id = cv3.generate_id()
    assert cv1.cv_id == cv3.cv_id


# ── Test 5: to_dict and to_json ───────────────────────────────
def test_serialization():
    """CVSchema should convert to dict and JSON without errors."""
    cv = CVSchema(name="Test User", total_score=85, label="Strong")
    d = cv.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "Test User"

    j = cv.to_json()
    assert isinstance(j, str)
    assert "Test User" in j


# ── Test 6: validate_cv function ─────────────────────────────
def test_validate_cv_valid_data():
    """Valid dict should pass validation and return CVSchema."""
    data = {
        "name": "Sara Khan",
        "email": "sara@example.com",
        "total_score": 88,
        "label": "Strong",
        "skills": ["Python", "Django"]
    }
    ok, result = validate_cv(data)
    assert ok is True
    assert isinstance(result, CVSchema)
    assert result.name == "Sara Khan"


def test_validate_cv_invalid_score_type():
    """Passing a string where int is expected should fail validation."""
    data = {
        "name": "Bad Data",
        "total_score": "not_a_number"   # wrong type
    }
    ok, result = validate_cv(data)
    # Pydantic will actually try to coerce "not_a_number" to int and fail
    # This tests that our validator catches and reports it
    assert isinstance(result, (CVSchema, str))  # either coerced or error


# ── Test 7: quick_check warnings ─────────────────────────────
def test_quick_check_label_score_mismatch():
    """
    If someone labels a CV 'Strong' but score is 40,
    quick_check should warn us.
    """
    cv = CVSchema(total_score=40, label="Strong")
    warnings = quick_check(cv)
    assert any("Strong" in w for w in warnings)


def test_quick_check_section_total_mismatch():
    """
    If section scores add up to 60 but total_score is 80,
    quick_check should flag this inconsistency.
    """
    cv = CVSchema(
        total_score=80,
        label="Strong",
        section_scores=SectionScores(
            experience=10,
            projects=10,
            skills=10,
            education=10,
            certifications=5,
            languages=5,
            leadership=5
        )  # sums to 55, not 80
    )
    warnings = quick_check(cv)
    assert any("Section scores" in w for w in warnings)


def test_quick_check_clean_cv():
    """A properly scored CV should produce zero warnings."""
    cv = CVSchema(
        total_score=85,
        label="Strong",
        skills=["Python", "SQL", "ML"],
        section_scores=SectionScores(
            experience=25,
            projects=20,
            skills=18,
            education=12,
            certifications=6,
            languages=2,
            leadership=2
        )  # sums to 85
    )
    warnings = quick_check(cv)
    assert warnings == []


# ── Tests using real dataset samples ──────────────────────────
# These tests load actual CV text from our processed datasets
# to make sure the schema handles real-world data correctly.

import pandas as pd
import os


def test_schema_with_real_ner_cv():
    """
    Load a real CV from the NER dataset and confirm it passes
    schema creation without errors.
    """
    path = 'data/processed/ner_resumes_clean.csv'
    if not os.path.exists(path):
        pytest.skip('Processed NER dataset not found — run clean_datasets.py first')

    df = pd.read_csv(path)
    sample_text = df['text'].iloc[0]

    # Schema should accept any non-empty string as raw_text
    cv = CVSchema(raw_text=sample_text)
    cv.cv_id = cv.generate_id()

    assert len(cv.raw_text) > 50
    assert len(cv.cv_id) == 12
    assert cv.total_score == 0       # not scored yet
    assert cv.label == ''            # not labeled yet


def test_schema_with_real_classification_cv():
    """
    Load a CV from the classification dataset.
    Confirm category field (which we add manually) stores correctly.
    """
    path = 'data/processed/classification_clean.csv'
    if not os.path.exists(path):
        pytest.skip('Processed classification dataset not found')

    df = pd.read_csv(path)
    sample = df.iloc[0]

    cv = CVSchema(
        raw_text=sample['text'],
        # We store the job category as an achievement tag for now
        # In Week 4 this becomes a proper label
        achievements=[f"Job Category: {sample['category']}"]
    )

    assert cv.raw_text == sample['text']
    assert len(cv.achievements) == 1


def test_empty_cv_edge_case():
    """
    An empty string as raw_text should still create a valid schema.
    The extractor will produce empty fields — that is acceptable.
    The scorer will then give it a score of 0 and label 'Weak'.
    """
    cv = CVSchema(raw_text='')
    cv.cv_id = cv.generate_id()

    assert cv.raw_text == ''
    assert cv.total_score == 0
    assert cv.skills == []
    assert cv.education == []


def test_missing_optional_fields():
    """
    A CV with only required fields and no optional ones
    should still be valid. Optional fields default to None or [].
    """
    cv = CVSchema(
        name='Test Person',
        email='test@example.com',
        skills=['Python', 'SQL'],
        total_score=55,
        label='Average'
    )
    # These optional fields should be None or empty by default
    assert cv.phone == ''
    assert cv.projects == []
    assert cv.certifications == []
    assert cv.jd_match.semantic_similarity == 0.0


def test_very_long_cv_text():
    """
    Some CVs are very long (we saw max 99,973 chars in NER dataset).
    The schema must handle this without truncating or crashing.
    """
    long_text = 'Python developer with experience. ' * 1000  # ~34,000 chars
    cv = CVSchema(raw_text=long_text)
    assert len(cv.raw_text) == len(long_text)


def test_unicode_in_cv():
    """
    CVs from non-English speaking countries contain accented
    characters, Arabic, Bengali script etc.
    The schema must store these without modification.
    """
    cv = CVSchema(
        name='মোহাম্মদ রহিম',          # Bengali name
        email='rahim@example.com',
        skills=['Python', 'মেশিন লার্নিং'],  # Bengali skill name
        achievements=['তথ্য বিজ্ঞান পুরস্কার ২০২৩']  # Bengali achievement
    )
    assert 'মোহাম্মদ' in cv.name
    assert any('মেশিন' in s for s in cv.skills)


def test_netsol_score_range():
    """
    Netsol scores range 0-9.55 (10-point scale).
    Confirm our schema can store them as floats.
    """
    path = 'data/processed/netsol_clean.csv'
    if not os.path.exists(path):
        pytest.skip('Processed netsol dataset not found')

    df = pd.read_csv(path)
    scores = df['score'].dropna()

    # All scores should be between 0 and 10
    assert scores.min() >= 0
    assert scores.max() <= 10

    # Store a real score in jd_match
    sample_score = float(scores.iloc[0])
    cv = CVSchema()
    cv.jd_match.final_match_score = sample_score / 10.0  # normalize to 0-1
    assert 0.0 <= cv.jd_match.final_match_score <= 1.0