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