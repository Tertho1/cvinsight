"""Tests for src/matcher/ — embedder, semantic_scorer, skill_overlap, ranker."""

from src.matcher.semantic_scorer import score as semantic_score
from src.matcher.skill_overlap import score as skill_overlap_score
from src.matcher.ranker import match_cv, rank_cvs


class TestSemanticScorer:
    def test_identical_texts_high_similarity(self):
        sim = semantic_score("Python developer with Django experience", "Python developer with Django experience")
        assert sim > 0.9

    def test_different_texts_low_similarity(self):
        sim = semantic_score("Python developer", "Cooking recipes for Italian pasta")
        assert sim < 0.5

    def test_empty_cv_text_returns_zero(self):
        sim = semantic_score("", "Python developer")
        assert sim == 0.0

    def test_empty_jd_text_returns_zero(self):
        sim = semantic_score("Python developer", "")
        assert sim == 0.0

    def test_both_empty_returns_zero(self):
        sim = semantic_score("", "")
        assert sim == 0.0

    def test_score_between_zero_and_one(self):
        sim = semantic_score("Software engineer skilled in Java", "Looking for a Java backend engineer")
        assert 0.0 <= sim <= 1.0


class TestSkillOverlap:
    def test_perfect_match(self):
        ratio, missing = skill_overlap_score(["Python", "Django", "SQL"], "Looking for Python, Django, SQL developer")
        assert ratio == 1.0
        assert missing == []

    def test_no_match(self):
        ratio, missing = skill_overlap_score(["Java"], "Looking for Python developer")
        assert ratio == 0.0
        assert "python" in missing

    def test_partial_match(self):
        ratio, missing = skill_overlap_score(["Python", "Java"], "Looking for Python, Django, PostgreSQL developer")
        assert 0.0 < ratio < 1.0
        assert "django" in missing
        assert "postgresql" in missing

    def test_empty_jd_returns_perfect(self):
        ratio, missing = skill_overlap_score(["Python"], "")
        assert ratio == 1.0
        assert missing == []

    def test_empty_cv_skills(self):
        ratio, missing = skill_overlap_score([], "Looking for Python developer")
        assert ratio == 0.0
        assert "python" in missing


class TestRanker:
    def test_match_cv_returns_all_keys(self):
        result = match_cv("Python developer", ["Python", "Django"], "Python job", rubric_score=80)
        assert "final_match_score" in result
        assert "semantic_similarity" in result
        assert "skill_overlap" in result
        assert "missing_skills" in result

    def test_match_cv_with_cv_dict(self):
        cv = {"raw_text": "Python developer", "skills": ["Python", "Django"], "total_score": 80}
        result = match_cv(cv=cv, jd_text="Python job")
        assert "final_match_score" in result
        assert result["final_match_score"] > 0

    def test_match_cv_score_between_zero_and_one(self):
        result = match_cv("Java developer", ["Java"], "Looking for Python developer", rubric_score=50)
        assert 0.0 <= result["final_match_score"] <= 1.0

    def test_rank_cvs_sorts_by_score(self):
        cvs = [
            {"cv_id": "1", "name": "A", "raw_text": "Python developer", "skills": ["Python"], "total_score": 90},
            {"cv_id": "2", "name": "B", "raw_text": "Chef", "skills": ["cooking"], "total_score": 30},
        ]
        ranked = rank_cvs(cvs, "Looking for Python developer")
        assert len(ranked) == 2
        assert ranked[0]["cv_id"] == "1"
        assert ranked[0]["final_match_score"] >= ranked[1]["final_match_score"]

    def test_rank_cvs_empty_list(self):
        assert rank_cvs([], "Python job") == []

    def test_match_cv_missing_skills_listed(self):
        result = match_cv("Python developer", ["Python"], "Looking for Python, Django, SQL developer", rubric_score=60)
        assert "django" in result["missing_skills"]
        assert "sql" in result["missing_skills"]

    def test_cv_to_text_fallback(self):
        cv = {"skills": ["Python"], "experience": [{"title": "Dev", "company": "Acme", "description": "built stuff"}]}
        from src.matcher.ranker import _cv_to_text
        text = _cv_to_text(cv)
        assert "Dev" in text
        assert "Acme" in text
        assert "Python" in text
