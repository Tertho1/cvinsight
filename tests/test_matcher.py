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

    def test_empty_jd_returns_zero(self):
        ratio, missing = skill_overlap_score(["Python"], "")
        assert ratio == 0.0
        assert missing == []

    def test_jd_no_taxonomy_skills_returns_neutral(self):
        ratio, missing = skill_overlap_score(["Python"], "Senior role requiring niche proprietary stack")
        assert ratio == 0.5
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


class TestSectionScoring:
    def test_score_sections_empty_jd_zero(self):
        from src.matcher.semantic_scorer import score_sections
        assert score_sections({"skills": "Python"}, "") == 0.0

    def test_score_sections_empty_cv_zero(self):
        from src.matcher.semantic_scorer import score_sections
        assert score_sections({}, "Python developer") == 0.0

    def test_score_sections_matching_skills_high(self):
        from src.matcher.semantic_scorer import score_sections
        sec = {"skills": "Python, Django, SQL, pytest"}
        sim = score_sections(sec, "Python developer with Django and SQL skills")
        assert sim > 0.5

    def test_score_sections_unrelated_low(self):
        from src.matcher.semantic_scorer import score_sections
        sec = {"skills": "Java backend"}
        sim = score_sections(sec, "Cooking recipes for Italian pasta")
        assert sim < 0.5

    def test_match_cv_section_mode_adds_mode_key(self):
        from src.matcher.ranker import match_cv
        res = match_cv(cv_text="Python developer", cv_skills=["Python"], jd_text="Python job",
                       rubric_score=50, mode="section", sections={"skills": "Python"})
        assert res["mode"] == "section"
        assert "final_match_score" in res

    def test_rank_cvs_section_mode(self):
        from src.matcher.ranker import rank_cvs
        cvs = [
            {"cv_id": "1", "name": "A", "sections": {"skills": "Python Django"}, "skills": ["Python", "Django"], "total_score": 90, "raw_text": ""},
            {"cv_id": "2", "name": "B", "sections": {"skills": "cooking"}, "skills": ["cooking"], "total_score": 30, "raw_text": ""},
        ]
        ranked = rank_cvs(cvs, "Looking for Python developer", mode="section")
        assert len(ranked) == 2
        assert ranked[0]["cv_id"] == "1"
        assert all(r["mode"] == "section" for r in ranked)


class TestBM25:
    def test_empty_inputs_zero(self):
        from src.matcher.bm25_scorer import score
        assert score("", "python") == 0.0
        assert score("python", "") == 0.0
        assert score("", "") == 0.0

    def test_exact_term_overlap_high(self):
        from src.matcher.bm25_scorer import score
        s = score("Senior Python backend developer with Django and PostgreSQL",
                  "We need a Senior Python backend developer with Django and PostgreSQL")
        assert s > 0.6

    def test_no_overlap_zero(self):
        from src.matcher.bm25_scorer import score
        assert score("Cooking recipes for Italian pasta", "Python developer") == 0.0

    def test_partial_overlap_mid(self):
        from src.matcher.bm25_scorer import score
        s = score("Java developer at a bank", "Python developer wanted")
        assert 0.0 < s < 1.0

    def test_score_bounded(self):
        from src.matcher.bm25_scorer import score
        s = score("python python python python python python",
                  "python python python python python python")
        assert 0.0 <= s <= 1.0

    def test_score_corpus_ranks_relevant_first(self):
        from src.matcher.bm25_scorer import score_corpus
        texts = [
            "Python backend developer",
            "Cooking recipes for Italian pasta",
            "Python data engineer with pandas",
        ]
        scores = score_corpus(texts, "Looking for a Python backend developer")
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_score_corpus_empty(self):
        from src.matcher.bm25_scorer import score_corpus
        assert score_corpus([], "python") == []


class TestRankerBM25:
    def test_bm25_zero_by_default(self):
        result = match_cv("Python developer", ["Python"], "Python job", rubric_score=80)
        assert result["weights"]["bm25"] == 0.0
        assert "bm25_score" in result

    def test_bm25_enabled_blends_score(self):
        result = match_cv("Python developer with Django", ["Python"],
                          "Python Django developer", rubric_score=80,
                          weights={"semantic": 0.5, "skill": 0.3, "rubric": 0.2, "bm25": 0.5})
        assert result["weights"]["bm25"] == 0.5
        assert result["bm25_score"] > 0
        assert result["final_match_score"] > 0

    def test_bm25_enabled_raises_score_vs_disabled(self):
        cv, jd = "Python Django PostgreSQL developer", "Need Python Django PostgreSQL developer"
        disabled = match_cv(cv, ["Python"], jd, rubric_score=80)
        enabled = match_cv(cv, ["Python"], jd, rubric_score=80,
                           weights={"semantic": 0.5, "skill": 0.3, "rubric": 0.2, "bm25": 0.5})
        assert enabled["final_match_score"] > disabled["final_match_score"]
