"""
tests/test_cleaner.py
CVInsight — Week 2, Day 12

Tests for src/parser/cleaner.py

Run:
    pytest tests/test_cleaner.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parser.cleaner import (
    clean_cv_text,
    normalize_whitespace,
    remove_special_chars,
    fix_encoding_artifacts,
    normalize_bullets,
    remove_page_artifacts,
)


# ---------------------------------------------------------------------------
# remove_special_chars
# ---------------------------------------------------------------------------

class TestRemoveSpecialChars:

    def test_removes_null_bytes(self):
        assert "\x00" not in remove_special_chars("hello\x00world")

    def test_removes_control_chars(self):
        result = remove_special_chars("text\x08with\x1fcontrol")
        assert "\x08" not in result
        assert "\x1f" not in result

    def test_keeps_newlines(self):
        result = remove_special_chars("line1\nline2")
        assert "\n" in result

    def test_keeps_tabs(self):
        result = remove_special_chars("col1\tcol2")
        assert "\t" in result

    def test_removes_box_drawing(self):
        # Box-drawing chars used as CV dividers
        result = remove_special_chars("Skills\u2500\u2500\u2500\u2500")
        assert "\u2500" not in result
        assert "Skills" in result

    def test_keeps_accented_chars(self):
        result = remove_special_chars("Ré sumé")
        assert "é" in result

    def test_empty_string(self):
        assert remove_special_chars("") == ""


# ---------------------------------------------------------------------------
# fix_encoding_artifacts
# ---------------------------------------------------------------------------

class TestFixEncodingArtifacts:

    def test_fixes_fi_ligature(self):
        result = fix_encoding_artifacts("pro\ufb01le")  # proﬁle
        assert "fi" in result
        assert "\ufb01" not in result

    def test_fixes_fl_ligature(self):
        result = fix_encoding_artifacts("re\ufb02ect")  # reﬂect
        assert "fl" in result

    def test_fixes_smart_quotes(self):
        result = fix_encoding_artifacts("\u201cHello\u201d")
        assert '"Hello"' in result

    def test_fixes_smart_apostrophe(self):
        result = fix_encoding_artifacts("don\u2019t")
        assert "don't" in result

    def test_fixes_en_dash_in_date_range(self):
        result = fix_encoding_artifacts("2020\u20132023")
        assert "2020-2023" in result

    def test_fixes_non_breaking_space(self):
        result = fix_encoding_artifacts("hello\u00a0world")
        assert "\u00a0" not in result
        assert " " in result

    def test_empty_string(self):
        assert fix_encoding_artifacts("") == ""


# ---------------------------------------------------------------------------
# normalize_bullets
# ---------------------------------------------------------------------------

class TestNormalizeBullets:

    def test_bullet_dot(self):
        result = normalize_bullets("• Built ML pipeline")
        assert result.startswith("- ")

    def test_bullet_arrow(self):
        result = normalize_bullets("➤ Led a team of 5")
        assert result.startswith("- ")

    def test_bullet_checkmark(self):
        result = normalize_bullets("✓ Certified AWS")
        assert result.startswith("- ")

    def test_bullet_asterisk(self):
        result = normalize_bullets("* Python experience")
        assert result.startswith("- ")

    def test_non_bullet_unchanged(self):
        result = normalize_bullets("Python, SQL, Docker")
        assert result == "Python, SQL, Docker"

    def test_multiple_bullets_in_text(self):
        text = "• Python\n• Docker\n• SQL"
        result = normalize_bullets(text)
        assert result.count("- ") == 3

    def test_empty_string(self):
        assert normalize_bullets("") == ""


# ---------------------------------------------------------------------------
# remove_page_artifacts
# ---------------------------------------------------------------------------

class TestRemovePageArtifacts:

    def test_removes_page_x_of_y(self):
        result = remove_page_artifacts("Some text\nPage 1 of 3\nMore text")
        assert "Page 1 of 3" not in result
        assert "Some text" in result

    def test_removes_page_x_of_y_lowercase(self):
        result = remove_page_artifacts("page 2 of 4")
        assert "page 2 of 4" not in result

    def test_removes_standalone_page_number(self):
        result = remove_page_artifacts("Content\n2\nMore content")
        assert "\n2\n" not in result
        assert "Content" in result

    def test_keeps_years(self):
        # 2020 is 4 digits — should NOT be removed as a page number
        # (our regex only targets 1-3 digit standalone numbers)
        result = remove_page_artifacts("Graduated 2020")
        assert "2020" in result

    def test_empty_string(self):
        assert remove_page_artifacts("") == ""


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:

    def test_collapses_multiple_spaces(self):
        result = normalize_whitespace("hello    world")
        assert "  " not in result
        assert "hello world" in result

    def test_strips_trailing_spaces_per_line(self):
        result = normalize_whitespace("hello   \nworld   ")
        for line in result.splitlines():
            assert not line.endswith(" ")

    def test_collapses_triple_blank_lines(self):
        result = normalize_whitespace("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_preserves_double_blank_lines(self):
        result = normalize_whitespace("a\n\nb")
        assert "\n\n" in result

    def test_strips_leading_trailing(self):
        result = normalize_whitespace("\n\ntext\n\n")
        assert result == "text"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""


# ---------------------------------------------------------------------------
# clean_cv_text (full pipeline)
# ---------------------------------------------------------------------------

class TestCleanCvText:

    def test_returns_string(self):
        assert isinstance(clean_cv_text("hello"), str)

    def test_empty_input(self):
        assert clean_cv_text("") == ""

    def test_none_safe(self):
        # Should handle empty gracefully
        assert clean_cv_text("") == ""

    def test_real_cv_text(self):
        raw = """John Doe\u00a0
john@email.com

\u2022 Built ML pipelines
\u2022 Led team of 5

Page 1 of 2

SKILLS
Python\ufb01le, Docker\u2013Kubernetes
"""
        result = clean_cv_text(raw)

        assert "\u00a0" not in result      # non-breaking space gone
        assert "\u2022" not in result      # bullet normalized
        assert "\ufb01" not in result      # fi ligature fixed
        assert "\u2013" not in result      # en dash fixed
        assert "Page 1 of 2" not in result # page artifact removed
        assert "John Doe" in result
        assert "Python" in result
        assert "Docker" in result

    def test_idempotent(self):
        """Cleaning already-clean text should return same result."""
        text = "John Doe\njohn@email.com\n\nSKILLS\nPython, SQL"
        once = clean_cv_text(text)
        twice = clean_cv_text(once)
        assert once == twice

    def test_no_excessive_blank_lines(self):
        raw = "section1\n\n\n\n\nsection2"
        result = clean_cv_text(raw)
        assert "\n\n\n" not in result

    def test_bullets_normalized(self):
        raw = "• Python\n▪ Docker\n✓ AWS"
        result = clean_cv_text(raw)
        assert "•" not in result
        assert result.count("- ") == 3