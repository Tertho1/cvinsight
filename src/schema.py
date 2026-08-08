# src/schema.py
#
# This file is the single source of truth for the data structure
# that flows through our entire pipeline.
#
# Every module in this project reads from or writes to this schema:
#   - parser      → produces raw text
#   - extractor   → fills in the fields below
#   - scorer      → fills in criteria_scores, total_score, label
#   - suggester   → fills in suggestions
#   - matcher     → fills in match
#   - app         → reads and displays the final object
#
# RULE: If you need to add a new field to the CV output,
#       add it here first, then update the relevant module.

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices
import hashlib
import json


# ──────────────────────────────────────────────
# SUB-MODELS
# These represent individual items inside lists.
# For example, a person can have multiple degrees,
# so Education is its own model used inside a list.
# ──────────────────────────────────────────────

class Education(BaseModel):
    """One educational qualification."""
    degree: str = ""               # e.g. "Bachelor of Science"
    institution: str = ""          # e.g. "BUET"
    field: str = ""                # e.g. "Computer Science"
    year: Optional[int] = None     # graduation year, e.g. 2022
    gpa: Optional[float] = None    # e.g. 3.8 — None if not mentioned


class Experience(BaseModel):
    """One job or work experience entry."""
    title: str = ""                # e.g. "Software Engineer"
    company: str = ""              # e.g. "Google"
    start: Optional[str] = None    # e.g. "2021-06"  (YYYY-MM format)
    end: Optional[str] = None      # e.g. "2023-03" or None if current job
    duration_months: int = 0       # calculated from start and end
    description: str = ""          # raw text of the job description


class Project(BaseModel):
    """One project entry."""
    name: str = ""                 # e.g. "E-commerce Website"
    tools: List[str] = []          # e.g. ["React", "Node.js", "MongoDB"]
    description: str = ""          # what the project does
    link: Optional[str] = None     # GitHub or live URL if mentioned


class Certification(BaseModel):
    """One certification or course."""
    name: str = ""                 # e.g. "AWS Certified Solutions Architect"
    issuer: str = ""               # e.g. "Amazon"
    year: Optional[int] = None     # year obtained


class Language(BaseModel):
    """One spoken/written language."""
    language: str = ""             # e.g. "English"
    proficiency: Optional[str] = None  # e.g. "C1", "Fluent", "Native"


class SectionScores(BaseModel):
    """
    Stores the score for each rubric section separately.
    This lets the UI show a breakdown chart:
    Experience: 20/25, Skills: 15/20, etc.
    All default to 0 before scoring runs.
    """
    experience: int = 0       # max 25
    projects: int = 0         # max 20
    skills: int = 0           # max 20
    education: int = 0        # max 15
    certifications: int = 0   # max 10
    languages: int = 0        # max 5
    leadership: int = 0       # max 5


class JDMatch(BaseModel):
    """
    Results of matching this CV against a Job Description.
    Only populated in Week 6 (V2 feature).
    All fields default to 0 / empty until then.
    """
    semantic_similarity: float = 0.0   # cosine similarity score 0–1
    skill_overlap: float = 0.0         # ratio of JD skills found in CV
    final_match_score: float = 0.0     # weighted combination of above
    missing_skills: List[str] = []     # skills in JD but not in CV


class CriterionScore(BaseModel):
    """
    One row in the auditable criteria_scores breakdown.

    Replaces the fixed SectionScores with a configurable list (see
    config/default_criteria.json). Every entry carries its own method tag,
    max_points, weight and a human-readable rationale so each score is
    explainable. `overridden_by` records the criterion name that took
    precedence (populated only when an override applies).
    """
    name: str = ""                 # e.g. "experience"
    score: int = 0                 # points earned (0..max_points)
    max_points: int = 0            # cap for this criterion
    weight: float = 0.0            # relative weight (normally max/100)
    method: str = ""               # e.g. "rule_years", "rule_count"
    rationale: str = ""            # human-readable explanation, no LLM
    overridden_by: Optional[str] = None  # which criterion took precedence


# ──────────────────────────────────────────────
# MAIN SCHEMA
# This is the master object that represents one CV
# flowing through the entire pipeline.
# ──────────────────────────────────────────────

class CVSchema(BaseModel):
    """
    The complete structured representation of one CV.

    Pipeline flow:
        Raw file → [parser] → raw text
                 → [extractor] → fills all fields below
                 → [scorer] → fills criteria_scores, total_score, label
                 → [suggester] → fills suggestions
                 → [matcher] → fills match  (Week 6)
    """

    # --- Identity ---
    cv_id: str = ""            # unique ID generated from file content hash
    raw_text: str = ""         # the full extracted text (kept for debugging)
    language: str = "en"       # "en" | "bangla" (Bengali script route)

    # --- Contact ---
    name: str = ""
    email: str = ""
    phone: str = ""

    # --- Main sections ---
    education: List[Education] = []
    experience: List[Experience] = []
    skills: List[str] = []
    projects: List[Project] = []
    certifications: List[Certification] = []
    languages: List[Language] = []
    achievements: List[str] = []
    leadership: List[str] = []

    # --- Scoring (filled by scorer module) ---
    # Legacy dict view (kept for backward compatibility).
    section_scores: SectionScores = Field(default_factory=SectionScores)
    # Canonical, auditable breakdown (V2). Scorer fills both.
    criteria_scores: List[CriterionScore] = Field(default_factory=list)
    total_score: int = 0
    label: str = ""            # "Strong" | "Average" | "Weak"

    # --- Feedback (filled by suggester module) ---
    suggestions: List[str] = []

    # --- JD Matching (filled in Week 6) ---
    # Renamed jd_match -> match (V2). Accepts legacy "jd_match" on load.
    match: JDMatch = Field(
        default_factory=JDMatch,
        validation_alias=AliasChoices("match", "jd_match"),
    )

    @property
    def jd_match(self) -> JDMatch:
        """Backward-compatible alias for `match` (pre-V2 name)."""
        return self.match

    @jd_match.setter
    def jd_match(self, value) -> None:
        self.match = value


    # ──────────────────────────────────────────
    # HELPER METHODS
    # These are utility functions attached to the
    # schema object for convenience.
    # ──────────────────────────────────────────

    def generate_id(self) -> str:
        """
        Creates a unique ID for this CV by hashing its raw text.
        Why hashing? Two identical CVs will always get the same ID.
        Two different CVs will always get different IDs.
        This is useful for deduplication later.
        """
        return hashlib.md5(self.raw_text.encode()).hexdigest()[:12]

    def to_json(self, indent: int = 2) -> str:
        """
        Exports the entire CV object as a formatted JSON string.
        Useful for saving results to disk or sending to the UI.
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict:
        """
        Exports the CV object as a plain Python dictionary.
        Useful for converting to pandas DataFrame rows.
        """
        return self.model_dump(by_alias=True, exclude_none=True)

    def summary(self) -> str:
        """
        Returns a short human-readable summary of this CV.
        Useful for quick debugging — print(cv.summary())
        """
        return (
            f"CV ID    : {self.cv_id}\n"
            f"Name     : {self.name or 'Unknown'}\n"
            f"Score    : {self.total_score}/100\n"
            f"Label    : {self.label or 'Not scored yet'}\n"
            f"Skills   : {len(self.skills)} found\n"
            f"Exp      : {len(self.experience)} roles\n"
            f"Projects : {len(self.projects)}\n"
            f"Education: {len(self.education)} entries\n"
        )