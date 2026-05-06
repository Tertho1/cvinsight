# src/schema_validator.py
#
# Utility functions for validating CV data against our schema.
# Used by every module to check its output before passing it forward.

from src.schema import CVSchema, Education, Experience, Project
from pydantic import ValidationError
from typing import Union
import json


def validate_cv(data: dict) -> tuple[bool, Union[CVSchema, str]]:
    """
    Tries to create a CVSchema object from a raw dictionary.

    Returns:
        (True, CVSchema object)   if the data is valid
        (False, error message)    if the data has problems

    Why return a tuple instead of raising an exception?
    Because in a pipeline, we want to log bad CVs and continue
    processing the rest — not crash the whole program.

    Example usage:
        ok, result = validate_cv(my_dict)
        if ok:
            print(result.summary())
        else:
            print("Validation failed:", result)
    """
    try:
        cv = CVSchema(**data)
        return True, cv
    except ValidationError as e:
        # Format the error nicely
        errors = []
        for err in e.errors():
            field = " → ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            errors.append(f"  Field '{field}': {msg}")
        error_str = "Validation errors:\n" + "\n".join(errors)
        return False, error_str


def validate_cv_from_json(json_str: str) -> tuple[bool, Union[CVSchema, str]]:
    """
    Same as validate_cv but accepts a JSON string instead of a dict.
    Useful when loading saved CV results from disk.
    """
    try:
        data = json.loads(json_str)
        return validate_cv(data)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def quick_check(cv: CVSchema) -> list[str]:
    """
    Runs a set of logical checks on a CVSchema object
    AFTER it has been created and validated.

    These are business-logic checks that Pydantic can't do:
    - Is the score in valid range?
    - Does the label match the score?
    - Are there suspiciously many skills?

    Returns a list of warning strings (empty list = all good).
    """
    warnings = []

    if cv.total_score < 0 or cv.total_score > 100:
        warnings.append(f"total_score {cv.total_score} is outside 0–100 range")

    valid_labels = {"Strong", "Average", "Weak", ""}
    if cv.label not in valid_labels:
        warnings.append(f"label '{cv.label}' is not one of: Strong, Average, Weak")

    if cv.label == "Strong" and cv.total_score < 80:
        warnings.append(f"Label is 'Strong' but score is only {cv.total_score}")

    if cv.label == "Weak" and cv.total_score >= 50:
        warnings.append(
            f"Label is 'Weak' but score is {cv.total_score} (should be < 50)"
        )

    if len(cv.skills) > 60:
        warnings.append(
            f"Unusually high skill count: {len(cv.skills)} — possible extraction error"
        )

    section_total = sum(
        [
            cv.section_scores.experience,
            cv.section_scores.projects,
            cv.section_scores.skills,
            cv.section_scores.education,
            cv.section_scores.certifications,
            cv.section_scores.languages,
            cv.section_scores.leadership,
        ]
    )
    if cv.total_score > 0 and abs(section_total - cv.total_score) > 2:
        warnings.append(
            f"Section scores sum to {section_total} but total_score is {cv.total_score}"
        )

    return warnings
