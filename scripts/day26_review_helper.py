"""
scripts/day26_review_helper.py

Run AFTER you've manually filled 'reviewer_label' in borderline_review.csv
for at least 50 rows. Reports agreement rate and, if weights look off,
tells you which rubric_config.json section to inspect.

After editing rubric_config.json, call scorer.reload_config() and re-run
Day 24 to confirm the distribution improved.
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())
from src.scorer.scorer import score_cv, reload_config  # noqa: E402

BORDERLINE_CSV = "data/processed/borderline_review.csv"
EXTRACTED_JSON = "data/processed/extracted_cvs.json"


def main():
    df = pd.read_csv(BORDERLINE_CSV)
    reviewed = df[df["reviewer_label"].notna() & (df["reviewer_label"] != "")]

    if reviewed.empty:
        print(
            "No rows have 'reviewer_label' filled in yet. "
            "Manually review at least 50 rows first (see Day 26 instructions)."
        )
        return

    agree = (reviewed["reviewer_label"] == reviewed["label"]).mean()
    print(f"Reviewed: {len(reviewed)} CVs")
    print(f"Rubric/reviewer agreement: {agree:.1%}")

    mismatches = reviewed[reviewed["reviewer_label"] != reviewed["label"]]
    if not mismatches.empty:
        print("\nMismatch direction breakdown:")
        print(mismatches.groupby(["label", "reviewer_label"]).size())
        print(
            "\nMean section scores for mismatched CVs (inspect which section "
            "is over/under-weighted):"
        )
        section_cols = [c for c in df.columns if c.startswith("score_")]
        print(mismatches[section_cols].mean().sort_values())
    else:
        print("No mismatches — rubric weights look well-calibrated.")

    # Apply corrected labels back onto the main labeled set for downstream use
    with open(EXTRACTED_JSON, encoding="utf-8") as f:
        cvs = {c["cv_id"]: c for c in json.load(f)}
    for _, row in reviewed.iterrows():
        cv = cvs.get(row["cv_id"])
        if cv:
            cv["label"] = row["reviewer_label"]  # human correction wins

    print(
        "\nIf you edit config/rubric_config.json now, call reload_config() "
        "and re-run Day 24 before continuing."
    )


if __name__ == "__main__":
    main()
