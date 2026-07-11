"""
scripts/day25_flag_borderline.py

Goal: pull out CVs sitting near the label boundaries (defined in
rubric_config.json -> borderline_bands) for the Day 26 manual review pass.

Output: data/processed/borderline_review.csv
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

INPUT_CSV = "data/processed/labeled_cvs.csv"
CONFIG_PATH = "config/rubric_config.json"
OUTPUT_CSV = "data/processed/borderline_review.csv"


def main():
    df = pd.read_csv(INPUT_CSV)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    bands = config["borderline_bands"]
    lo_lo, lo_hi = bands["lower"]
    hi_lo, hi_hi = bands["upper"]

    mask = df["total_score"].between(lo_lo, lo_hi) | df["total_score"].between(
        hi_lo, hi_hi
    )
    borderline = df[mask].copy()
    borderline["reviewer_label"] = ""  # fill in during Day 26
    borderline["reviewer_notes"] = ""

    os.makedirs("data/processed", exist_ok=True)
    borderline.to_csv(OUTPUT_CSV, index=False)

    print(
        f"{len(borderline)} / {len(df)} CVs flagged as borderline "
        f"({len(borderline)/len(df):.1%})"
    )
    print(f"Bands: {lo_lo}-{lo_hi} and {hi_lo}-{hi_hi}")
    print(f"Saved {OUTPUT_CSV} — fill in 'reviewer_label' for each row.")


if __name__ == "__main__":
    main()
