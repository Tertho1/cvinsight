"""
scripts/day24_score_distribution.py

Goal: run score_cv() on all extracted CVs from Week 3, inspect the score
distribution, and confirm labels aren't all landing in one bucket.

Outputs:
  - data/processed/labeled_cvs.csv
  - data/processed/score_distribution.png
"""

import json
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.scorer.scorer import score_cvs  # noqa: E402

INPUT_JSON = "data/processed/extracted_cvs.json"
OUTPUT_CSV = "data/processed/labeled_cvs.csv"
OUTPUT_PNG = "data/processed/score_distribution.png"


def main():
    with open(INPUT_JSON, encoding="utf-8") as f:
        cvs = json.load(f)

    print(f"Loaded {len(cvs)} extracted CVs")

    scored = score_cvs(cvs)

    rows = []
    for cv in scored:
        row = {"cv_id": cv["cv_id"], "name": cv.get("name", ""),
               "total_score": cv["total_score"], "label": cv["label"]}
        row.update({f"score_{k}": v for k, v in cv["section_scores"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nMean total_score: {df['total_score'].mean():.1f}")
    print(f"Std total_score:  {df['total_score'].std():.1f}")

    # Flag if distribution is degenerate (Risk Register item)
    label_counts = df["label"].value_counts(normalize=True)
    if label_counts.max() > 0.85:
        print(f"\nWARNING: '{label_counts.idxmax()}' is {label_counts.max():.0%} "
              "of all CVs. Rubric weights likely need adjustment (see Day 26).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["total_score"], bins=20, edgecolor="black")
    axes[0].set_title("Total Score Distribution")
    axes[0].set_xlabel("Score (0-100)")
    axes[0].set_ylabel("Count")

    df["label"].value_counts().reindex(["Weak", "Average", "Strong"]).plot(
        kind="bar", ax=axes[1], color=["#d62728", "#ff7f0e", "#2ca02c"]
    )
    axes[1].set_title("Label Distribution")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=120)
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()