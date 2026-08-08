"""Build curated quality-label corpora for classifier experiments.

Outputs (all under data/curated/, each step checkpointed by existence):
  - corpus_primary_v1.csv   rubric-tier: classifier_training_data (datasetmaster-derived) + numeric sub-scores
  - corpus_ats_v1.csv       human-tier: ATS original_label mapped to Quality {Weak,Average,Strong}
  - corpus_netsol_v1.csv    human-tier aux (caveat: score not strict rubric, file_type match/mismatch)
  - corpus_summary.json     counts, dedupe stats, class distributions
  - run.log                 step log
Run from repo root.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURATED = DATA / "curated"
CURATED.mkdir(exist_ok=True)
LOG = CURATED / "run.log"


def log(msg, *args):
    if args:
        msg = msg % args
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def text_hash(s):
    return hashlib.sha1(" ".join(str(s).lower().split()).encode("utf-8")).hexdigest()


def done(path):
    return path.exists()


cls_range = {"Strong": (7.0, 10.1), "Average": (4.0, 7.0), "Weak": (-0.1, 4.0)}


def score_to_label(v):
    for label, (lo, hi) in cls_range.items():
        if lo <= v < hi:
            return label
    return "Strong"


def score_to_tier(v):
    """Map NETSOL human score (0-9.55) to Weak/Average/Strong tier."""
    return score_to_label(float(v))


def load_primary():
    """Rubric-tier: classifier_training_data (4,612 rows w/ label + numeric sub-scores)."""
    df = pd.read_csv(DATA / "processed" / "classifier_training_data.csv")
    df = df.rename(columns={"cv_id": "doc_id"})
    df["source"] = "rubric"
    df["label_source"] = "rubric"
    df["text_hash"] = df["raw_text"].map(text_hash)
    # drop empty-text rows
    df = df[df["raw_text"].fillna("").str.strip() != ""]
    # weak class too thin to split reliably; mid-oversample only, not delete
    return df[["doc_id", "raw_text", "label", "total_score", "score_experience",
               "score_projects", "score_skills", "score_education",
               "score_certifications", "score_languages", "score_leadership",
               "text_hash", "source", "label_source"]]


def load_ats():
    """Human-tier: ats_scores original_label mapped to quality 3-class."""
    df = pd.read_csv(DATA / "processed" / "ats_scores_clean.csv")
    m = {"No Fit": "Weak", "Potential Fit": "Average", "Good Fit": "Strong"}
    df["label"] = df["original_label"].map(m)
    df["label_source"] = "human"
    df["source"] = "ats"
    df["text_hash"] = df["text"].map(text_hash)
    # human ats_score retained as a soft label, but not train label
    df = df[df["text"].astype(str).str.strip() != ""]
    try:
        df = df[df["label"].notna()]
    except Exception:
        pass
    out = df[["text", "label", "ats_score", "text_hash", "source", "label_source"]].copy()
    # typical columns for primary to reuse
    out = out.rename(columns={"text": "raw_text"})
    for c in ["score_experience", "score_projects", "score_skills", "score_education",
              "score_certifications", "score_languages", "score_leadership", "total_score"]:
        out[c] = pd.NA
    out["doc_id"] = out.index.map(lambda i: f"ats-{i}")
    return out[["doc_id", "raw_text", "label", "total_score", "ats_score", "text_hash",
                "source", "label_source"]]


def load_netsol():
    """NETSOL: aux human-tier, scored file_type match/mismatch, predicted label tier from score."""
    df = pd.read_csv(DATA / "processed" / "netsol_clean.csv")
    df["label"] = df["score"].map(score_to_tier)
    df["label_source"] = "human"
    df["source"] = "netsol"
    # concatenate list-like strings for a full-doc text view
    df["raw_text"] = df.apply(lambda r: "\n".join(
        [str(r["candidate_name"] or ""), "Job Description:", str(r["job_description"] or ""),
         "Skills:", str(r["skills"] or ""), "Education:", str(r["education"] or ""),
         "Experience:", str(r["experience"] or "")]), axis=1)
    df["total_score"] = df["score"]
    df["text_hash"] = df["raw_text"].map(text_hash)
    df["doc_id"] = df.index.map(lambda i: f"netsol-{i}")
    for c in ["score_experience", "score_projects", "score_skills", "score_education",
              "score_certifications", "score_languages", "score_leadership"]:
        df[c] = pd.NA
    return df[["doc_id", "raw_text", "label", "total_score", "text_hash", "source",
               "label_source", "file_type"]]


def main():
    log("%s Build curated corpora", "/BEGIN")
    if not done(CURATED / "corpus_primary_v1.csv"):
        dfp = load_primary()
        dfp.to_csv(CURATED / "corpus_primary_v1.csv", index=False)
        log("primary corpus written (%d rows)", len(dfp))
    else:
        dfp = pd.read_csv(CURATED / "corpus_primary_v1.csv")
        log("primary corpus SKIP (exists, %d rows)", len(dfp))

    if not done(CURATED / "corpus_ats_v1.csv"):
        dfa = load_ats()
        dfa.to_csv(CURATED / "corpus_ats_v1.csv", index=False)
        log("ats corpus written (%d rows)", len(dfa))
    else:
        dfa = pd.read_csv(CURATED / "corpus_ats_v1.csv")
        log("ats corpus SKIP (exists, %d rows)", len(dfa))

    if not done(CURATED / "corpus_netsol_v1.csv"):
        dfn = load_netsol()
        dfn.to_csv(CURATED / "corpus_netsol_v1.csv", index=False)
        log("netsol corpus written (%d rows)", len(dfn))
    else:
        dfn = pd.read_csv(CURATED / "corpus_netsol_v1.csv")
        log("netsol corpus SKIP (exists, %d rows)", len(dfn))

    # dedupe across corpora by text_hash. Priority: rubric > ats > netsol (keep first preferred)
    if not done(CURATED / "corpus_merged_v1.csv"):
        dfp["_p"] = 0
        dfa["_p"] = 1
        dfn["_p"] = 2
        merged = pd.concat([dfp, dfa, dfn], sort=False)
        log("total rows %d", len(merged))
        # keep earliest priority per hash, remove dup rows after sort
        merged = merged.sort_values("_p").drop_duplicates(subset="text_hash", keep="first")
        merged.to_csv(CURATED / "corpus_merged_v1.csv", index=False)
        log("merged corpus written (%d after dedupe)", len(merged))
    else:
        merged = pd.read_csv(CURATED / "corpus_merged_v1.csv")
        log("merged corpus SKIP (exists, %d rows)", len(merged))

    # class distribution summary
    before = len(dfp) + len(dfa) + len(dfn)
    summary = {
        "primary": {k: int(v) for k, v in dfp["label"].value_counts().to_dict().items()},
        "ats": {k: int(v) for k, v in dfa["label"].value_counts().to_dict().items()},
        "netsol_human": {k: int(v) for k, v in dfn["label"].value_counts().to_dict().items()},
        "netsol_filetype": {k: int(v) for k, v in dfn["file_type"].value_counts().to_dict().items()},
        "merged": {k: int(v) for k, v in merged["label"].value_counts().to_dict().items()},
        "dedup_dropped": int(before - len(merged)),
    }
    with open(CURATED / "corpus_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("summary: %s", json.dumps(summary, indent=2))
    log("/END corpus build")


if __name__ == "__main__":
    sys.exit(main())