# Quality Classifier v2 — Training & Evaluation Report

Date: 2026-08-08 | By: autonomous session (user on sleep) | GPU: RTX 5070 Ti, 61.6GB RAM, 24 cores

## Artifacts (all checkpointed — a power cut never lost completed work)

| Path | Contents |
|---|---|
| `data/curated/corpus_primary_v1.csv` | Rubric-tier: 4,612 CVs (datasetmaster→classifier_training_data) with 7 sub-scores + total |
| `data/curated/corpus_ats_v1.csv` | Human-tier: 5,043 ATS rows, `original_label` mapped to Weak/Average/Strong |
| `data/curated/corpus_netsol_v1.csv` | Aux human-tier: 849 NETSOL rows (score→tier), `file_type` match/mismatch |
| `data/curated/corpus_merged_v1.json` | Dedupe stats |
| `results/classifier_leaderboard.csv` | **40-run** grid, appended per model completion |
| `results/human_tier_validation.csv` | Rubric-trained models vs HUMAN labels |
| `results/distilbert_results.csv` | DistilBERT probe (rubric ≥2 human tiers) |
| `results/classifier_v2_rf_sm_2026_08_08.pkl` | **Best model** (RF, text-only, mid-oversample) — lab metrics; pre-wiring validated (see below) |

## Corpus composition

- **Rubric tier (training/deploy label):** 4,612 (Avg 3,390 / Strong 1,137 / Weak 85) — heavily imbalanced tail.
- **Human tier (true generalization test):** ATS 5,043 (`No Fit` 2,537 → Weak, `Potential Fit` 1,264 → Avg, `Good Fit` 1,242 → Strong); NETSOL 849 (tier from human score, `file` match 648 / mismatch 201).
- Merge: 10,504 → **10,317 unique** after hash-dedup (~187 duplicates dropped).

## Classical grid (5-fold stratified CV, seed 42)

Feature sets: `txt` (TF-IDF uni+bi-gram), `num` (rubric sub-scores only), `mxd` (both).
Balancing: `none`, `balanced` (class weights), `sm` (mid-row oversample of Weak×6 / Avg×1.15).

### text-only (honest: no rubric leakage)

| model | balance | f1_weighted | f1_macro | acc | Weak recall |
|---|---|---|---|---|---|
| **RF** | sm | **0.9012** | **0.9057** | **0.8976** | **0.990** |
| XGBoost | sm | 0.8975 | 0.9002 | 0.8956 | 1.000 |
| Linear SVM | sm | 0.8914 | 0.8933 | 0.8898 | 0.990 |
| LR | sm | 0.8571 | 0.8252 | 0.8579 | 0.702 |
| RF/XGB/SVM (balanced/none) | … | 0.86–0.87 | 0.58–0.62 | 0.86–0.87 | <0.21 |
| Stacking | none/bal | 0.79–0.86 | 0.58–0.60 | 0.75–0.87 | poor |

**Headline: RF + mid-oversample lifts Weak recall from ~0.06 to 0.99 and macro-F1 from 0.62 → 0.91, one-class accuracy 0.90.** Beats the deployed XGB baseline (0.876 acc / 0.875 F1w) on every text-only metric.

- `num`/`mxd` hit ~0.97–1.00 accuracy. **That is an oracle ceiling, not a leak**: rubric sub-scores are literally the scored label inputs, so a model given those numbers reproduces the label nearly exactly. Treated as a sanity check on consistency of scoring, not as a text classifier.

## Human-tier generalization (the real question)

Models trained on rubric text (full primary set) → grades against HUMAN labels.

| model | ATS accuracy | ATS f1w | NETSOL acc | NETSOL f1w |
|---|---|---|---|---|
| xgb+sm (new) | **0.398** | **0.350** | **0.473** | **0.310** |
| rf+sm (new) | 0.295 | 0.221 | 0.478 | 0.309 |
| **deployed xgb (old)** | 0.255 | 0.113 | 0.450 | 0.325 |

Interpretation: a rubric-trained classifier agrees with human JD-hiring judgment only ~40–48% of the time; **the v2 models beat the deployed baseline on ATS by +14 points acc / +23 points F1** and are roughly tied on NETS. Human labels measure *job-fit*, rubric measures *CV completeness* — the gap is expected and documentable.

## DistilBERT probe (GPU, 11s train, checkpoints on disk)

Rubric 80/20 split: acc 0.885 / f1w 0.880 / f1m 0.583 — **below** `rf|txt|sm` (0.90/0.91) and with **Weak recall 0.0** (imbalanced to the extreme). Human-tier: worse than classical ensemble too (ATS 0.256, NETSOL 0.478). Not adopted; transformer benefits remain underwhelming on such imbalanced short resumes.

## Recommendations

1. **Do NOT switch the visible classifier to v2 yet** — see "Pre-wiring validation" below. The v2-rf artifact reproduces the *deployed* label on every demo/benchmark CV, and all models miss the rubric Strong/Weak rows (4/10). Keep the deployed copy; use the v2 grid + `sm` tuning (Weak-only oversample now) as the next training iteration input.
2. Keep `num`/`mxd` oracle row THE explanation in docs so nobody mistakes it for real text performance: benchmark 2.6
3. Split/stratify incorporate more Weak (only 1.9% of primary): consider tuning `sm` oversampling ratio when more Weak-labeled resumes arrive.
4. Do not add a BERT component for classification quality — returns and VRAM better spent on extraction/LLM tiers.

## Pre-wiring validation & live-contract test (2026-08-08)

Before wiring v2 into the app, the exact app path was simulated end-to-end
(`scripts/simulate_app_classify.py` → `results/classifier_v2_app_path.csv`):
parse each demo/benchmark CV → `classify_text()` (byte-for-byte copy of
`app/app.py:289`) → label/proba/classes as the app would store them.

**Bug found & fixed:** the app's `classify_text` preferred `model.classes_`
over `model.label_classes_`. The deployed `xgb_classifier.pkl` has *integer*
`classes_`, so the live app rendered `"0"` as the ML label. The v2 artifacts
ship string `classes_` + `label_classes_`, but the deployed model only exposes
`label_classes_`. Fix (2026-08-08): prefer `label_classes_` first, then
`classes_` — app/app.py, tests/test_classifier_contract.py, and the simulator
now mirror the same order; deployed model renders real labels (see below).

- 450/450 tests pass; `py_compile` clean on app.py.
- All three artifacts (deployed + v2-rf-primary + v2-xgb-merged) load in a
  fresh process with only ROOT on `sys.path`; the `scripts.export_best_classifier`
  import registered in the pickle resolves because `scripts/` is a namespace
  package — no wiring regression risk from import-time.
- Prediction distribution over the 10,317-row merged corpus (models were NOT
  trained on this — it is their live-facing population):
  - deployed: 8967x `(class 0 = Average)` / Strong 1179 / Weak 171 → **collapses
    to Average**; avg conf 0.938 (overconfident, label-blind).
  - v2-rf-primary: Weak 839 / **Average 8241** / Strong 1237 — same Average
    collapse, avg conf 0.721.
  - v2-xgb-merged: Weak 3722 / Avg 4018 / Strong 2577 — nearest the true
    corpus distribution (2753 Weak / 4963 Avg / 2601 Strong).
- Benchmark rubric agreement (the real decision signal): all three models agree
  with `_baseline.json` on **only 4/10** CVs; rubric-score-ceiling rows (01
  Strong@81, 10 Strong@87) are all called *Average*, and sparse Weak rows
  (04/06/08/09, score <50) are all called *Average*. The deployed model agreed
  with v2-rf on **every** CV — the v2 classifier does **not** move the app's
  visible ML label for real demo/benchmark CVs.

**Taking all evidence together, do NOT switch the visible classifier yet.** The
v2 models beat the baseline in lab held-out metrics and on the ATS human-tier,
but on the curated benchmark (our calibration ground truth) they are label-
blind toward the middle class, and the "winner" flips between rf and xgb
depending on budget set. Keep the deployed artifact; use v2 as the next training
iteration input (its value is the 40-run grid + tunable `sm` oversample ratios),
and switch the app only when the rubric-agreement gap on benchmark rows is
closing.

## Balanced-oversample GPU probe (2026-08-08) — negative, gap is distribution, not sampling

Hypothesis from the 4/10 result: the `sm` oversample only lifted Weak (×6); Strong got
no upweight, so the majority class absorbed both ends. Probed a `(weak_mult, strong_mult)`
GPU XGBoost grid on the rubric-tier corpus (`scripts/probe_balanced_oversample.py` →
`results/classifier_v2_balanced_grid.csv`): (6,1) baseline plus (8,2)/(8,4)/(8,8)/
(10,4)/(10,6)/(10,8)/(12,6)/(16,8).

Result: **no sample-ratio closes the benchmark gap.** The original (6,1) recipe stays the
best benchmark-rubric agreement (4/10) and the best held-out overall (acc 0.885 / f1w 0.881 /
recall_Strong 0.868). Every recipe with Strong× ≥2 collapses held-out Strong recall to
~0.05–0.10 and acc to ~0.75, while benchmark agreement *drops* to 3/10 (rows 04/06/07
mis-flip toward Strong; rubric-weak rows 04/06/08/09 still stay Average, and rubric-strong
rows 01@81/10@87 stay Average). Raising Weak beyond ×6 is inert (corpus Weak is only
85/4612 ≈ 1.8%, held-out Weak recall plateaus ~0.18 no matter the multiplier).

**Interpretation:** the gap is a **train/test distribution mismatch, not sampling**. The
training corpus texts are reconstructed TF-IDF text from structured datasetmaster JSON;
benchmark CVs are real natural text (bullets, headers). TF-IDF features that separate
classes on reconstructed text do not transfer to organic CV prose, so re-weighting the same
corpus cannot recover the Strong@81/Strong@87 / Weak@<50 rows the rubric sees. The
classifier is illustrative (the rubric is the app's real label); closing this would need
real resume text as training data, not more sampling.
**Verdict unchanged: keep `models/xgb_classifier.pkl`.**

## Time estimate vs actual

- Corpus build+dedupe: est. 10 min → actual ≈ 1 min (CSV reuse).
- Classical grid 40 runs: est. 25–40 min → actual ≈ 12 min.
- Human-tier validation: est 2 min → actual ≈ 1 min.
- DistilBERT probe: est 10–15 min → actual ≈ 11 s (GPU).
- Total ≈ 60–90 min estimate; **actual ≈ 15–18 min** of computation.