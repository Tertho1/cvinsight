# Classifier v3 — Hybrid Architecture: Why v1/v2 Failed & What To Do

Date: 2026-08-08 | Status: design — implementation follows in this file tree

## 1. What the current classifier actually is (what we fed, and why it plateaus)

- **Data:** `data/curated/corpus_primary_v1.csv` (4,612 CVs, rubric labels) —
  and `corpus_merged_v1.csv` (10,317) — for the v2 exports.
- **Labels:** our rubric thresholds on `total_score`: `Weak < 50` /
  `Average 50–79` / `Strong ≥ 80`. Not human labels at the primary tier.
- **Features (v1 + v2):** `raw_text` → `TfidfVectorizer(ngram_range=(1,2), min_df=2,
  sublinear_tf=True)`. That's it. No engineered facets, no numeric features, no
  semantic vectors. (The `num`/`mxd` grid variants reached ~0.97–1.00 accuracy —
  because they fed the rubric sub-scores `score_*`, which are *the literal score
  inputs of the label*. That row is an oracle ceiling, not a model.)

Three measured failures, in order of importance:

1. **Train/test distribution mismatch.** Training text is *reconstructed* from
   structured datasetmaster JSON (`scripts/build_classifier_data.py`
   `reconstruct_resume()`: name/summary/experience/education/skills flat-lines).
   Benchmark + demo CVs are real prose (headers, bullets, paragraph breaks).
   TF-IDF unigram/bigram features that separate classes on reconstructed text do
   NOT transfer to organic CV prose. This is why every model — deployed xgb,
   v2-rf, v2-xgb — lands on *Average* for the Strong rows (01@81, 10@87) and Weak
   rows (04/06/08/09) of `demo/benchmark/_baseline.json`. Benchmark rubric
   agreement is **4/10** for all three.
2. **Sampling can't fix #1.** The Weak+Strong oversample GPU grid
   (`scripts/probe_balanced_oversample.py`) *confirmed* no `(weak, strong)`
   multiplier lifts benchmark agreement above 4/10 and raising Strong above ×1
   crushes held-out Strong recall (0.87 → 0.06). Classic: oversampling a proxy
   text distribution does not rescue OOD prose.
3. **Class imbalance at the label source.** Weak is 85/4,612 (~1.8%); semantic
   models have essentially nothing to learn from there.

## 2. What we deliberately did NOT do (from the survey)

| Method | Used? | Why not / why deferred |
|---|---|---|
| Engineered structured features (years, degree level, skill/acts counts, links) | **NO** | Required a real `extract_all` on *raw_text*, which the corpus build skipped. |
| Text embeddings + head (bge/mpnet → LR/SVM/MLP) | **NO** | Only the *matcher* uses `models/matcher-confit` embeddings. Were never used for classification. |
| Fine-tuned encoder (DeBERTa/BERT) | Probe only | DistilBERT probe: F1 0.880, Weak recall 0.0 → below the TF-IDF head on the exact class where the rubric matters. |
| LLM-as-judge (Claude/GPT/Qwen inference) | **NO** | GPU per-CV latency 27–32s and no exactness guarantee; the rubric already expresses the quality dimension precisely. |
| **Ordinal / regression-to-score formulation** | **NO** | All our heads are 3-class softmax, which treats "Weak vs Strong" and "Weak vs Average" *equally* — the textbook mistake the write-up calls out. |

## 2. The v3 plan — hybrid features + score regression (ordinal by construction)

The rubric we ship is already a **continuous score (0–100)** with the label as an
after-threshold. So the correct ML problem is **regression to `total_score`** —
which is *ordinal by construction* (pred Δ error weighted by how far off the score
is, not by class flip). We keep a lightweight 3-class read-out by thresholding the
prediction, matching the app display.

### Feature vector (concatenated)

1. **`matcher-confit` embedding of `raw_text`** — 384-d, `normalize_embeddings=True`.
   The same embedder the matcher ships (bge-small fine-tuned to `resume-job-desc`
   fit). Semantic transfer from prose to prose is the whole point here.
2. **Structured metadata** `build_features()` of a **real `extract_all` pass**
   (`src/scorer/feature_builder.py`, 12 features): `highest_degree_level`,
   `total_experience_years`, `skill_count`, `project_count`, `has_github_link`,
   `certification_count`, `language_count`, `leadership_count`, `achievement_count`,
   `avg_gpa`, `experience_entry_count`, `education_entry_count`.
3. **Organic-prose macro features** (cheap text-laws, no NER needed):
   `section_presence` (which of ~8 canonical headers are detected),
   `word_count`, `line_count`, `mean_sentence_len`, `date_count` (regex range
   scan of the raw text), `flesch_reading_ease`.

We do **NOT** add the rubric sub-scores (`score_*`) — that's oracle leakage and
defeats the point.

### Model head

- Stage A: `XGBoost` **regressor** (`reg:squarederror`) on the concatenated vector
  → predict `total_score` (0–100).
- Stage B: label = thresholds on the regressor output (Weak < 50 / Average < 72 /
  Strong else), preserving the existing app display contract. Regression is
  ordinal by construction — smaller absolute error is always preferred, unlike a
  3-class softmax that can't encode "Weak vs Strong is a bigger miss than Weak vs
  Average".
- Expose the **same app surface** as the deployed pipeline: `predict([text]) ->
  str label`, `predict_proba -> (len 3, sums to 1)`, and `predict_score`. Label
  order = `classes_ = label_classes_ = CLASSES`.

### Runtime behavior

- At **train time** the CVs are run through `extract_all` once and cached (fast).
  `matcher-confit` embeds in batch on GPU (RTX 5070 Ti).
- At **app time**, the classifier consumes the same extracted CVSchema the app
  already has for scoring — cost is one embedding call, matching current latency
  (single-digit ms on the embed; XGBoost is trivial).

## 3. Experiments that must pass before this is a "better" classifier

Measured on the SAME harness as before (parse each benchmark CV → model → label):

| Signal | Target | Current (deployed xgb) |
|---|---|---|
| Benchmark rubric agreement (10 CVs) | **≥ 6/10** | 4/10 |
| Held-out rubric agreement (holdout of corpus_primary) | ≥ deployed | 0.885 acc / 0.881 f1w |
| Held-out Pearson/Spearman `pred_score` vs `total_score` | ρ ≥ 0.55 | n/a (classifier has no score) |
| Weak recall (the class that matters) | ≥ 0.10 | 0.118 (held 0.1176) |

If the hybrid cannot beat those numbers on real prose, the conclusion is that a
bag-of-words + metadata classifier cannot predict rubric score from *reconstructed*
training text onto *real* CV prose — and any real fix requires collecting/annotating
real-text CVs as training data (a data task, not a modeling one).

## 4. Implementation artifacts

- `scripts/build_hybrid_classifier.py` — engineered + embedding feature cache → GPU
  XGBoost regressor → app-contract wrapper. Outputs:
  - `results/hybrid_features_primary.csv` (feature cache),
  - `results/hybrid_embeddings_primary.npz` (embedding cache),
  - `results/classifier_v3_hybrid.pkl`,
  - `results/classifier_v3_hybrid_eval.json` (held-out + benchmark agreement).
- `src/extractor/quality_features.py` — `text_macro_features(text) -> dict` (organic
  prose heuristics) + `engineered_features(cv) -> np.ndarray` (thin, CPU-safe wrapper
  over `feature_builder.build_features`).
- `tests/test_hybrid_classifier.py` — contract (predict/classes/predict_proba/
  predict_score) + macro-feature sane-range tests.

## 5. Deployed in the app (2026-08-08)

The synthetic-corpus model (`results/classifier_v3_hybrid_synth.pkl`,
`data/curated/corpus_synth_v1.csv`) was acceptance-tested then shipped:

- `scripts/acceptance_hybrid_classifier.py` — fresh-process load, contract attrs,
  edge inputs (empty/garbage/unicode/LLM-JSON/long), determinism, batch-vs-single,
  threshold boundaries (49.99/50.01, 71.99/72.01), demo CVs through real `parse_cv`.
  Exit 0.
- **predict_proba fix** — original Gaussian bucket centers [25,60,88] flip at 74/42.5,
  not the rubric 50/72 thresholds, so a score ≈73 was labeled Strong but proba argmax
  said Average. Rewrote as normal-CDF of the 50/72 thresholds (`scipy.special.erf`,
  vectorized), keeping argmax ≡ predict() for every score.
- Full suite 450 passing; classifier contract tests 14 passing.
- App: `load_classifier()` prefers `models/classifier_v3_hybrid_synth.pkl`
  ("Hybrid (v3 synth)") with `xgb_classifier.pkl` fallback. Warm per-CV classify
  ~120-180ms (embedder is the singleton `preload_matcher()` already warms).
- Rationale: 7/10 benchmark agreement + score-level Spearman +0.758 vs deployed
  XGBoost 4/10 / ~0.