# JD Matching — Datasets, Methods & Latency Research

**Date:** 2026-08-04
**Status:** research + findings; implementation backlog in `TODO.md`

---

## 1. Current latency (measured, bge-small embedder)

| Stage | Time |
|---|---|
| First call (model load, one-time cold start) | **~10.4 s** |
| Subsequent matches (model warm) | **~0.061 s per CV–JD pair** |
| CV processing (extract+score, reference) | ~0.5–2 s per CV |

**Conclusion:** after the embedder is warm (the app already holds it via
`@st.cache_resource`), JD matching is **faster than CV processing** — well inside the
"similar to CV processing time" budget. The only cost is a one-time ~10 s cold start
on the first match, which is a warm-up/UX concern, not a per-match concern.

Cold-start mitigation options (see §4): lazy warm-up on app start, smaller/faster model,
or a lightweight lexical fallback until the embedder is ready.

---

## 2. Scoring datasets we have (CV↔job matching)

| Dataset | n | Has job text | Score source | Used for |
|---|---|---|---|---|
| **ATS** (`ats_scores_clean`) | 5043 | ✅ resume `SEP` job | human ordinal label + jina-v2 similarity | matcher eval, embedder choice, weight fitting |
| **NETSOL** (`netsol_clean`) | 849 | ✅ `job_description` | numeric score (0–10), generated | classifier build; **candidate for matcher cross-check** |
| `labeled_cvs` | 4500 | ❌ | rubric quality label (Strong/Average/Weak) | classifier (CV quality, NOT matching) |
| `classification_clean` | 12078 | ❌ | category | classifier |
| `borderline_review` | 1382 | ❌ | rubric + reviewer | classifier edge cases |

**Key point:** matching requires a *job*; only ATS and NETSOL have job text. The other
datasets score CV *quality* and cannot train the matcher.

**Backlog — NETSOL cross-check:** run the embedder + learned-weight test on NETSOL
(849 real scored candidate↔job pairs) as an independent validation that the bge-small
advantage isn't an ATS-only fluke.

---

## 3. Research notes — external datasets & methods

(_from web research 2026-08-04_)

### Candidate external datasets for JD matching training

| Dataset | Size | Labels | Notes |
|---|---|---|---|
| **`cnamuangtoun/resume-job-description-fit`** (HF, MIT) | 6,241 resume–JD pairs | fit labels (binary/multi) | The standard HF resume–JD matching set; used by LoRA fine-tunes (e.g. `shashu2325/resume-job-matcher-lora` on bge-large) and an XGBoost baseline (78% acc, 89% AUC). ⭐ top candidate |
| **`bwbayu/job_cv_supervised`** (HF) | — | supervised pairs | resume↔JD supervised pairs; used by community matchers |
| **`0xnbk/resume-ats-score-v1-en`** (HF) | 5,099 (already have) | human ordinal + jina sim | our current matcher eval |
| `darysha/hse-hackathon` (Kaggle) | — | Invitation/Rejection (real outcome) | authentic job-market outcome signal; Russian text |
| `datasetmaster/resumes` (HF, MIT) | 4,817 | none (CVs only) | real+synthetic CVs, no JD — useful for CV corpus, not labels |
| `facehuggerapoorv/resume-jd-match` (HF) | — | match labels | small community set |
| `pranavvenugo/resume-and-job-description` (Kaggle) | 2,482 | none | resumes + JDs, no outcome labels |

**Licensing/PII caveat:** most resume corpora contain PII; the ConFit papers and
community code use them for research. Prefer MIT-labeled pairs (`resume-job-description-fit`)
for anything shipped; synthetic/anonymized for demos.

### Methods to consider (ranked by value-to-effort)

1. **ConFit / ConFit v2** (ACL 2025 / RecSys 2024, arXiv:2401.16349, 2502.12361) —
   the current state of the art for sparse resume–JD labels. Key idea: label sparsity is
   <0.05% of possible pairs, so they (a) **augment** data by paraphrasing CV/JD sections,
   (b) train a **contrastive (Siamese) encoder** so matching pairs are pulled together and
   random/near-miss pairs pushed apart, and (c) mine **hard negatives** ("runner-up" pairs).
   v2 adds LLM-generated "hypothetical reference resumes" + Runner-Up Mining (RUM). Reports
   ~13% absolute gain over BM25 / OpenAI embeddings. This is the "right" long-term fix and
   fits our existing sentence-transformer setup.
2. **conSultantBERT** (arXiv:2109.06501) — fine-tuned Siamese SBERT on **270k**
   consultant-labeled resume–vacancy pairs. Confirms contrastive fine-tune >> TF-IDF + BERT
   embeddings. Labels are proprietary, but the *method* is the same as ConFit.
3. **Contrastive fine-tune on our own data** — cheap: use `resume-job-description-fit`
   (+ optionally ATS) to fine-tune `bge-small` with a cosine/InfoNCE loss, then swap it into
   our embedder. No new infra (sentence-transformers supports `SentenceTransformer.train`).
4. **XGBoost/LambdaMART learning-to-rank** (we already use XGBoost for the classifier) —
   `rank:ndcg` objective over features (semantic, skill-overlap, rubric, TF-IDF). Needs
   query-grouped (qid) data; our ATS/`resume-job-description-fit` pairs + a JD group key fit
   this. A natural upgrade from hand-set 0.5/0.3/0.2 weights.
5. **BM25 hybrid (backlog E)** — cheap lexical pre-filter + top-K semantic re-rank; the
   ConFit v2 baseline includes BM25, so this is a *baseline to beat*, not the end goal.

### What NOT to do
- Do **not** train directly on `ats_score` (jina-derived) as ground truth — it is another
  model's output; we already use it only as a reference. Use the human ordinal labels or a
  human-labeled set like `resume-job-description-fit`.
- Do **not** chase huge embedders (bge-base was slower *and* worse on our data); if we
  fine-tune, fine-tune the small model.

---

## 4. Latency budget vs CV processing

**Measured (bge-small): first call ~10.4 s (model load), subsequent matches ~0.061 s/pair.**
CV processing is ~0.5–2 s. So JD matching **after warm-up is faster than CV processing** —
the budget is met. The only issue is the one-time cold start.

### Cold-start mitigations (in priority order)
1. **Eager warm-up at app start** — preload the embedder in the app's init (not on first
   match). Streamlit `@st.cache_resource` already caches it for the session.
2. **Batch embed candidate CVs once** — embed all CVs in the candidate pool a single time,
   cache by `cv_id`; matching N CVs then costs only 1 JD embed + N dot products (~microseconds
   each). This makes pool re-ranking with a *different* JD essentially free.
3. **BM25 pre-filter (backlog E)** — for very large pools, score cheap BM25 first, embed
   only top-K. Protects throughput, not the first-match cold start.
4. **ONNX/int8 quantization** — if cold start or throughput ever matters more, quantize the
   embedder (faster load + inference). Not needed yet.

### Keeping it fast while improving accuracy
- Fine-tuning (ConFit-style) improves the *embedding*, not the latency — still one encode per
  doc. So accuracy upgrades don't cost speed.
- Avoid per-pair model re-loading; keep the embedder cached for process lifetime.

---

## 5. Open questions / next steps

- [x] Measure matching latency (done: cold ~10.4 s, warm ~0.061 s/pair)
- [x] Research external datasets + methods (done, see §3)
- [x] Implement E as cheap pre-filter / baseline (done 2026-08-05: `bm25_scorer.py`,
      opt-in 4th ranker signal, default weight 0.0; `score_corpus()` for pool pre-filter)
- [x] ConFit-style contrastive fine-tune of bge-small (done 2026-08-05:
      `scripts/train_matcher_confit.py` on the 6,241 `resume-job-description-fit` pairs →
      `models/matcher-confit`. Held-out binary-fit ρ 0.216→0.332; our ATS ρ 0.314→0.436;
      NDCG@5 0.985 no regression. **Adopted as the default embedder**; `CV_EMBEDDER` overrides.)
- [x] NETSOL cross-check of bge-small + learned weights (done 2026-08-05:
      `scripts/eval_netsol_crosscheck.py` → `models/netsol_crosscheck.json`. 849 pairs:
      matcher-confit pure-semantic ρ 0.345 vs base 0.329; learned 1.0 blend beats hand 0.5.
      Confirms the advantage is not ATS-only.)
- [x] Evaluate `cnamuangtoun/resume-job-description-fit` (6,241 pairs) as a second,
      human-labeled matching set (done 2026-08-05: `scripts/eval_resume_jd_fit.py` →
      `models/resume_jd_fit_eval.json`. Test 1,759 pairs: binary-fit ρ matcher-confit
      0.332 vs base 0.216; retrieval NDCG@10 0.309 vs 0.296)
- [x] Add app-start eager warm-up of the embedder (done 2026-08-05: `warm_up()` in
      `embedder.py` + `preload_matcher()` cache_resource in `app/app.py`; loads at app
      start, cached for the session)
- [x] Settled BM25 hybrid default (done 2026-08-05: `scripts/eval_bm25_hybrid.py`, test
      1,759 pairs: any `bm25` weight lowers binary-fit ρ AND NDCG@10 (ρ 0.332→0.265, cut
      NDCG@10 0.309→0.213). Kept default 0.0; BM25 stays opt-in pool pre-filter.)
- [x] Learning-to-rank probed (done 2026-08-05: `scripts/train_ranker_ltr.py`, XGBoost
      `rank:ndcg`, 6 feats [semantic, skill, bm25, token-iou, len_cv, len_jd], qid=JD).
      Test NDCG@10: hand blend 0.7401, pure semantic (ConFit) 0.7805, LTR 0.6816 (early
      stop at round 1). Negative — lexical/auxiliary feats dilute the ConFit signal. Not
      adopted; pure semantic stays the ranker. Feature embeddings cache to
      `models/ranker_ltr_emb_{split}/` for any rerun.
