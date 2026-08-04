# Extraction Audit — findings, fixes & prioritized improvements

Status: extracted from the 2026-08-03/04 audit pass. Supports the two research
threads (`docs/research_ner_hybrid_extraction.md`,
`docs/research_text_reformatting.md`).

## 1. Current state

### Original demo set (demo/, 10 real-world CVs)

| File | Score | Label | Notable gaps |
|---|---|---|---|
| resume_04_vikram_singh.pdf | 68 | Average | none significant |
| ocrtest.pdf | 68 | Average | none significant |
| senior_python_dev.txt | 66 | Average | — |
| resume_02_rahul_verma.pdf | 64 | Average | — |
| srbhr_repo_barry_allen_fe.pdf | 58 | Average | exp title empty, company="Front-End" (ORG FP) |
| resume_03_ananya_patel.pdf | 54 | Average | proj=8 (titles missing) |
| Rebecca_Software or Computational Roles.docx | 53 | Average | exp company="Canvas" (ORG FP), proj=0 |
| junior_dev.txt | 48 | Weak | genuinely sparse CV (1 internship) |
| priya_dwivedi_repo_MathewElliot.docx | 40 | Weak | duplicated DOCX content; now fixed |
| pro-cv-template-burgundy.docx | 19 | Weak | placeholder dates "20XX" → exp=0 |

**Mean 53.8** (rule-based), stable after the multi-entry fix (below).

### Controlled benchmark (demo/benchmark/, 10 scenario CVs)

Generated reproducibly by `scripts/generate_benchmark_cvs.py` (manifest +
baseline in `demo/benchmark/`). One CV per failure scenario. Scores after the
second fix batch:

| File | Score | Label | Entry count | Notes |
|---|---|---|---|---|
| 01_clean_standard.docx | 73 | Strong | exp=2 | control; edu was 0 until rubric degree-key fix |
| 02_two_column.pdf | 55 | Average | exp=3 | two-column layout |
| 03_table_docx.docx | 60 | Average | exp=2 | table-based DOCX |
| 04_duplicated_cells.docx | 40 | Weak | exp=1 | merged-cell duplication; skills=7 still low |
| 05_academic.docx | 56 | Average | exp=3 | academic headings |
| 06_date_first.pdf | 44 | Weak | exp=3 | date-first format; titles now populated |
| 07_multidegree.docx | 53 | Average | exp=1 | multi-degree paragraph |
| 08_org_false_positive.docx | 48 | Weak | exp=3 | company now "University of Texas at Austin" (fixed) |
| 09_sparse_entry.txt | 32 | Weak | exp=0 | genuinely thin profile |
| 10_strong_senior.txt | 75 | Strong | exp=3 | strong senior |

**Mean 53.6** (was 46.4 baseline before fixes).

### Root causes of the remaining spread
- **placeholder data** (template "20XX" dates) — not fixable by extraction;
  the CV genuinely has no real dates.
- **genuinely sparse CVs** (junior_dev, 09_sparse) — correct low score.
- **ORG false positives** (Rebecca "Canvas", srbhr "Front-End") — spacy NER
  picking up short ORG spans; needs span-repair heuristics (see §4).
- **layout formats** (02 two-column, 06 date-first) — reading-order / format
  normalization is parser-level work (research thread 2).

## 2. Corpus scan (data/processed/labeled_cvs.csv, 4500 CVs)

Percentiles of total_score: min 38, p1 48, p5 53, median 67, p95 78, max 80.

Bottom 12 CVs (38–44) are all consistent synthetic resumes:
- `score_experience` 2–9 (short/absent work history — content, not extraction)
- `score_skills` 12–16, `score_certifications` 0, `score_languages` 2
- No profile shows an extraction bug; lows are genuinely thin profiles.

Insight (from feature_builder diagnosis): `_total_experience_years` smooths
gaps with a 0.67 weight, so duplicate/broken positions look near-identical in
ML features — the classifier can't see extraction errors the score does.

## 3. Deterministic fixes applied (2026-08-04)

### Batch 1 — `src/extractor/experience_extractor.py`
1. **Company-on-next-line fallback** — when the leading text before the date
   range only yields a title, the company often sits on the line right after
   the date (table-flattened DOCX): `Web Developer - 09/2015…` then
   `Luna Web Design, New York`. Added a fallback that checks the first line
   after the date with `_looks_like_company()` before the ORG-entity fallback.
2. **Trailing separator strip on titles** — `Web Developer -` → `Web Developer`.
3. **Duplicated-block truncation** — DOCX templates emit each table cell twice;
   if the same date range reappears in the paragraph, keep only the first
   occurrence (steps back to the line boundary).

Result: priya_dwivedi extraction clean (title `Web Developer`, company
`Luna Web Design, New York`, deduped 467-char description).

### Batch 2 — multi-entry experience (root cause)
**Finding:** DOCX paragraphs are joined with a single `\n` (no blank lines), so
the old `\n\s*\n` paragraph splitting collapsed multi-job experience to **1
entry** (date ranges were found, only the first used). Reproduced in the
benchmark (01_clean had 2 jobs but exp=1).

**Fix:** refactored `_parse_experience_text` to process the whole section as
one date-anchored stream:
- `_find_all_dates()` — collects every date range; overlapping-range dedup
  (`_DATE_RANGE_RE` and `_YYYY_RANGE_RE` both match "Jan 2021 - Present").
- segments between dates become entries; description lines are skipped via
  `_looks_like_job_header()`.

Result: entry counts now correct (01:2, 03:2, 05:3, 06:3, 08:3); experience
scores 18 → 25 on affected files.

### Batch 3 — cross-cutting
4. **"at" false split** — `_parse_title_company` now tries comma/pipe split
   **before** the `at/@|–|—|-` regex → "Teacher's Assistant, University of
   Texas at Austin" keeps the full company (was "Austin").
5. **Rubric degree-key gap** — the extractor emits `M.Sc`/`M.Tech`/`M.A`/`M.E`
   (also `B.A`/`B.E`) but `config/rubric_config.json` only had `MSc`/`MS`/
   `B.Sc` etc. Master's degrees scored **0** for education. Added the missing
   keys to `degree_points`.

**361 tests passing** throughout; original demo mean unchanged at 53.8;
benchmark mean 46.4 → 53.6. Follow-on hardenings (2026-08-04): benchmark mean
**56.4**, demo mean **56.2**; suite at **380 tests** after Schema v2
(criteria_scores + rationales) landed — benchmark 56.4 / demo 56.2 unchanged.

## 4. Prioritized improvements (effort → payoff)

### Tier 1 — cheap, generalizable (do next)
- **Span repair for experience title/company** (Rebecca "Canvas", srbhr
  "Front-End"): a small rule layer that validates/replaces ORG-fallback spans
  against the surrounding lines (company lines contain location words or
  `.inc/.ltd`; titles match `_JOB_TITLE_WORDS`). Prevents low-precision spacy
  spans leaking into `company`.
- **Project title extraction** (resume_03, senior, srbhr all return
  `title=None`): projects score only counts entries, but suggestions/display
  degrade; extract a headline from the first project line.
- **Section aliases for academic headings** (`teaching experience`, `research
  experience` already mapped; add `invited talks`, `conferences`,
  `publications` → achievements). Template file's `Invited Talks`/`Conferences`
  currently bleed into experience.
- **04_duplicated_cells skills=7** — the merged-cell dedup should also recover
  the skills section; investigate why the skills scan under-counts.

### Tier 2 — parser-level normalization (research thread 2)
Follow `docs/research_text_reformatting.md` P0 items:
- Unicode/text cleanup (NFKC, separators, `(cid:)`) — config-driven, structure-preserving.
- Parser reading order: PyMuPDF `sort=True` / blocks / dict, or pymupdf4llm.
- DOCX tables → pipe-joined lines in document order (`iter_block_items`).

### Tier 3 — NER (research thread 1)
- Keep `ner-v1`; encoder token-NER is the only CPU-real-time family.
- Use NER spans as **candidates that rules validate**, not replacements
  (span-NER can't do relational fields: durations, degree↔institution pairing).
- Consider per-class span repair: title/company, degree/institution,
  date-range repair.
- A model swap to `oksomu/resume-ner` / `yashpwr/...` is a dead end (~54 mean,
  68–140 ms) — verified in `scripts/gate_external_ner.py`.

## 5. Not worth it (verified or high-risk)
- **LLM hybrid in the app** — 1 min/CV on CPU; kept offline in
  `src/extractor/hybrid.py` (device toggle).
- **Generic "reformat any broken CV" engine** — open-ended, poor payoff.
- **Heavy text preprocessing before extraction** — destroys rule signal.
- **Two-column XY-cut** — only if two-column PDFs actually appear in uploads
  (check the 4500-CV corpus first).
- **Dehyphenation / line-joining** — low payoff for CVs.
