# Extraction Logic — Full Improvement List

Consolidated from the 2026-08-03/04 audit (`docs/extraction_audit.md`) and the
two research threads (`docs/research_ner_hybrid_extraction.md`,
`docs/research_text_reformatting.md`). Every item maps to a failure mode found in
the demo set, the corpus, or the research evidence.

Legend: **[P1]** high-ROI / quick, **[P2]** medium effort, **[P3]** larger /
conditional, **[SKIP]** verified not worth it.

---

## A. Experience extraction

- **[P1] Company-on-next-line fallback — DONE** (`experience_extractor.py`)
  Title+date on one line, company on the next (table-flattened DOCX).
  Falls back to the line after the date range before using spacy ORG.
- **[P1] Title separator strip — DONE** `Web Developer -` → `Web Developer`.
- **[P1] Duplicated-block truncation — DONE** merged-cell DOCX tables emit
  each cell twice; truncate at the repeated date range.
- **[P1] Multi-entry experience (root cause) — DONE** DOCX paragraphs join
  with a single `\n` (no blank lines), so old blank-line splitting collapsed
  2 jobs → 1 entry. Refactored `_parse_experience_text` to a whole-section
  date-anchored stream with `_find_all_dates()` (overlap dedup) and
  `_looks_like_job_header()` bullet-line skip. Entry counts now correct in the
  benchmark (01:2, 03:2, 05:3, 06:3, 08:3).
- **[P1] "at" false split — DONE** `_parse_title_company` tries comma/pipe
  split before the `at/@|–|—|-` regex → "University of Texas at Austin" keeps
  full company.
- **[P1] Date-first format** (srbhr barry): `June 2022 – Software Engineer
  (Front-End), Google, Mountain View, CA, USA Present`. Dates come BEFORE the
  title/company. Partially covered by the whole-section refactor (benchmark
  06 date-first now yields all 3 titled entries); still verify srbhr.
- **[P1] ORG false positives in company** (Rebecca "Canvas", srbhr
  "Front-End"): spacy picks short/partial ORG spans. Validate against the
  line: company lines usually contain a location word or a company suffix
  (`.inc`, `.ltd`, "Corp"). Drop the ORG fallback if the span lacks both and
  the surrounding lines contradict it.
- **[P2] Duration sanity / location leakage** (known issue #1): strip
  city/state/country words from parsed date strings before `compute_months`.
- **[P2] `Present`/`Current` glued to company line** (srbhr: "USA Present"):
  split "Present" off the end of the leading line before title/company parse.
- **[P2] Experience dedup beyond start-date key**: two different roles can
  share a start month; consider a signature of `(start, company, title)`.

## B. Education extraction

- **[P1] Multi-degree paragraph split** (template: `PH.D.` / `MBA` / `BBA` in
  one blank-line-free paragraph). Split on degree-keyword lines rather than
  blank lines only.
- **[P1] Degree ↔ institution pairing**: currently one degree keeps the whole
  paragraph's first NER ORG; "Thesis:"/"Dissertation:" lines must be excluded
  from institution (template got "Supply Chain Management…" as institution).
- **[P1] Institution keyword preference over NER ORG**: when an ORG span
  lacks an institution keyword (`University|College|Institute|School…`),
  prefer the keyword-line match.
- **[P2] Multi-line institution abbreviation** (Rebecca: "University of
  Illinois at Urbana-Champaign … Expected May 2024" glued on one line).

## C. Name / contact

- **[P1] Name from stacked lines** (template: `Denice` / `Harris` /
  `ASSOCIATE PROFESSOR`). First+last on separate lines with a title line
  below → combine.
- **[P1] Reject section-heading-like names** (known issue #5: template
  sometimes returns "Email"). Add a name-confidence heuristic: if the
  candidate equals a section heading or a common word, retry with the first
  line before any heading.

## D. Projects

- **[P1] Project title extraction** (resume_03, senior, srbhr all return
  `title=None`): take the first line of each project block as the title; fall
  back to first tool noun phrase.

## E. Section splitting

- **[P1] Academic/other headings** (template): `Invited Talks`, `Conferences`
  currently bleed into experience. Map to `other` or `achievements`; add
  `publications` variants already partially present.
- **[P2] Heading re-detection from formatting**: PyMuPDF span bold/size flags
  (PDF) + python-docx run bold (DOCX) to restore heading boundaries lost to
  layout flattening (research thread 2, Parsr-style).
- **[P2] Cue-phrase unification** (TSHD): "Academic Background" ≈ "Education"
  etc. — largely covered by `SECTION_ALIASES`; extend the alias table.

## F. Parser-level normalization (research thread 2)

- **[P2] DOCX tables → pipe-joined lines in document order** via
  `iter_block_items()` (CT_P + CT_Tbl), handling merged/omitted cells with
  `grid_cols_before/after`. Prevents the one-entity-per-line / duplicated
  content classes at the source.
- **[P2] PDF reading order**: PyMuPDF `get_text("text"|"blocks"|"dict",
  sort=True)` or pymupdf4llm for true reading order + multi-column + table
  detection. Two-column layouts are the #1 ATS breaker (38% of resumes, −14
  pts in 2026 data).
- **[P2] Conservative text cleanup layer**, config-driven, structure-
  preserving: NFKC, unicode-space collapse, `•`/`|`/em-dash standardization,
  `(cid:NNN)` removal, header/footer duplication suppression. **Must not**
  strip punctuation/stopwords — that hurt rule extraction in the literature.
- **[P3] Two-column reading-order reconstruction** (XY-cut-lite / half-crop)
  — only if the corpus scan shows two-column PDFs actually appear.

## G. NER usage (research thread 1)

- **[P1] NER spans as candidates, rules validate** — keep `ner-v1`, use it to
  propose title/company/degree/institution spans and let the rule layer
  accept/reject (currently only skills are fused via `ner_tag.py`).
- **[P2] Per-class span repair**: title/company, degree/institution,
  date-range repair using validated spans.
- **[SKIP] Model swap** to `oksomu/resume-ner` / `yashpwr/...`: verified dead
  end (~54 mean, 68–140ms), `scripts/gate_external_ner.py`.
- **[SKIP] LLM hybrid in the app**: ~1 min/CV on CPU; kept offline in
  `src/extractor/hybrid.py`.

## H. Scorer / feature consistency

- **[P1] Rubric degree-key gap — DONE** extractor emits `M.Sc`/`M.Tech`/`M.A`/
  `M.E` (and `B.A`/`B.E`) but `rubric_config.json` `degree_points` only had
  `MSc`/`MS`/`B.Sc` etc. — master's degrees scored 0 for education. Added the
  missing keys (config-driven). Verify the corpus doesn't shift labels.
- **[P2] feature_builder gap-smoothing masks extraction errors**:
  `_total_experience_years` uses 0.67 weight over gaps, so the ML classifier
  can't see the same extraction defects the score does. Consider emitting an
  `extraction_conf` signal or fixing duplicates before feature building.

## I. Benchmark / testing

- **[P1] Controlled demo corpus — DONE** — `demo/benchmark/` with one CV per
  failure scenario (generated reproducibly by `scripts/generate_benchmark_cvs.py`),
  a manifest (`manifest.json`), and baseline scores (`_baseline.json`).
  Baseline mean 46.4 → 53.6 after the fix batches. Re-run after every
  extraction change and keep `_baseline.json` updated.

## Explicitly NOT worth it

- Generic "reformat any broken CV" engine (open-ended, low payoff).
- Dehyphenation / line-joining (low payoff for CVs).
- LayoutParser / Docling / DocTR / LayoutLM (break the 1–2 s CPU budget).
- Generative small-LLM extraction (10–60 s/CV).
