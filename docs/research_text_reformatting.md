# Normalizing raw CV text before extraction — research report (2021–2026)

## TL;DR verdict

**Worth it — but only as a bounded, two-layer step, not a "reformat any broken CV" project.**

The damage from layout-broken CVs is almost always structural, not lexical — scrambled reading order, table cells flattened into one line, headings merged into body text. Those are exactly the things a conservative, geometry-aware normalization pass can fix cheaply at parse time (coordinates still exist), and that you cannot fix later with text-only regex once the layout is flattened.

Aggressive "clean everything" pipelines are where the trap is: at least one published experiment found heavy preprocessing hurt rule-based resume extraction (it deleted useful signal). The high-ROI play is: fix reading order + DOCX tables at the parser level, then run a light, structure-preserving text cleanup before your section splitter — and keep aggressive cleaning out.

Bottom line for cvinsight: a ~1–2 week, LLM-free "layout → linear text + light normalization" layer is high-ROI and comfortably fits your 1–2s CPU budget (PyMuPDF/pymupdf4llm-style C-engine sorting is milliseconds/page). A generic "repair arbitrary broken text" engine is an open-ended project with poor payoff for single-user uploads — don't start it.

---

## 1. Why "clean the text then extract" is the right framing

Our splitter and per-line parsers depend on newlines and headings — and the failures (columns flattened, tables one-line, headings merged, bullets glued) are precisely the ones measured as the top resume-parsing breakers:

- ATSChecker 2,417-resume study: two-column layouts in 38% of resumes with 31% parsing-failure rate (−14 pts); table-based layouts in 24.6% (27.8% failure, −11 pts).
- Controlled one-resume/six-layout ATS benchmark: two-column was the only layout that broke (100 → 85, "reading-order scramble").
- 36-template × 4-parser diff: losses clustered where reading order matters — work-history segmentation and skills-section parsing.
- pdfplumber/pypdf fixture benchmark: two-column returned ~93–100% of fields but section order was not preserved; raster-image PDF returned 0%.

Note the nuance: it isn't columns that break — it's the ORDER in which text runs are emitted. If the extractor emits the sidebar interleaved with the main column, you get exactly the "19/100 vs 61 with LLM" symptom. That's a parser-side (geometry still available) problem, so the normalization layer belongs BEFORE the section splitter, versus patching each per-line regex.

## 2. Three layers of "text normalization"

### Layer A — Text-level cleanup (cheap, config, safe)
- Unicode/whitespace: `unicodedata.normalize("NFKC", …)`, collapse Unicode spaces, multi-space runs.
- Standardize separators (`|`, `•`, `·`, `‑`, em-dash → consistent), strip decorative glyphs/bullet junk.
- Strip `(cid:NNN)` markers, smart-quote/ligature expansion.
- Make it config-driven and structure-preserving (oksomu/resume-ner `pre_processing` block is a model).
- Caution: must NOT delete content. Aggressive preprocessing (stopwords, lemmatization, punctuation stripping) made rule-based resume extraction WORSE.

### Layer B — Layout → linear text (the high-ROI layer)
Converting two-column PDFs, tables, and heading-style loss into clean, newline-corrected, reading-order linear text. Where the benchmarks say points are lost.

### Layer C — Line repair / de-noising (can become open-ended)
Dehyphenation, joining wrapped lines, OCR-garbage removal, header/footer stripping, paragraph re-flow. Some sub-items cheap (page-number/header/footer, `(cid:)`); full paragraph reconstruction is research-scale. Skip the open-ended part.

## 3. Techniques + libraries (prioritized)

### P0 — Parser-level reading order (hours–1 day, biggest bang)
- PyMuPDF: `page.get_text("text", sort=True)` / `blocks sort=True` / `dict sort=True` — vertical-then-horizontal sort; span font flags (bit 4 = bold) + font size for heading re-detection.
- pdfplumber/pdfminer.six: `extract_words()` left-to-right/top-to-bottom but columns still interleave — crop per column or cluster on x-positions. `LAParams(boxes_flow=…, line_margin=…, char_margin=…)` is the documented lever for column-merge problems.
- pymupdf4llm (CPU-only, C engine): reading-order reconstruction, multi-column, table→markdown, heading-level detection, header/footer removal.

### P0 — DOCX tables → logical lines (half-day)
python-docx `iter_block_items()` over `document.element.body` yields `CT_P` and `CT_Tbl` in order. Emit each row as a pipe-joined line; use `row.grid_cols_before/after` + cell paragraphs for merged/omitted cells. (RAGFlow resume module emits DOCX table rows as `" | ".join(cells)`.)

### P0 — `(cid:NNN)` and glyph garbage (an hour)
- Best: subclass/patch `PDFLayoutAnalyzer.handle_undefined_char` → return `""`.
- Rebuild CID→Unicode map from `font.unicode_map.cid2unichr`.
- Regex `(cid:(\d+))` → `chr(n)` is only correct for ASCII-mapped fonts — don't trust generally.

### P1 — Two-column reading-order reconstruction (a few days)
XY-Cut (Meunier, ICDAR 2005) / XY-Cut++ (2025, 98.8 BLEU, ~500 FPS CPU). For CVs (usually just two side-by-side zones): cluster word `x0`s to find the gutter, crop left/right halves, extract each, concatenate.

### P1 — Heading re-detection and reinsertion (a few days)
- Format heuristics (PDF/DOCX still have it): bold flag / font size / all-caps / underline → "heading-shaped" lines (Parsr titleScores).
- Lexical cue-phrase matching: ~96% F1 sectioning resumes by cue-phrase + cue-word scans; XGBoost on line features baseline 90.1% F1 heading detection.

### P2 — Dehyphenation / line-joining (only if data shows it matters)
Dictionary/self-document heuristic is usually enough (join only if merged word is known or appears elsewhere in doc).

### ❌ Avoid (CPU/1–2s budget)
LayoutParser (Detectron2), Docling (RT-DETR + TableFormer, ~633ms/page + ~1.7s/table CPU), DocTR, LayoutLM/DocLLM. All break the latency budget. Docling's RULE-BASED ReadingOrderPredictor idea (R-tree + overlap heuristics) can be stolen without models.

## 4. ROI evidence: preprocessing vs robustifying extractor

FOR a normalization pass:
- Alibaba SmartResume (2025): layout regeneration ("re-order text from complex multi-column layouts into a single, indexed sequence") is a MANDATORY stage before extraction; ~20% of real resumes have non-linear multi-column layouts.
- oksomu/resume-ner: 99.2% micro-F1 clean vs 69.2% noisy — same model, noise = broken-text category.
- Hierarchical resume parsing assumes "section and group boundaries are always placed at the end of a line" — depends on the line structure normalization must preserve/restore.
- Cheap geometric sorters (XY-Cut++) beat expensive learned ones on throughput near state-of-the-art accuracy.

AGAINST / boundary conditions:
- Aggressive cleaning hurt rule-based extraction (destroyed signal rules depended on).
- ResumeBench (EMNLP 2025): even GPT-4o degrades on asymmetric two-column/hybrid layouts; no text-level processing fully recovers a flattened layout.
- Skills grids and punctuation are basically fine in 2026 parsers; real culprits are reading order, table-emission order, headers/footers.

Synthesis: normalize-at-the-source (geometry) is cheap and prevents the loss; robustify-the-extractor is the fallback for already-done damage. Ratio heavily weighted to parsing/normalization.

## 5. Effort vs payoff for cvinsight

| Action | Effort | Payoff | Verdict |
|---|---|---|---|
| Unicode/text cleanup (NFKC, spaces, separators, bullets, `(cid:)`, ligatures) — config-driven, structure-preserving | ~half a day + tests | High | Do it first |
| Parser-level reading order (PyMuPDF `sort=True`/dict, or pymupdf4llm) | 0.5–2 days | Very high | Do it first |
| DOCX tables → pipe-joined lines in document order | 0.5 day | High | Do it |
| Two-column reading-order reconstruction (XY-cut-lite / half-crop) | 2–4 days | High (conditional on two-column PDFs in uploads) | Conditional do |
| Heading re-detection (bold/size from spans + cue-phrase) | 2–4 days | High | Do it |
| Dehyphenation / line-joining | 2–5 days | Low–medium | Skip / dictionary-join only |
| Generic "reformat any broken CV" | weeks+ | Low | Skip |
| LayoutParser / Docling / DocTR / LayoutLM | days-weeks + hosting | High accuracy but breaks budget | Skip |

Verdict: A bounded, LLM-free normalization layer is clearly worth it — (a) parser-level reading order + DOCX table linearization, (b) conservative text cleanup, (c) heading re-detection. ~1–2 weeks incl. tests, fits 1–2s CPU budget. "Clean text then extract" only underperforms "fix extractor" when cleaning is destructive or balloons into a generic repair engine. Keep it source-format-aware, structure-preserving (never destroy line/heading boundaries before the splitter), config-driven.

## 6. Key sources

- PyMuPDF reading-order, spans, bold flags — https://pymupdf.readthedocs.io/en/latest/recipes-text.html · FAQ · PR #3878
- pdfplumber two-column guidance — https://github.com/jsvine/pdfplumber/discussions/885
- pdfminer LAParams column-merge — https://github.com/pdfminer/pdfminer.six/issues/276
- pymupdf4llm — https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
- XY-Cut++ — https://arxiv.org/abs/2504.10258 · impl in MinerU
- Docling (rule-based reading order; CPU timing) — https://arxiv.org/abs/2501.17887
- LayoutParser — https://arxiv.org/abs/2103.15348
- python-docx iter_block_items — https://github.com/python-openxml/python-docx/issues/276
- RAGFlow resume module — https://github.com/infiniflow/ragflow/blob/main/rag/app/resume.py
- Parsr heading heuristics — https://github.com/axa-group/Parsr
- TSHD cue-phrase segmentation — https://onlinelibrary.wiley.com/doi/10.1155/2023/6044007
- pdfminer (cid:) issues — https://github.com/pdfminer/pdfminer.six/issues/1056 · #746
- Dehyphenation thesis — https://ad-publications.informatik.uni-freiburg.de/theses/Bachelor_Mari_Hernaes_2019.pdf
- oksomu/resume-ner pre_processing + noise gap — https://huggingface.co/oksomu/resume-ner
- Alibaba SmartResume — https://arxiv.org/abs/2510.09722
- Counter-evidence (aggressive cleaning hurts) — https://github.com/JennyTan5522/NLP-Resume-Parsing
- ResumeBench — https://aclanthology.org/2025.emnlp-main.1626.pdf
- ATSChecker 2,417-resume study — https://www.atschecker.ai/research/ats-resume-study-2026
- ATS Verification 6-layout benchmark — https://atsverification.com/blog/ats-parsing-benchmark-2026/
- 36-layout × 4-parser diff (emission-order insight) — https://dev.to/resumap/we-ran-the-same-resume-through-4-real-ats-parsers-in-36-layouts-same-text-different-parses-3mfc
- sweresume fixture benchmark — https://www.sweresume.app/research/resume-format-benchmark/
- Enhancv reading-order overview — https://enhancv.com/blog/ats-resume-parsing/
- pdfmux column-detection — https://pdfmux.com/blog/multi-column-pdf-extraction-python/
