# Efficient NER for Resume / Semi-Structured Document Extraction
### Research review 2021–2026, focused on what cvinsight can add without an LLM

**Context this answers:** cvinsight runs a rule-based extractor (spaCy `EntityRuler` + `PhraseMatcher`) in production and has a small fine-tuned distilbert resume-NER tagger (`models/ner-v1`). Fusing the tagger's SKILL spans into the rules gained ~+1 pt because the gazetteer already saturates skills. The real score gap is *relational*: experience start/end dates → duration, degree↔institution pairing, title↔company pairing. Span NER does not produce relations. This report reviews techniques (with sources) that make NER extraction more effective on such documents, then gives a prioritized, LLM-free action list.

---

## 1. Label schemes: BIO vs BIOES, nested/discontinuous, and date/duration labels

### BIO vs BIOES
- Standard token-classification heads use BIO (IOB2). BIOES/BILOU adds explicit `E`/`S` tags. Evidence is that BIOES helps modestly and mostly on **boundary-sensitive / long-span** evaluation; it is not a free lunch (larger output vocabulary, more label confusion).
  - Study of 7 schemes (IO, IOB, IOE, IOBES, BI, IE, BIES) across 5 classifiers, NER F1: scheme choice is dataset-dependent. https://www.sciencedirect.com/science/article/pii/S1110866520301596
  - Detailed practical write-up of BIO/BIOES/span-based trade-offs. https://mbrenndoerfer.com/writing/bio-tagging-sequence-labeling-ner
- Practical relevance to cvinsight: skill/degree/title spans in resumes are short; the measured BIOES gain there is marginal. **Not the bottleneck.** If you retrain, an equally cheap improvement is fixing the class-imbalance (resume text is dominated by `O`) via class-weighted cross-entropy or focal loss, and evaluating with **entity-level exact-match F1 (seqeval)** instead of token accuracy.
  - Imbalance/weighted-loss discussion. https://dataannotationcompanies.com/blog/ner-data-labeling-ai-ultimate-guide

### Nested / discontinuous entities
- Handled properly only by span-based prediction (enumerate candidate spans, classify each), which is quadratic in sentence length vs linear sequence tagging. See the summary in the BIO tagging guide above (link: https://mbrenndoerfer.com/writing/bio-tagging-sequence-labeling-ner).
- Resume relevance: nested entities (e.g., a degree inside "Bachelor of Science in Computer Science" vs field-of-study inside it) do occur, but the common workaround is a *second label set* (venkatasagar's model uses separate `DEGREE`, `FIELD_OF_STUDY`, `GRADUATION_YEAR` labels — see below). Flat BIO is fine for cvinsight; do not buy span-based complexity.

### Can you put DATE / DURATION / RELATION labels on a token-classification head?
- **DATE ranges as a single span: yes, precedent exists.** `oksomu/resume-ner` labels `DATE` spans like "2020-2023" and "January 2022" — i.e., a whole range as one token span. https://huggingface.co/oksomu/resume-ner
- **Separate START/END labels: yes, precedent exists.** `venkatasagar/NER-roberta-finetuned` uses `START_DATE`, `END_DATE`, `DEGREE`, `FIELD_OF_STUDY`, `GRADUATION_YEAR`, `GPA` labels — showing a fine-tuned RoBERTa head can learn disjoint start/end date spans on a modest custom dataset. https://huggingface.co/venkatasagar/NER-roberta-finetuned
- **DURATION labeling has a deep literature** but it is a *markup* not a flat-span convention: TimeML's `TIMEX3` has a `type="DURATION"` with ISO-8601 `value` and `beginPoint`/`endPoint` attributes pointing at the two `DATE` anchors ("1992 through 1995" → duration `P4Y` with begin/end points). This is exactly your "two dates → duration" problem, standardized 20 years ago. Systems: **SUTime** (Stanford, rule-based) and **HeidelTime** (multilingual) implement it.
  - TIMEX3 spec incl. duration begin/end anchors. https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html
  - TimeBank 1.2 (the canonical annotated corpus; TIMEX3 DATE/DURATION/TIME/SET). https://catalog.ldc.upenn.edu/docs/LDC2006T08/timebank.html
- **Honest verdict for cvinsight:** you can add a `DATE` (or `START_DATE`/`END_DATE`) label to your distilbert head, but a single BIO span cannot express a *disjoint* "2020 → Present" pair as one entity, and **DURATION and RELATION are not flat spans** — they are links between spans. The low-effort path is: keep the model emitting flat `DATE` spans, and do pairing/duration math in a rule layer (you already have the date regexes). TimeML's beginPoint/endPoint framing is the model to copy.

---

## 2. Combining NER with rules (hybrid patterns)

The dominant, repeatedly-confirmed pattern in resume parsing is **rule-first, NER-as-support**, not NER-replacing-rules.

- **Hybrid = rules (high precision) + statistical NER (high recall), merged by a reconciliation layer.** This is the standard architecture description: conflict resolution via confidence-weighted voting, heuristic precedence for known-high-precision patterns, span-boundary reconciliation, plus a validation layer that checks plausibility constraints (dates within a valid timeline, etc.). https://inferensys.com/glossary/clinical-workflow-automation/medical-named-entity-recognition/hybrid-ner
- **Concrete resume hybrid systems report this beats any single method.** A rule+ML+transformer hybrid reported precision 87.62% / recall 96.91% (better than each component alone). https://github.com/JennyTan5522/NLP-Resume-Parsing — academic version (Bhoir et al. 2023, hybrid spaCy+BERT). https://www.authorea.com/doi/10.22541/au.168170278.82268853
- RegEx + spaCy hybrid for name/phone/email/university/experience/org + SBERT cosine matching. https://race.reva.edu.in/race-lab/a-hybrid-resume-parser-and-matcher-using-regex-and-ner

### Three concrete hybrid patterns worth copying

1. **Two-phase / extract-then-fill (rule-first reduces label space).** A 2026 low-resource NER paper makes this explicit: rule-based components first extract *deterministically recognizable* entities (phone, IDs, **dates via SUTime**), shrinking the label space; the neural model only handles the semantically ambiguous remainder; a post-processing module then restores fine-grained labels. https://arxiv.org/pdf/2605.04489
   - cvinsight mapping: run your regex/EntityRuler layers first for dates/phones/emails/obvious skills; feed the distilbert tagger the *remainder* or let it vote only on TITLE/COMPANY/DEGREE/INSTITUTION, where rules are weaker.
2. **NER as correction / veto with a confidence threshold.** Use the tagger's spans as *candidate* spans; keep the rule output as default; override rules only where the model is high-confidence and the two disagree (e.g., title/company swap, degree-vs-institution confusion you fixed manually in Phase 1c). spaCy ships a threshold mechanism in its `EntityLinker` (`threshold` param, predictions below it are dropped → NIL), the same idea. https://spacy.io/api/entitylinker/
3. **Cascade + overlap resolution.** In hybrid entity-recognition/linking pipelines, outputs from gazetteer vs supervised NER are merged by resolving overlaps: take the union of boundaries, keep the longest span for nested mentions, prefer the model's type for the surviving span; a pruning stage then raises precision. https://www.eurecom.fr/publication/4613/download/mm-publi-4613_1.pdf
   - This "longest-wins / union boundaries" merge is exactly the dedup + precedence logic you'd write for title/company conflicts.

### Why this matters for your +1pt skill finding
A dedicated paper documents the limitations of neural NER on resumes (entity type confusion, boundary errors, poor handling of sparse/spurious spans), which is precisely why gazetteer-based skill extraction saturates and why hybrid systems keep winning. https://github.com/dotin-inc/resume-dataset-NER-annotations (dataset + SEPLN paper *"Limitations of Neural Networks-based NER for Resume Data Extraction"*, 545 annotated resumes, 12 entities).

---

## 3. Context / section awareness: per-section NER

This is the single most transferable idea in the 2021–2026 resume literature and the one that most directly attacks your relational gap.

- **Section classifier → per-section NER** (zero-shot text classifier to label the section, then NER scoped to each section) is a recommended pattern specifically when labeled data is scarce (<200–300 documents). https://dredyson.com/how-i-mastered-advanced-resume-parsing-with-huggingface-models-the-complete-expert-configuration-guide-for-2026-including-hidden-ner-zero-shot-and-layoutlmv3-techniques-that-pros-dont-want-you-t/
- **Smart-Hiring (arXiv 2025)** operationalizes the ideal hybrid: names via a trained classifier, contact via regex *scoped to Contact/Profile sections*, skills via fuzzy lexicon match, education via fuzzy degree list, experience via **date-interval parsing inside the Experience section** with heuristics to fill missing entries. Section boundaries are the backbone of the whole extractor. https://arxiv.org/html/2511.02537v1
- **`oksomu/resume-ner` uses "section-aware chunked inference"** — sections are the unit of inference, which simultaneously solves >512-token documents (see §4). https://huggingface.co/oksomu/resume-ner
- Header/heading detection as its own ML task: SVM over typography/formatting/position features classifies section-heading tokens (vs content). https://github.com/Harshi115/Resume-Section-Classifier
- Industry ATS view on why standard headings + typography cues are how sections are found (keyword dict + bold/all-caps + position/proximity). https://www.jobshinobi.com/blog/ats-optimized-resume-section-headings-that-parse

**Why per-section beats whole-document here:** (a) the model stops confusing "Python" (skill) vs course names in Education; (b) TITLE/COMPANY conflicts resolve because in the Experience section a line has exactly one title and one company; (c) **degree↔institution and start↔end-date pairing become local, within-block problems** — an order of magnitude easier than global linking.

---

## 4. Long-document handling (resumes >512 tokens)

- Hugging Face's `TokenClassificationPipeline` now supports a `stride` argument: long text is split into overlapping windows and duplicate tokens are resolved by keeping the higher-scoring prediction. This is the canonical, built-in sliding-window NER. https://github.com/huggingface/transformers/issues/14631
- Reference defaults in the EDS-NLP library: window 512, stride 256. https://aphp.github.io/edsnlp/latest/pipes/trainable/embeddings/transformer/
- Chunking vs sliding-window practicalities (splitting at sentence/paragraph boundaries to avoid slicing entities; overlap cost). https://medium.com/@fz.iguenfer/handling-long-texts-in-ner-chunking-vs-sliding-window-79c89223b6db
- Resume-specific precedent: `oksomu/resume-ner` hand-crafted **50 long resumes (>512 tokens) specifically for chunked-inference training**, and does the chunking at section boundaries. https://huggingface.co/oksomu/resume-ner
- Sliding window + label alignment details for fine-tuning (word_ids, `-100` masking, `label_all_tokens`, stride/window). https://stackoverflow.com/questions/76351780/sliding-window-approach-while-finetuning-bert-for-ner-task

**Recommendation:** because you already have `section_splitter.py`, treat *sections* as the inference unit (never cut a section). Only if a single section exceeds 512 tokens, apply the HF `stride` sliding window with max-score overlap resolution. This preserves intra-section context (which matters for pairing) and avoids cross-section bleed.

---

## 5. Training-data side: diversity, augmentation, and DATE-labeled resume data

### What's public and usable
- **DataTurks Resume Entities for NER**: 220 manually labeled resumes, 10 labels — including **`Graduation Year` and `Years of Experience`**, which are effectively date/range labels. The most-used public resume NER corpus. https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
- **dotin-inc / SEPLN**: 545 resumes, 12 entities (incl. Graduation Year, Years of Experience). CC0. https://github.com/dotin-inc/resume-dataset-NER-annotations
- **Mehyaar/Annotated_NER_PDF_Resumes**: 5,029 CVs with manual **IT-skill** annotations (useful only for skills). https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes
- **datasetmaster/resumes**: ~4,817 real + synthetic resumes in a normalized JSON schema (experience dates, education, skills) — good for silver-label generation, not manually annotated. https://huggingface.co/datasets/datasetmaster/resumes
- **yashpwr/resume-ner-bert-v2**: the "22k" model you referenced — trained on 22,542 samples (349 Resume-Corpus + 420 DataTurks + 21,773 rule-generated "silver" labels + Mehyaar skills), 25 entity types, 90.87% F1. This is the strongest published evidence that **large silver-label + augmentation mixes make distilbert/bert resume NER work**. https://huggingface.co/yashpwr/resume-ner-bert-v2
- **oksomu/resume-ner** — the most complete open recipe: DataTurks + generated templates + 12 manual templates + **50 long resumes** + 93 gold-labeled PDFs + 2,483 Kaggle resumes with **Gemini-extracted silver labels + BIO tagging**, then **2× noise augmentation** (separator swaps, char corruption, case changes) for OCR robustness. Result: entity F1 97.77% clean / 69.24% noisy. https://huggingface.co/oksomu/resume-ner

### Is there a public DATE-annotated resume dataset?
**Partially — no TimeBank-for-resumes exists.** The closest public resources are (a) DataTurks/dotin `Graduation Year` + `Years of Experience` labels, and (b) `oksomu`'s `DATE` spans ("2020-2023") in its training set (reproducible from its open training data pipeline). The `venkatasagar` model with `START_DATE`/`END_DATE` does *not* publish its dataset ("contact the maintainer"). So: you will have to create start/end-date annotations yourself — but your 4,500 extracted CVs + existing date regexes give you a cheap **silver-label** source, following the oksomu/yashpwr recipe.

### Augmentation & noise
- Simple augmentation (synonym replacement, mention replacement, shuffle-within-segments) is very effective at small data sizes — the canonical reference is Dai & Adel (2020), summarized here: https://link.springer.com/article/10.1007/s10772-023-10055-8 ; recent survey: https://www.sciencedirect.com/science/article/pii/S0925231225015280
- **Caution — silver labels are noisy:** NoiseBench shows real label noise (LLM/auto-annotation, exactly what silver labels are) is much harder than simulated noise and that noise-robust methods under-deliver. Budget a small human-verified gold set and keep exact-match F1 as your metric. https://aclanthology.org/2024.emnlp-main.1011/
- DistilBERT is a legitimate choice on cost grounds: fine-tuned DistilBERT matched BERT F1 for NER on medical PHI at ~half the runtime/disk. https://aclanthology.org/2020.clinicalnlp-1.18/

---

## 6. Post-processing: dedup, alias normalization, phrase merging, taxonomy mapping

This is where NER becomes *usable* and where you get cheap, reliable points — and it maps 1:1 onto your `skill_taxonomy.json`.

- **`oksomu/resume-ner` treats post-processing as a first-class, config-driven stage**: `resume_config.json` defines normalization (CRLF/em-dash/bullet normalization, stripping `Phone:`/`Email:` labels, expanding flattened two-column skill tables), plus gazetteers (`companies.json` for company normalization, `city_country_map.json` with 317 cities for country inference) and structured-field assembly. https://huggingface.co/oksomu/resume-ner
- **Skills normalization onto a taxonomy** is a solved pattern with a clear output schema: canonical name + `synonyms[]` + `match_method` (`exact`/`alias`/`implied`/`none`) + category + proficiency + confidence. "React JS"/"ReactJS" → `React`, and matched skills imply others ("Django" ⇒ "Python"). https://theresumeparser.com/help/guides/skills-normalization
- Open implementations of the same idea (alias table + canonical id + transferability): https://github.com/tanova-ai/skills-taxonomy
- **Public taxonomies to anchor against** if you ever want to grow beyond 270 skills: ESCO (~14k skills, 28 languages) https://esco.ec.europa.eu/en/classification/skill_main ; O*NET; Lightcast Open Skills (30k+, posting-derived). Comparison: https://jobspipe.dev/blog/skills-taxonomy
- Transformer-based **normalized** skill extraction onto ESCO exists (SkiLLMo) if you later want a learned layer: https://dl.acm.org/doi/10.1145/3672608.3707960
- spaCy's `EntityLinker` gives you the machinery for gazetteer-style disambiguation with a threshold: https://spacy.io/api/entitylinker/
- Overlap/nested resolution rules (longest wins, union boundaries) from §2 apply here for dedup: https://www.eurecom.fr/publication/4613/download/mm-publi-4613_1.pdf

**Cheap wins for cvinsight in this bucket:** canonicalize skill surface forms before scoring (alias dictionary: React.js/ReactJS/React → one id; "ML"/"Machine Learning"), and make the *score* run on canonical ids rather than raw extracted strings — this is pure config + a dict, no model work.

---

## 7. Can encoder-NER produce relational structure cheaply?

Yes, but not with a plain token head. Three cost tiers, all LLM-free:

### Tier A — reframe as QA (strongest precedent, highest effort)
**Multi-turn QA extraction (Li et al., ACL 2019)** is *the* canonical relational-resume citation: it casts extraction as answering "What company did X work for?" / "What position?" / "What time period?" over a paragraph, and it released the **RESUME** dataset — executive work-history paragraphs with `Person/Company/Position/Time` entities and dependency chains (position depends on company, time depends on both). The QA model reached SOTA on that dataset (entity F1 83.6, relation F1 49.4). This is proof that **encoder-only models can do the exact degree↔institution / title↔company / start↔end structure** if you reformulate the task. https://aclanthology.org/P19-1129/ (data+code: https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA)

### Tier B — token-pair linking / joint tagging (medium effort)
**TPLinker (COLING 2020)** turns joint extraction into *token-pair linking*: for each relation, tag the token pairs (head-head, tail-tail) linking two spans, decoded in one stage. Single-sequence-labeling head on pairs → no autoregressive decoding, handles nested/overlapping spans, still just a fine-tuned encoder. https://aclanthology.org/2020.coling-main.138/
- Span-based joint NER+RE is the complementary family (enumerate spans, classify span pairs): Ji et al. COLING 2020 https://www.semanticscholar.org/paper/2b0489a440f8a39ed320e0ed879e29e0fcb87e09 ; STSN (sequence tagging + span) https://link.springer.com/article/10.1007/s11432-022-3608-y
- Multi-head selection (Bekoulis et al. 2018) is the lightest joint formulation (per-token heads predict links to other tokens). https://www.sciencedirect.com/science/article/pii/S0957417418302307

### Tier C — pipeline: block segmentation + a small pairwise classifier (lowest effort, best ROI for cvinsight)
Your relation set is tiny (4 relations: degree↔institution, title↔company, start↔end, company↔dates) and — crucially — **all pairing is local**: within one experience block or one education entry. That means:
1. NER (or your existing sections) produces block-local candidate spans.
2. A **pairwise classifier over span-pair features** (concatenated distilbert span embeddings + distance/line-distance/block-membership/section features) decides each pair. This is a small logistic/MLP head — not a new tagger — and it is exactly the "second lightweight pairwise classifier" you asked about. Because candidates per block are few, it is cheap to train and fast to run.
3. Dates pair as start/end by block + temporal order + "Present"/"Till date" sentinel, following TimeML beginPoint/endPoint semantics (§1) — this is mostly rules you already have, and it directly produces the **duration** that closes your score gap.

**Honest answer to "can encoder-NER do relations cheaply":** not with a BIO head alone, but with a *post-NER span-linking layer* on top of a fine-tuned encoder it is very practical — the pair space is tiny in resumes because structure makes the problem local. TPLinker/QA are the general solutions; for cvinsight a per-section pairwise classifier is the proportionate one.

---

## What we can actually add to cvinsight (no LLM) — prioritized by ROI

Ranking rationale: highest ROI = attacks the actual score gap (relational fields), reuses existing assets (sections, date regexes, taxonomy, 4,500 extracted CVs, `models/ner-v1`), and needs no new infrastructure. Effort is honest person-days for one engineer.

### P1 — Per-section, block-scoped extraction with local pairing  (effort: low-medium, payoff: high)
Run your pipeline per section (Education vs Experience vs Skills), and within each section segment into blocks (one job = one block via section splitter + layout/blank-line heuristics). Then do **local pairing inside each block**:
- title↔company, degree↔institution: nearest/line-proximity pairing (rule first, NER spans as tie-break and validation).
- start↔end dates → duration: regex DATE spans per block, temporal-order + "Present"/"Till date" sentinels, TimeML beginPoint/endPoint style.
This is the cheapest possible realization of the relational fields and lands directly on your biggest score loss.
- Sources: Smart-Hiring section-scoped extractor https://arxiv.org/html/2511.02537v1 ; per-section NER pattern https://dredyson.com/how-i-mastered-advanced-resume-parsing-with-huggingface-models-the-complete-expert-configuration-guide-for-2026-including-hidden-ner-zero-shot-and-layoutlmv3-techniques-that-pros-dont-want-you-t/ ; TIMEX3 duration model https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html

### P2 — Use models/ner-v1 as a high-confidence veto/corrector, not a replacer  (effort: low, payoff: medium)
Keep rule output as default; where the tagger's TITLE/COMPANY/DEGREE/INSTITUTION spans are above a calibrated confidence threshold *and* disagree with rules, override (this automates the manual title/company-swap and degree-vs-institution fixes from Phase 1c). Below threshold, fall back to rules. Calibrate the threshold on your labeled demo/borderline set (you already have `borderline_review.csv`).
- Sources: hybrid reconciliation layer https://inferensys.com/glossary/clinical-workflow-automation/medical-named-entity-recognition/hybrid-ner ; threshold semantics https://spacy.io/api/entitylinker/ ; NER-limitations motivation https://github.com/dotin-inc/resume-dataset-NER-annotations

### P3 — Skill/entity normalization onto skill_taxonomy.json (canonical ids)  (effort: low, payoff: medium, and prerequisite for JD matching)
Alias dictionary (React.js/ReactJS/React, ML/Machine Learning), canonical-id output, `match_method` bookkeeping, implied-skill expansion. Run scoring on canonical ids. This is config + dict work; it also makes Week-6 JD matching meaningful.
- Sources: https://theresumeparser.com/help/guides/skills-normalization ; https://github.com/tanova-ai/skills-taxonomy ; oksomu post-processing config https://huggingface.co/oksomu/resume-ner

### P4 — Section-aware / sliding-window inference for the tagger on long CVs  (effort: low, payoff: medium)
Infer per-section (never cut a section); if a section >512 tokens, use HF `stride` sliding window with max-score overlap resolution. Expect recall gains on verbose CVs and cleaner DATE/TITLE spans feeding P1.
- Sources: https://github.com/huggingface/transformers/issues/14631 ; https://aphp.github.io/edsnlp/latest/pipes/trainable/embeddings/transformer/ ; oksomu long-resume training https://huggingface.co/oksomu/resume-ner

### P5 — Retrain the tagger with silver labels + augmentation + exact-match eval  (effort: medium, payoff: medium — enables P2)
You have 4,500 extracted CVs: generate silver BIO labels from your rules (title/company/degree/institution/date), mix in DataTurks (220) and dotin (545) gold data, add 2× OCR-style noise (separator swap, char corruption, case changes — oksomu recipe), evaluate with **entity-level exact-match F1 (seqeval)**, and consider class-weighted loss for the O-class imbalance. This is the proven path (yashpwr 22k, oksomu) and directly improves the TITLE/COMPANY/DATE spans that P1/P2 consume. Budget a small human-verified gold set because silver noise is real (NoiseBench).
- Sources: https://huggingface.co/oksomu/resume-ner ; https://huggingface.co/yashpwr/resume-ner-bert-v2 ; https://aclanthology.org/2024.emnlp-main.1011/ ; https://www.sciencedirect.com/science/article/pii/S0925231225015280

### P6 — Per-block pairwise relation classifier (the deep version, only if heuristics underperform)  (effort: high, payoff: medium)
If P1's proximity heuristics fail on a holdout (say <90% pairing accuracy), train a small pairwise classifier over span-pair features (concat distilbert span embeddings + line distance + block + section) for degree↔institution and title↔company. Cheap because the pair space is local. Do **not** build full TPLinker/QA machinery for four local relations.
- Sources: span-pair classification https://www.semanticscholar.org/paper/2b0489a440f8a39ed320e0ed879e29e0fcb87e09 ; TPLinker as the general fallback https://aclanthology.org/2020.coling-main.138/ ; QA as the gold-standard benchmark https://aclanthology.org/P19-1129/

---

## Honest what-NOT-to-do

- **Do not add DURATION or RELATION labels to the token head.** DURATION is a link between two DATE anchors, not a span; a disjoint "2020 – Present" cannot be one BIO span. Keep spans flat; do linking separately (TimeML/TIMEX3 semantics). https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html
- **Do not buy nested/discontinuous or span-based NER** for this domain — flat BIO covers resume entities; the relational gap is a *linking* gap, solved in P1/P6.
- **Do not expect NER to improve skills.** Your +1pt finding is consistent with the literature (gazetteer/lexicon skill extraction saturates; hybrid systems exist precisely because of NER's weaknesses on sparse resume text). Spend the model budget on TITLE/COMPANY/DEGREE/INSTITUTION/DATE where rules are weaker.
- **Do not treat silver labels as gold.** LLM/rule-generated labels carry real noise that naive models absorb (NoiseBench). Keep a small hand-checked set and report exact-match F1.

---

## Key sources at a glance

| Topic | Source | URL |
|---|---|---|
| Label schemes | 7-scheme NER study (2021) | https://www.sciencedirect.com/science/article/pii/S1110866520301596 |
| BIO/BIOES/span-based | BIO tagging guide | https://mbrenndoerfer.com/writing/bio-tagging-sequence-labeling-ner |
| Duration as links | TimeML/TIMEX3 spec | https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html |
| DATE/START-END labels | oksomu / venkatasagar models | https://huggingface.co/oksomu/resume-ner · https://huggingface.co/venkatasagar/NER-roberta-finetuned |
| Hybrid resume NER | JennyTan repo / Bhoir 2023 | https://github.com/JennyTan5522/NLP-Resume-Parsing · https://www.authorea.com/doi/10.22541/au.168170278.82268853 |
| Two-phase rule-first NER | Low-resource hybrid (2026) | https://arxiv.org/pdf/2605.04489 |
| Section-scoped extraction | Smart-Hiring | https://arxiv.org/html/2511.02537v1 |
| Per-section NER w/ zero-shot sections | Dredyson guide (2026) | https://dredyson.com/how-i-mastered-advanced-resume-parsing-with-huggingface-models-the-complete-expert-configuration-guide-for-2026-including-hidden-ner-zero-shot-and-layoutlmv3-techniques-that-pros-dont-want-you-t/ |
| Sliding-window NER | HF pipeline stride / EDS-NLP | https://github.com/huggingface/transformers/issues/14631 · https://aphp.github.io/edsnlp/latest/pipes/trainable/embeddings/transformer/ |
| Silver labels + augmentation | oksomu / yashpwr (22k) | https://huggingface.co/oksomu/resume-ner · https://huggingface.co/yashpwr/resume-ner-bert-v2 |
| Label noise | NoiseBench (EMNLP 2024) | https://aclanthology.org/2024.emnlp-main.1011/ |
| Skills normalization | The Resume Parser / tanova | https://theresumeparser.com/help/guides/skills-normalization · https://github.com/tanova-ai/skills-taxonomy |
| Taxonomies | ESCO / Lightcast / O*NET | https://esco.ec.europa.eu/en/classification/skill_main · https://jobspipe.dev/blog/skills-taxonomy |
| Relational resume extraction | Multi-turn QA (RESUME dataset) | https://aclanthology.org/P19-1129/ |
| Token-pair joint extraction | TPLinker | https://aclanthology.org/2020.coling-main.138/ |
| Span-based joint | Ji et al. / STSN | https://www.semanticscholar.org/paper/2b0489a440f8a39ed320e0ed879e29e0fcb87e09 · https://link.springer.com/article/10.1007/s11432-022-3608-y |
