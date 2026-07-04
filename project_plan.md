CV EVALUATOR & RANKING
SYSTEM
Full Project Plan — Week-by-Week Roadmap
Version 1.0 | This document is the final project metric
PROJECT AT A GLANCE
Goal: Build an automatic CV evaluator, scorer, and ranking system
Duration: 7 weeks (Week 1 – Week 7)
Versions: V1 = Upload → Extract → Score → Suggest
V2 = Add job description → Compute match % → Rank CVs
V3 = Fine-tune NER model on self-labeled data
Stack: Python 3.10, spaCy, XGBoost, Sentence-Transformers, Streamlit
Deploy: Hugging Face Spaces (free, shareable link)
Rule: All coding decisions in this project refer back to this document as the single source of truth.
Table of Contents

1. Overview & Scope Definition
2. Dataset Reference
3. Output JSON Schema
4. Scoring Rubric
5. Folder Structure
6. Technology Stack
7. Week-by-Week Plan
8. Evaluation Metrics
9. Risk Register
10. Delivery Checklist
11. Overview & Scope Definition
    This document defines the complete execution plan for the CV Evaluator project. Every coding session,
    dataset choice, model decision, and deliverable is governed by the specifications below. Do not deviate
    from the architecture described here without updating this document first.
    Project versions (build in order)
    V1 — Core Evaluator (Weeks 1 – 5) — MANDATORY
    ✔ Upload CV (PDF / DOCX / plain text)
    ✔ Extract: name, education, skills, experience, projects, certifications, languages
    ✔ Score CV with weighted rubric (0–100)
    ✔ Classify: Strong / Average / Weak
    ✔ Generate improvement suggestions per section
    V2 — JD Matching & Ranking (Week 6) — TARGET
    ✔ Paste or upload a Job Description alongside CV(s)
    ✔ Compute semantic similarity score (CV vs JD)
    ✔ Compute skill overlap score
    ✔ Produce a final match % and ranked list of multiple CVs
    Ranking formula: 0.5 × semantic_similarity + 0.3 × skill_overlap + 0.2 × rubric_score
    V3 — Model Fine-Tuning (Week 7) — OPTIONAL STRETCH
    ✔ Label 200–500 CVs using the rubric as a pseudo-labeling tool
    ✔ Fine-tune NER model on self-generated labels
    ✔ Retrain classifier on improved feature vectors
    ✔ Compare V1 vs V3 accuracy and report improvement
12. Dataset Reference
    Use exactly these five datasets. Do not substitute without updating this document.
    Dataset Size Used For Source
    Mehyaar/Annotated_NER_PDF_Resumes 5,029 CVs NER model
    training (skills,
    entities)
    Hugging Face
    datasetmaster/resumes ~3,
    CVs
    Parser dev &
    structured JSON
    examples
    Hugging Face
    noran-mohamed/Resume-Classification-
    Dataset
    13,
    CVs
    Category
    classification,
    quality labels
    GitHub
    0xnbk/resume-ats-score-v1-en 6,374 pairs JD matching &
    ATS score
    prediction
    Hugging Face
    netsol/resume-score-details 1,031 pairs Extra scored
    resume-JD
    examples (GPT-
    4o)
    Hugging Face
    IMPORTANT — Label noise warning
    Public resume datasets are labeled by job category, NOT by quality.
    A "Data Science" CV could be excellent or terrible — that tag tells you nothing about strength.
    You must generate Strong/Average/Weak labels yourself using the rubric in Section 4.
    This is called pseudo-labeling and it is the hardest part of the project. Budget time for it in Week 3.
    IMPORTANT — PDF parsing failure warning
    Approximately 20–30% of real CVs are image-based (scanned) PDFs.
    pdfplumber returns empty text on these. Add pytesseract as a fallback OCR step.
    Detection rule: if len(extracted_text.strip()) < 50: run OCR pipeline instead.
13. Output JSON Schema
    Define this schema FIRST. Every module — parser, scorer, suggester, UI — reads and writes to this
    structure. Lock it in before writing any code.
    {
    "cv_id": "string — unique hash of the CV file",
    "name": "string — extracted full name",
    "email": "string — extracted email address",
    "phone": "string — extracted phone number",
    "education": [
    { "degree": "string", "institution": "string",
    "year": "int|null", "gpa": "float|null", "field": "string" }
    ],
    "experience": [
    { "title": "string", "company": "string",
    "start": "YYYY-MM|null", "end": "YYYY-MM|null",
    "duration_months": "int", "description": "string" }
    ],
    "skills": ["list of strings"],
    "projects": [
    { "name": "string", "tools": ["list"], "description": "string", "link":
    "string|null" }
    ],
    "certifications": [{ "name": "string", "issuer": "string", "year": "int|null" }],
    "languages": [{ "language": "string", "proficiency": "string|null" }],
    "achievements": ["list of strings"],
    "leadership": ["list of strings"],
    "section_scores": {
    "experience": "int", "projects": "int", "skills": "int",
    "education": "int", "certifications": "int",
    "languages": "int", "leadership": "int"
    },
    "total_score": "int — 0 to 100",
    "label": "string — Strong | Average | Weak",
    "suggestions": ["list of strings — max 5 actionable tips"],
    "jd_match": {
    "semantic_similarity": "float 0–1",
    "skill_overlap": "float 0–1",
    "final_match_score": "float 0–1",
    "missing_skills": ["list of strings"]
    }
    }
14. Scoring Rubric
    This rubric is stored in config/rubric_config.json so weights can be changed without modifying any
    code. The table below is the default configuration.
    Section Points Scoring Logic Sub-criteria
    Work Experience 25 0 - 1yr=8, 1-3yr=15,
    3 - 5yr=20, 5+yr=
    Relevance to role,
    title seniority,
    quantified
    achievements
    Project Experience 20 Count×4 (max 20),
    bonus for GitHub
    links
    Technical depth,
    tools used, outcomes
    described
    Technical Skills 20 Matched skills /
    required skills × 20
    Skill taxonomy
    match, proficiency
    indicators
    Education 15 PhD=15,
    Masters=13,
    Bachelors=10,
    Diploma=
    Institution ranking,
    GPA if listed,
    relevant field
    Certifications &
    Achievements
    10 Count×2 (max 10) Issuing body quality,
    recency, relevance
    Languages 5 1 lang=2, 2
    langs=4, 3+
    langs=
    Proficiency level
    specified (B2/C1 etc.)
    Leadership & Extras 5 Count of roles × 2
    (max 5)
    Clubs, volunteering,
    open source
    contributions
    Score categories 80 – 100 = Strong | 50–79 =
    Average | 0–49 = Weak
    Rubric config file — config/rubric_config.json
    Store all weights in a JSON file. Your code reads this at startup.
    This means your teacher can change scoring without touching Python code — this is impressive.
    Example key: "experience": { "max_points": 25, "thresholds": [1,3,5] }
15. Folder Structure
    Create this folder structure on Day 1 of Week 1. Every file you write goes into the correct folder from
    the start.
    cv_evaluator/ Root project folder
    data/ Raw and processed CV datasets
    raw/ Downloaded datasets (do not edit)
    processed/ Cleaned, labeled CSV/JSON files
    src/ All source code modules
    parser/ PDF/DOCX ingestion & text cleaning
    extractor/ NER + rule-based entity extraction
    scorer/ Rubric scoring engine
    matcher/ JD semantic similarity module
    suggester/ Feedback generation logic
    models/ Saved trained model files (.pkl, .pt)
    config/ rubric_config.json, skill_taxonomy.json
    app/ Streamlit UI code
    notebooks/ Jupyter notebooks for EDA & experiments
    tests/ Unit tests per module
    requirements.txt All Python dependencies
    README.md Project overview & setup guide
16. Technology Stack
    These are the only libraries approved for this project. Do not add new major dependencies without
    updating this section.
    Layer Library / Tool Purpose
    Ingestion pdfplumber, python-docx Extract text from PDF and DOCX files
    Ingestion pytesseract (fallback) OCR for scanned/image-based PDFs
    NER / Extraction spaCy (en_core_web_sm) Named entity recognition, EntityRuler,
    PhraseMatcher
    NER / Extraction Hugging Face transformers Fine-tuned BERT on NER resume dataset
    Scoring scikit-learn Logistic Regression, Random Forest baseline
    Scoring XGBoost Improved classifier for Strong/Average/Weak
    Matching sentence-transformers all-MiniLM-L6-v2 for CV-JD semantic similarity
    Matching sklearn cosine_similarity Compute similarity scores
    Data pandas, numpy Feature engineering, data manipulation
    Config JSON config files rubric_config.json (editable weights without
    code changes)
    App / UI Streamlit Web interface for upload, scoring, results
    Deployment Hugging Face Spaces Free cloud hosting with shareable link
    Environment Python 3.10+, virtualenv Reproducible development environment
    Versioning Git + GitHub Code + experiment tracking
    Notebooks Jupyter Lab EDA, model experiments, visualizations
17. Week-by-Week Plan
    This is the core of the document. Each week has a fixed goal, daily tasks, and concrete deliverables. A
    week is considered complete only when all deliverables are produced.
    WEEK 1 — Foundation & Dataset Setup
    Days 1 – 7 | Goal: environment running, datasets downloaded, folder structure created
    Day Task Output/Deliverable Tools / Notes
    Day 1 Create GitHub repo, set up Python
    virtualenv, install all libraries from
    requirements.txt
    Working virtualenv,
    requirements.txt
    committed
    Python 3.10,
    pip, git init
    Day 2 Create the full folder structure (Section 5).
    Create rubric_config.json with default
    weights. Create skill_taxonomy.json (start
    with 200 IT skills)
    Folder structure + config
    files
    Use VS Code,
    JSON files
    Day 3 Download
    Mehyaar/Annotated_NER_PDF_Resumes
    and datasetmaster/resumes from Hugging
    Face. Run basic pandas EDA on both
    EDA notebook: shape,
    columns, null counts,
    sample rows
    datasets library,
    pandas, Jupyter
    Day 4 Download noran-mohamed Resume
    Classification dataset from GitHub.
    Download 0xnbk/resume-ats-score-v1-en
    and netsol/resume-score-details from
    Hugging Face
    All 5 datasets downloaded
    and stored in data/raw/
    Kaggle API or
    manual
    download
    Day 5 Clean all datasets: strip HTML, normalize
    whitespace, remove duplicates,
    standardize column names. Save cleaned
    versions to data/processed/
    cleaned_ner.csv,
    cleaned_resumes.csv,
    cleaned_classification.csv,
    cleaned_ats.csv
    pandas, regex,
    chardet
    Day 6 Write the output JSON schema (Section

3) as a Python dataclass or TypedDict.
   Write a schema validator function
   schema.py with
   CVSchema class +
   validate_schema()
   function
   dataclasses or
   pydantic
   Day 7 Weekly review: all datasets loaded
   without errors, schema defined, folder
   clean, git push. Fix any environment
   issues
   Week 1 progress
   notebook, git tag v0.
   git
   Week 1 deliverables checklist
   ☐ GitHub repo created and README written
   ☐ All 5 datasets downloaded into data/raw/
   ☐ All 5 datasets cleaned and in data/processed/
   ☐ rubric_config.json created with weights from Section 4
   ☐ skill_taxonomy.json created with 200+ skills
   ☐ schema.py with CVSchema and validator committed
   WEEK 2 — CV Parser & Text Extraction
   Days 8 – 14 | Goal: read any CV file and produce clean structured text
   Day Task Output/Deliverable Tools / Notes
   Day 8 Write PDF parser using
   pdfplumber. Extract raw text.
   Handle multi-column layouts
   (try both column-aware and
   raw-text modes)
   src/parser/pdf_parser.py
   with parse_pdf(path) → str
   pdfplumber
   Day 9 Write DOCX parser using
   python-docx. Write plain text
   (.txt) loader. Unify all three
   into a single entry function
   src/parser/parser.py with
   parse_cv(path) → str
   (handles all formats)
   python-docx
   Day 10 Add pytesseract OCR
   fallback: if extracted text <
   50 chars, run OCR pipeline.
   Test on 5 scanned CVs
   OCR fallback integrated in
   parser.py. Test results
   notebook
   pytesseract, Pillow,
   pdf2image
   Day 11 Write section detector: regex

- keyword rules to split CV
  text into sections (Education,
  Experience, Skills, etc.)
  src/parser/section_splitter.py
  with split_sections(text) →
  dict
  regex, spaCy
  Day 12 Write text cleaner: remove
  special chars, normalize
  whitespace, fix encoding
  issues. Write unit tests for
  parser
  src/parser/cleaner.py,
  tests/test_parser.py (5+ tests
  pass)
  pytest
  Day 13 Test parser on 20 real CVs
  from dataset. Log failure
  cases. Fix top 3 most
  common failure patterns
  parser_test_results.csv with
  pass/fail/reason for 20 CVs
  pandas
  Day 14 Weekly review, git push.
  Parser must handle PDF,
  DOCX, TXT and produce
  section dict for 85%+ of test
  CVs
  Week 2 notebook, git tag
  v0.
  git
  Week 2 deliverables checklist
  ☐ parse_cv(path) works for PDF, DOCX, and TXT
  ☐ OCR fallback integrated for image-based PDFs
  ☐ split_sections() returns dict with section names as keys
  ☐ Unit tests pass: pytest tests/test_parser.py
  ☐ Tested on 20 real CVs — 85%+ success rate
  WEEK 3 — Information Extraction (NER)
  Days 15 – 21 | Goal: extract all entities from CV text into structured JSON
  Day Task Output/Deliverable Tools / Notes
  Day 15 Set up spaCy pipeline.
  Load en_core_web_sm.
  Write regex extractors for
  email, phone, LinkedIn
  URL
  src/extractor/contact_extractor.py —
  tested on 10 CVs
  spaCy, regex
  Day 16 Build EntityRuler +
  PhraseMatcher for skills
  using skill_taxonomy.json.
  Match skills in text
  regardless of case/order
  src/extractor/skill_extractor.py with
  extract_skills(text) → list
  spaCy EntityRuler
  Day 17 Write education extractor:
  find degree names,
  institution names, years,
  GPA. Use keyword lists +
  NER
  src/extractor/education_extractor.py
  — returns list of education dicts
  spaCy NER +
  regex
  Day 18 Write experience extractor:
  find job titles, company
  names, date ranges.
  Compute duration_months
  for each role. Sum to
  total_experience_years
  src/extractor/experience_extractor.py
  — returns list of experience dicts +
  total years
  regex, dateparser
  Day 19 Write extractors for:
  projects, certifications,
  languages, achievements,
  leadership. Use section
  text + keyword matching
  src/extractor/misc_extractor.py — all
  remaining fields extracted
  regex + keyword
  lists
  Day 20 Build master extractor:
  calls all sub-extractors and
  returns full CVSchema
  JSON object. Validate
  output against schema.py
  src/extractor/extractor.py with
  extract_all(text) → CVSchema
  All extractors +
  schema validator
  Day 21 Test extractor on 30 CVs.
  Manually verify 10 of them.
  Compute precision/recall
  for skills extraction. Target
  F1 ≥ 0.
  extraction_eval.csv, F1 score
  reported in notebook
  sklearn.metrics
  Week 3 deliverables checklist
  ☐ extract_all(text) returns valid CVSchema JSON
  ☐ Skills extraction F1 ≥ 0.75 on manual test set
  ☐ Experience duration computed correctly in months
  ☐ Education, certifications, languages extracted
  ☐ 200 – 300 CVs processed and stored as JSON in data/processed/
  WEEK 4 — Scoring Engine & Label Generation
  Days 22 – 28 | Goal: produce a score + Strong/Average/Weak label for every CV
  Day Task Output/Deliverable Tools / Notes
  Day 22 Write scoring functions for
  each rubric section. Read
  weights from
  rubric_config.json. Score
  experience, education,
  skills
  src/scorer/section_scorers.py —
  functions for each section
  JSON config,
  pydantic
  Day 23 Write scoring functions for
  projects, certifications,
  languages, leadership.
  Assemble total_score and
  label
  src/scorer/scorer.py with
  score_cv(cv_schema) → scored
  CVSchema
  All section scorers
  Day 24 Run scorer on all 200– 300
  extracted CVs. Inspect
  score distribution. Check
  that labels are reasonably
  distributed (not all Weak)
  score_distribution.png notebook
  chart, labeled_cvs.csv
  pandas, matplotlib
  Day 25 Generate pseudo-labels:
  use the rubric score as the
  ground truth label for
  supervised training. Flag
  borderline CVs (score 45–
  55 and 75–85) for manual
  review
  labeled_cvs.csv with label column,
  borderline_review.csv
  pandas
  Day 26 Manually review and
  correct 50 borderline CVs.
  Adjust rubric weights if the
  scoring feels wrong.
  Update rubric_config.json
  Corrected labels committed,
  updated rubric_config.json
  Manual review +
  JSON edit
  Day 27 Write suggestion
  generator: for each section
  with score below threshold,
  generate a specific
  suggestion string from a
  template dict
  src/suggester/suggester.py with
  generate_suggestions(cv_schema)
  → list of strings
  Template dict + if-
  then rules
  Day 28 Weekly review. Score +
  label + suggestions
  working end-to-end for
  single CV. Git push
  End-to-end test notebook: input
  PDF → CVSchema with score,
  label, suggestions. git tag v0.
  Full pipeline test
  Week 4 deliverables checklist
  ☐ score_cv() reads weights from rubric_config.json
  ☐ Score distribution is reasonable (not all one category)
  ☐ labeled_cvs.csv with 200+ CVs has Strong/Average/Weak labels
  ☐ 50 borderline CVs manually reviewed and corrected
  ☐ generate_suggestions() returns 3–5 specific tips per CV
  ☐ End-to-end: PDF → JSON → score → label → suggestions works
  WEEK 5 — Classifier Training & Streamlit V
  Days 29 – 35 | Goal: trained model + working web app for V
  Day Task Output/Deliverable Tools / Notes
  Day 29 Build feature vector from
  CVSchema: convert all
  extracted fields to numbers
  (degree_level, exp_years,
  skill_count, project_count,
  cert_count, etc.)
  src/scorer/feature_builder.py
  with
  build_features(cv_schema)
  → np.array
  numpy, pandas
  Day 30 Train Logistic Regression
  baseline classifier on
  labeled_cvs.csv. Run 5-fold
  cross-validation. Report
  accuracy + F
  models/lr_baseline.pkl,
  baseline_results.txt
  sklearn
  Day 31 Train XGBoost classifier.
  Compare with LR baseline.
  Save best model. Report
  improvement
  models/xgb_classifier.pkl,
  model_comparison.csv
  xgboost
  Day 32 Build Streamlit app V1: file
  uploader, calls full pipeline,
  displays score, label, section
  breakdown bar chart,
  suggestions list
  app/app.py — running
  locally on localhost:
  streamlit, plotly or
  st.bar_chart
  Day 33 Polish UI: add color coding
  (green=Strong,
  orange=Average,
  red=Weak), add section-by-
  section score table, add
  download button for JSON
  output
  app/app.py V1 polished streamlit
  Day 34 Test V1 with 10 different
  CVs. Fix bugs. Write unit
  tests for scorer and feature
  builder. Ensure all tests pass
  tests/test_scorer.py,
  tests/test_features.py — all
  passing
  pytest
  Day 35 Deploy to Hugging Face
  Spaces. Verify it works
  online. Share link. Git push.
  Tag v1.
  Live URL on HuggingFace
  Spaces, git tag v1.
  HuggingFace Spaces,
  requirements.txt
  Week 5 deliverables checklist
  ☐ XGBoost classifier trained, accuracy ≥ 0.80 on validation set
  ☐ Streamlit app V1 running locally and on Hugging Face Spaces
  ☐ App displays: score, label, section breakdown, suggestions
  ☐ All unit tests pass (parser, extractor, scorer)
  ☐ Git tag v1.0 pushed
  WEEK 6 — JD Matching, Ranking & V2 App
  Days 36 – 42 | Goal: add job description matching and CV ranking to the app
  Day Task Output/Deliverable Tools / Notes
  Day 36 Set up sentence-
  transformers. Load all-
  MiniLM-L6-v2 model. Write
  function to embed a CV and
  a JD as single vectors
  src/matcher/embedder.py with
  embed_text(text) → np.array
  sentence-
  transformers
  Day 37 Write semantic similarity
  scorer: cosine similarity
  between CV embedding
  and JD embedding
  src/matcher/semantic_scorer.py
  with semantic_score(cv_text,
  jd_text) → float
  sklearn
  cosine_similarity
  Day 38 Write skill overlap scorer:
  extract JD skills using
  skill_taxonomy.json,
  compute overlap ratio with
  CV skills. Identify
  missing_skills list
  src/matcher/skill_overlap.py
  with
  skill_overlap_score(cv_skills,
  jd_text) → float, list
  spaCy
  PhraseMatcher
  Day 39 Write ranking function using
  the formula: final =
  0.5×semantic +
  0.3×skill_overlap +
  0.2×rubric_score. Rank list
  of CVs against one JD
  src/matcher/ranker.py with
  rank_cvs(cv_list, jd_text) →
  sorted list with scores
  All matcher modules
  Day 40 Evaluate matching on
  0xnbk/resume-ats-score-v1-
  en dataset. Compute
  Spearman correlation vs
  provided scores. Target ρ ≥

0. matching_eval.ipynb, Spearman
   ρ reported
   scipy.stats
   Day 41 Add V2 features to Streamlit
   app: JD text box, upload
   multiple CVs, ranked results
   table with match %, missing
   skills highlighted
   app/app.py V2 with ranking tab streamlit
   Day 42 Test V2 end-to-end. Deploy
   updated app. Git push. Tag
   v2.
   Live V2 URL, git tag v2.0 HuggingFace
   Spaces
   Week 6 deliverables checklist
   ☐ embed_text() produces consistent vectors for CV and JD text
   ☐ skill_overlap_score() returns missing_skills list
   ☐ rank_cvs() returns sorted list using the defined formula
   ☐ Spearman correlation ρ ≥ 0.65 on ATS dataset
   ☐ Streamlit V2 deployed with ranking tab live
   ☐ Git tag v2.0 pushed
   WEEK 7 — Fine-Tuning, Evaluation & Final Report
   Days 43 – 49 | Goal: evaluate all modules, write report, prepare demo
   Day Task Output/Deliverable Tools / Notes
   Day 43 (V3 optional) Label 200– 500
   CVs using rubric scorer as
   pseudo-labeler. Fine-tune
   spaCy NER on Mehyaar
   annotated dataset
   models/ner_finetuned/ saved
   model
   spaCy training CLI
   Day 44 (V3 optional) Retrain XGBoost
   with fine-tuned NER features.
   Compare V1 vs V3 accuracy
   in a notebook
   model_v3_comparison.ipynb
   with improvement table
   xgboost, sklearn
   Day 45 Run full evaluation suite: NER
   F1, classifier accuracy/F1,
   ranking NDCG@5, Spearman
   ρ. Spot-check 10 suggestions
   for quality
   evaluation_report.ipynb with
   all metrics filled in
   All eval metrics from
   Section 8
   Day 46 Fix any failing eval metrics.
   Re-tune rubric weights if
   needed. Update
   rubric_config.json. Re-run
   eval
   Updated models + config,
   final eval scores
   JSON + model
   retraining
   Day 47 Write README.md: project
   overview, setup instructions,
   how to run, how to deploy,
   dataset citations, known
   limitations
   Complete README.md in
   repo root
   Markdown
   Day 48 Prepare demo: create 3 test
   CVs (strong/average/weak).
   Prepare 2 job descriptions.
   Record or practice live demo
   walkthrough
   demo/ folder with test CVs,
   demo_script.md
   Manual prep
   Day 49 Final git push of everything.
   Verify HuggingFace Spaces is
   live. Tag v3.0 (or v2.1 if V
   skipped). Project complete
   Final live app URL, git tag
   v3.0, all code + notebooks
   committed
   git
   Week 7 deliverables checklist
   ☐ All evaluation metrics from Section 8 computed and documented
   ☐ README.md is complete with setup + run instructions
   ☐ 3 test CVs and 2 test JDs prepared for demo
   ☐ All code committed — no uncommitted files
   ☐ App live on Hugging Face Spaces with final version
   ☐ Git tag v3.0 (or v2.1) pushed
1. Evaluation Metrics
   Each module is evaluated independently. These are the minimum targets. Exceed them where
   possible.
   Module Metric Target (minimum) How to measure
   NER Extraction Precision / Recall /
   F
   F1 ≥ 0.75 on skills Compare against manually
   tagged 50 CVs
   Classification Accuracy +
   Weighted F
   Accuracy ≥ 0.80 5 - fold cross-validation on
   labeled set
   JD Matching Spearman
   Correlation
   ρ ≥ 0.65 Compare rank order vs
   human ranking
   Ranking NDCG@5 NDCG ≥ 0.75 Evaluate top-5 ranked CVs
   per JD
   Suggestions Manual Review Relevance ≥ 4/5 in 10
   test CVs
   Spot-check by team /
   teacher
2. Risk Register
   Risk Likelihood Mitigation
   Scanned PDFs return
   empty text
   High Add pytesseract OCR fallback in Week 2 (Day 10)
   Score distribution all-
   Weak due to rubric
   weights
   Medium Inspect distribution in Week 4 Day 24. Adjust
   rubric_config.json
   NER F1 below 0.75 Medium Switch to Hugging Face fine-tuned NER model in Week
   7 (V3 path)
   Sentence-transformer too
   slow for batch ranking
   Low Precompute and cache embeddings. Use MiniLM (fast)
   not large models
   Streamlit app crashes on
   large PDF
   Low Add file size limit (5MB) and error handler in app.py
   Hugging Face Spaces
   memory limit exceeded
   Medium Use CPU-only models. Do not load all models at once.
   Use @st.cache_resource
   Running out of time (V
   not finished)
   Medium V1 is mandatory. V2 is target. V3 is stretch. Prioritize in
   that order
3. Final Delivery Checklist
   The project is complete when every item below is checked off.
   CODE
   ☐ parse_cv() handles PDF, DOCX, and TXT with OCR fallback
   ☐ extract_all() returns valid CVSchema JSON for 85%+ of test CVs
   ☐ score_cv() reads from rubric_config.json and produces 0–100 score
   ☐ generate_suggestions() returns 3–5 actionable, specific tips
   ☐ rank_cvs() uses the formula: 0.5×semantic + 0.3×skill + 0.2×rubric
   ☐ All unit tests pass: pytest tests/
   MODELS
   ☐ XGBoost classifier trained and saved in models/xgb_classifier.pkl
   ☐ Sentence-transformer loaded and tested (all-MiniLM-L6-v2)
   ☐ Model accuracy ≥ 0.80, NER F1 ≥ 0.75, Spearman ρ ≥ 0.
   APP
   ☐ Streamlit V1: upload CV → score → label → breakdown → suggestions
   ☐ Streamlit V2: upload JD + multiple CVs → ranked list with match %
   ☐ App deployed and live on Hugging Face Spaces
   ☐ App tested with 3 sample CVs (strong/average/weak)
   DOCUMENTATION
   ☐ README.md complete with setup, run, and deploy instructions
   ☐ rubric_config.json documented with comments explaining each field
   ☐ evaluation_report.ipynb with all metric results filled in
   ☐ This project plan document stored in repo root as PROJECT_PLAN.md or .docx
   GIT
   ☐ All code committed — no uncommitted files
   ☐ Clean commit history with meaningful commit messages
   ☐ Tags: v0.1 (foundation), v1.0 (V1 complete), v2.0 (V2 complete), v3.0 (final)
