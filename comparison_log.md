# Manual (human-read) vs Pipeline Comparison Log

Reading method: For each demo CV, I read the raw text with an INDEPENDENT extractor
(pypdf) NOT the project's parser (pdfplumber), then diff (a) parser vs independent read
(for parser errors), and (b) manual rubric scoring vs pipeline `score_cv(extract_all(...))`.
All manual scoring uses identical config/rubric_config.json.

---

## CV 1 — resume_02_rahul_verma.pdf (Rahul Verma, Data Analyst)

- Parser: pdfplumber === side read (pypdf). No parser text loss.
- Manual score: 52 (exp 14, proj 8, skills 20, edu 10, cert 0, lang 0, lead 0)
- Pipeline score: 64 (exp 18, proj 16, skills 20, edu 10, cert 0, lang 0, lead 0)
- Δ = +12 (pipeline higher)

FINDINGS:
1. [EXTRACTOR] **Project over-split** — description bullet `Built a simple REST API…`
   treated as a SECOND project. 1 project → 2 counted. Projects 8→16. (8 in manual vs 16 pipeline)
2. [EXTRACTOR] **Company pipe bug** — `Analytics Corp | Bangalore |` → location string
   leaked/included in `company` field. Parser output correctly, extractor mis-split.
3. [EXTRACTOR] **"Present" date overestimate** — `March 2022 - Present` → computed
   53 months (Present = ~2026). Manual used the literal "2 years" → score 14 vs 18.
4. [EXTRACTOR] **Education field** — `Bachelor of Science in Statistics` → field shows "Science"
   (minor, no score impact).

---

## CV 2 — resume_03_ananya_patel.pdf (Ananya Patel, Frontend Dev)

- Parser: pdfplumber == pypdf read. No parser text loss.
- Manual score: 64 (exp 14, proj 20, skills 20, edu 10, cert 0, lang 0, lead 0)
- Pipeline score: 56 (exp 18, proj 8, skills 18, edu 12, cert 0, lang 0, lead 0)
- Δ = -8 (pipeline lower)

FINDINGS:
1. [EXTRACTOR] **Projects collapsed (undercount)** — 3 single-line portfolio projects
   (`E-commerce Dashboard`, `Personal Blog`, `Weather Dashboard`) became ONE project;
   the other two lines were stuffed into the first project's `description`.
   Projects 20→8. Opposite bug from CV1 (CV1 split 1→2; CV2 merged 3→1).
2. [EXTRACTOR] **Skills under-extraction** — only 9 caught; missed HTML5, CSS3, SCSS,
   Storybook, Webpack, Store, Cypress, Jest, Vue. Leaked `next.js`/`redux` from portfolio line.
   Skills 20→18.
3. [SCORER] **GPA 10-scale vs 4-scale blunder** — `CGPA 8.2/10` read as `8.2`, and the scorer
   compares it against `gpa_bonus_threshold: 3.5` (4.0 scale) → gave false +2 bonus. Should be
   ~3.28 on 4.0, no bonus. Education 10→12.
4. [EXTRACTOR] **Company pipe** at end (`DesignStudio Web Agency | Pune |`).
5. Experience "Present" overestimate again: Aug 2022→45 months (pipeline 18) vs "2+ years" (manual 14).

---

## CV 3 — resume_04_vikram_singh.pdf (Vikram Singh, ex-mech → backend; 2-page PDF)

- Parser: pdfplumber == pypdf (both pages). No parser text loss.
- Manual score: 66 (exp 18, proj 16, skills 20, edu 10, cert 0, lang 0, **lead 2**)
- Pipeline score: 68 (exp 18, proj 20, skills 20, edu 10, cert 0, lang 0, **lead 0**)
- Δ = +2 (pipeline higher)

FINDINGS:
1. [EXTRACTOR] **Project over-split AGAIN** — 2 real projects became 3. The E-Commerce
   capstone's bullet `Built complete REST API with Django REST Framework` split into its own
   project. Projects 16→20. Same recurring bug as CV1.
2. [EXTRACTOR] **Leadership missed** — `Led team of 5 technicians` is a clear leadership role,
   but `leadership` = []. Extractors only grab leadership from a "leadership" heading, not from
   experience bullets. Leadership 2 vs 0.
3. [EXTRACTOR] **[Career Break] treated as work experience** — `[Career Break: Oct 2021 - Dec 2023]`
   parsed as experience entry titled `"[Career Break` with 22 months. Gap counted as employment.
4. [EXTRACTOR] **Education polluted by bootcamp line** — `Masai School | 2003 | Focus: Python...`
   → spurious education entry `institution: "Focus: Python"`, empty degree. (No score impact.)
5. [EXTRACTOR] Company pipe bug again.

NOTE: scores close (66 vs 68) because project-inflation (+4) & missed-leadership (-2) cancel.

## CV 4 — pro-cv-template-burgundy.docx (Denice Harris, Associate Professor/academic)

- Parser: independent XML read == pipeline text (both unfold the DOCX table cells to lines). No text lost.
- It's an academic CV → the tech-centric rubric inherently scores it low on skills/projects.
- Manual score: ~40-44 (edu 15, exp ~"indeterminate" 15-18, proj 0, skills 0, lang 5, lead 2)
- Pipeline score: **15** (edu 15, everything else 0, exp 0, missing name)
- Δ = pipeline far lower.

FINDINGS (severest so far):
1. [EXTRACTOR] **Nothing** — `name` is EMPTY, email picked but no name. Clearly present at top in the DOCX.
2. [EXTRACTOR] **No experience extracted at all** — this is a professor with Teaching Experience,
   Research Experience. The extractor only matches a "Work Experience"/"Experience" heading, so
   "Teaching Experience" and "Research Experience" are dropped. `experience=[]`.
3. [EXTRACTOR] **Languages missed** — `Spanish`, `French` are in the Skills section; `languages=[]`.
4. [EXTRACTOR] **Education mismatched** — only 1 of 3 degrees kept; `institution` shows a THESIS line
   (`Supply Chain Management:....`) not the school; MBA/BBA dropped.
5. [DATE] **20XX placeholder dates** — CV uses `20XX` everywhere; date parser can't compute years,
   so `duration=0`/none even for the current professor role.

Takeaway: DOCX tables + non-standard headings + academic style break the NER heuristics hard.
A LoRA/LLM extractor would help a lot on this type.

---

## CV 5 — senior_python_dev.txt (John Doe, Senior Python/backend)

- Parser: plain text, no loss.
- Manual: 72 (exp 18, proj 16, skills 20, edu 12, cert 4, lang 0, lead 2) — STRONG
- Pipeline: 64 (exp 22, proj 8, skills 20, edu 12, cert 2, lang 0, lead 0) — AVERAGE
- Δ = -8 (pipeline lower)

FINDINGS:
1. [EXTRACTOR] **Projects collapsed AGAIN** — 2 projects merged into 1: project 2's description got
   stuffed into project 1's fields (tools `python`, `fastapi`). proj 16→8.
2. [EXTRACTOR] **Second certification missed** — only `AWS Certified Developer` kept; `Python Institute
   PCPP` dropped (PCPP not in cert regex?). certs 4→2.
3. [EXTRACTOR] **Mentoring not leadership** — `Mentored team of 3 junior developers` not counted. 2→0.
4. [EXTRACTOR/DATE] **Company pipe bug again** — `TechCorp Inc. | San Francisco, CA |` location
   swallowed. Recurring in EVERY CV.
5. [DATE] **"Present"→2026 again** — senior role 55 months vs claimed 4+ yrs; exp 18→22.
6. [EXTRACTOR] Minor: `end` shows `2021`/`2020` not `Dec 2021`/`May 2020` (month dropped on 2 roles);
   skills noisy-extra `apache` (from 'Apache Kafka'), `s3`, `pair programming`.

So even a very clean CV still loses:
  - pipeline 64 (Average) vs manual 72 (Strong) → worst missing: projects & second cert & leadership.

---

## CV 6 — junior_dev.txt (Sarah Chen, entry-level / new grad)

- Manual: 52 (edu 10, exp 2 [intern ~3mo], proj 16, skills 20, cert 0, lang 0, lead 0) — AVERAGE
- Pipeline: **50** (edu 10, exp 0, proj 16, skills 20, cert 0, lang 0, lead 4) — AVERAGE
- Δ = -2 (small, but errors are in OPPOSITE directions & partially cancel)

FINDINGS:
1. [EXTRACTOR] **"INTERNSHIP EXPERIENCE" not parsed as experience** — heading not in the recognized
   set, so the intern role is dropped. `experience=[]`, exp 2→0.
2. [EXTRACTOR] **"ACTIVITIES" → false leadership** — `Member, Women in Tech Club` & `Volunteer,
   Code.org` counted as leadership roles (2×2=4). Being a club member/volunteer is NOT a leadership
   role. lead 0→4 (false positive).
   - Net: internship miss (-2) hidden by leadership false-positive (+4).
3. Projects parsed correctly this time (2 projects) ✓.
4. Education correct + GPA 3.4<3.5 no bonus ✓.

Takeaway: heading-sensitive extractor fails on "INTERNSHIP EXPERIENCE", over-tags "ACTIVITIES" as
leadership. False-positive + False-negative slack mostly cancel.

---

## CV 7 — Rebecca_Software or Computational Roles.docx (Rebecca Smith, CS undergrad)

- Manual: 58 (edu 10, exp 14 [TA+SWE intern ~2yr], proj 16, skills 12, cert 0, lang 4, lead 2) — AVERAGE
- Pipeline: **53** (edu 10, exp 18, proj 0, skills 20, cert 0, lang 0, lead 5) — AVERAGE
- Δ = -5, but internal components are wildly different.

FINDINGS:
1. [EXTRACTOR] **"PROJECT HIGHLIGHTS" heading NOT parsed as projects** → the 2 real projects
   (Snake Game, Courseable) got dumped into/around `experience`. `projects=[]` (proj 16→0), and 2
   spurious "experience" entries (`Courseable Application (Java)` listed as a job title with company
   `each Computer Science`).
2. [EXTRACTOR] **"Spoken Languages" not picked up** — Mandarin & English under skills not read as
   languages (`languages=[]`, lang 4→0).
3. [EXTRACTOR] **Skills over-detected from wrong scope** — pulled `data analysis`, `ios`, `android`,
   `leadership` from the Extracurricular/Projects text, not the skills list (skills still 20, but
   inflated; manual ~12).
4. [EXTRACTOR] **Education institution polluted** — `...at Urbana-Champaign      Expected May 2024` →
   "Expected" leaked into the institution string.
5. [DATE] "Present"→2026 again (TA = 48 months, exp inflated 14→18).
6. [EXTR] company pipes bad again.

Takeaway: heading "PROJECT HIGHLIGHTS" + a real "Languages" subsection break both projects & language
detection; projects bleed into experience causing double-count.

---

## CV 8 — ocrtest.pdf (VIKRAM SINGH — 2nd render, scanned/raster variant)

- This is the SAME resume as CV 3 but re-rendered as a scanned-style PDF.
- Manual: ~66 (same as resume_04). Pipeline: **68** (edu 10, exp 18, proj 20, skills 20) — AVERAGE (Δ+2).
- NOTE: raster + embedded-text compare identical to `resume_04`. Parser still read it (pdfminer text
  present). So Vikram's landing results hold regardless of render source.

FINDINGS (regression of the same theme as CV 3):
1. [EXTRACTOR] **Career Break again as a "job"** — `[Career Break: Oct 2021 - Dec 2023)` → title `[Career
   Break`, 22 months. Also company=`Hyderabad |` on the freelance role (pipe bug again).
2. [EXTRACTOR] **Project over-split again** — 2 projects → 3 (Task Scheduler split in two). proj 20.
3. [EXTRACTOR] **Leadership missed** — `Led team of 5 technicians...` → `leadership=[]` (0 vs 2).
4. [EXTRACTOR] **Education polluted by bootcamp** — `institution="Focus: Python"` (2023), + Bachelor OK.
5. [EXTRACTOR] Skills noise: `react`, `data analysis` pulled from non-skill context; still 20.

IMPORTANT: `[Career Break: ...)` — the mixed bracket `(` at the end + the `]` start — plus "Septernber"
typo come through the OCR/scanner path and still confuse the parser exactly as the original PDF did.

---

## CV 9 — srbhr_repo_barry_allen_fe.pdf (Barry Allen, Front-End @ Google — real scraped repo)
- Manual: 58 (edu 12 [B.Tech IIT GPA 9.5/10], exp 18 [SWE @ Google Jun2022-Present ~4yr], proj 8
  [1 project], skills 20, cert 0, lang 0, lead 0) — AVERAGE
- Pipeline: **40** (edu 12, **exp 0**, proj 8, skills 20, cert 0, lang 0, lead 0) — **WEAK**
- Δ = **-18**

FINDINGS:
1. [BLOCKING] **The whole Experience section is skipped** — `Software Engineer (Front-End), Google,
   Mountain View, CA, USA / June 2022 - Present` gets NONE of it. `experience=[]`, exp 18→0.
   Cause suspects: this PDF renders the date "June 2022 – Present" with "Present" on its own line and
   the company on the TITLE line (not a "Company | City | date" single line). The role title/company/
   date detection can't handle this split → the whole section silently dropped → ~18 pts wiped.
   - This is the SAME class of bug as CV7 (responsibility bleeding) but here it's a full drop instead
     of a wrong heading: either the experience skips or projects swallowed it.
2. This single loss changes the LABEL: manual AVERAGE → pipeline WEAK (40 < 50).
3. Education GPA 9.5/10 correctly awarded +2 (same as manual, since 9.5≥3.5).

Takeaway: experience-fixing is the #1 priority. Two different real resumes (MSc PhD academic + this
FE) had ZERO experience because of date-placement/heading deviations. This is worse than the
project/leadership issues.

---

## CV 10 — priya_dwivedi_repo_MathewElliot.docx (Mathew Eliot, Senior Web Developer — real repo)
- Manual: ~36-40 (edu 10, exp 14 [Web Dev 2015-2019 ~3.7yr], proj 0, skills ~10, cert 2, lang 0,
  lead 0) — WEAK
- Pipeline: **42** (edu 10, exp 14, proj 0, skills 14, cert **4**, lang 0, lead 0) — WEAK
- δ = +2..+6 (pipeline slightly higher, mostly from duplicated cert)

FINDINGS:
1. [EXTRACTOR] **Duplicate-resume inflation on Certifications** — the whole CV body is duplicated
   (2 identical copies). Experience & Education are deduped OK, but **Certifications is NOT** → the
   PHP Framework cert counted twice (certs 2→4). Inconsistent dedup across sections.
2. [EXTRACTOR] `field` degraded to `Science` (should be "Computer Information Systems").
3. [EXTRACTOR] role title gained trailing `-` (`Web Developer -`) and `company` = `Develop` (bad split).
4. Manual/pipeline both WEAK — consistent label.

---

# OVERALL SUMMARY (CV 1-10, manual-vs-pipeline)

| # | CV | Manual | Pipe | Δ | label manual→pipe |
|---|----|--------|------|---|-------------------|
| 1 | resume_02_rahul | 52 | 64 | +12 | Avg→Avg |
| 2 | resume_03_ananya | 64 | 56 | -8 | Avg→Avg |
| 3 | resume_04_vikram | 66 | 68 | +2 | Avg→Avg |
| 4 | pro-cv-template-burgundy | 42 | 15 | -27 | Avg→Weak |
| 5 | senior_python_dev.txt | 72 | 64 | -8 | Strong→Avg |
| 6 | junior_dev.txt | 52 | 50 | -2 | Avg→Avg |
| 7 | Rebecca...roles.docx | 58 | 53 | -5 | Avg→Avg |
| 8 | ocrtest.pdf | 66 | 68 | +2 | Avg→Avg |
| 9 | srbhr_barry_allen_fe | 58 | 40 | -18 | Avg→Weak |
| 10 | MathewElliot.docx | 36 | 42 | +6 | Weak→Weak |

## Top recurring failure classes (by impact)
1. **EXPERIENCE completely lost** (CV4, CV9): when role/company/date aren't on the expected single
   line, or the heading differs, the WHOLE section drops → 0. Most damage (CV9 -18 flips label to
   Weak; CV4 -27).
2. **Company/location pipe bug** (CV1,2,3,5): `Company | City |` → location+pipe leaked into `company`.
3. **"Present"→current-year overestimate** (CV1,2,3,5,7): inflates experience by assuming 2026.
4. **Projects**: collapsed to 1 (CV2,5) OR over-split into extras (CV1,3,8) — unstable.
5. **Leadership**: missed in exp bullets "led team" (CV3,8); false-positive on "Activities" (CV6).
6. **GPA scale** (CV2): 10-scale (8.2/10) compared as-is to a 4.0 bonus threshold.
7. **Languages**: only caught under a `languages` heading; "Spoken Languages"/skills language line
   missed (CV4, CV7).
8. **Education pollution**: cert/bootcamp line read as `institution` (CV3, CV7, CV8).
9. **Duplication**: docx copy-paste inflates Certifications (CV10).
10. **Career Break** parsed as a job (CV3, CV8).

## Fix priorities
- EXPERIENCE parsing robustness (multi-line dates, company-on-title line, heading synonyms) — highest.
- Dedup in every section (esp. Certifications) for duplicated docx.
- Reconsider "Present" anchor & 10-vs-4 GPA normalization.
- Leadership: detect "led/mentored" in experience bullets; don't treat Activities as leadership.

---

# FIXES APPLIED + RE-RUN (Tier 1 & Tier 2)

Code changed (no push):
- `src/parser/section_splitter.py`: added heading aliases (`teaching experience`,
  `research experience`, `employment experience`, `professional roles`, `work placements`).
- `src/extractor/experience_extractor.py`: lenient "[Present on next line]" date fallback
  (`_find_date_range_permissive`); `_clean_company` strips `Company | location |` pipes.
- `src/extractor/extractor.py`: cert dedup by name; languages fallback from the skills section;
  "led/mentored team" detection in experience bullets → leadership.
- `src/extractor/misc_extractor.py`: leadership filter to drop Member/Volunteer lines.
- `src/scorer/section_scorers.py`: GPA 10→4.0-scale normalization before bonus threshold.

RE-RUN (old → new, manual):
 1 R:   64 → 64  (m=52)   company pipe fixed (Analytics Corp)
 2 A:   56 → 54  (m=64)   GPA fix edu 12→10 (removed false +2) ✓
 3 V:   68 → 68  (m=66)   company fixed (Self-Employed)
 4:     15 → 19  (m≈42)   lang 0→4 (Spanish/French); exp still 0 (20XX dates)
 5:     64 → 66  (m=72)   lead 0→2 (mentored); company fixed
 6:     50 → 46  (m=52)   lead 4→0 (member/volunteer false-) correct
 7:     53 → 53  (m=58)   lang 0→-4? (still 0; not verified)
 8:     68 → 68  (m=66)   replicate
 9:     40 → 58  (m=58)   exp 0→18 (permissive date) BIG WIN ✅
10:     42 → 40  (m≈36)   cert 4→2 (dedup) ✅

Net: the two worst single failures were fixed —
  * CV9 experience was dropped → now 58 == manual   ✅
  * company/location pipe bug removed across CVs     ✅
Section fixes working: GPA (CV2), cert-dedup (CV10), languages (CV4), leadership (CV6 member/volunteer), leadership (CV5 mentored), exp date.

Still-open (need structure-aware/LLM pass, not simple regex):
  * CV4 academic 20XX placeholder dates → exp stays 0.
  * projects collapse (CV2,CV5) & over-split (CV1,CV3) — net-unstable.
  * PCPP cert missed in CV5 (cert keywords lack PCPP/PPCP).
  * non-parsed "INTERNSHIP ... Summer 2023" seasonal dates (CV6).
  * "Spoken Languages" only when languages section empty (CV7 still 0).

All 361 unit tests pass after changes.
---