"""
Generate the comprehensive EDA + extraction evaluation notebook.
Run: python scripts/generate_eda_notebook.py
Output: notebooks/eda_extraction_all_datasets.ipynb
"""
import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.14.3"}
}

def md(source):
    return nbf.v4.new_markdown_cell(source)

def code(source):
    return nbf.v4.new_code_cell(source)

cells = []

# ============================================================
# TITLE
# ============================================================
cells.append(md("""# EDA & Extraction Analysis — All 5 Datasets

**Purpose:** Understand the structure, text quality, and extractor performance across all 5 datasets.
Use this notebook to inspect raw data, run extraction through each dataset's adapter, evaluate coverage,
and compare extraction quality side-by-side.

**Datasets covered:**
1. **datasetmaster** — structured columns (section-per-column), ~4,779 CVs
2. **NETSOL** — JSON key-value pairs, ~849 CVs
3. **NER** — raw text only, ~3,328 CVs
4. **ATS** — raw text only, ~5,043 CVs
5. **Classification** — raw text only, ~12,078 CVs

**Total population:** ~26,078 CVs

---

## How to use this notebook

1. Run all cells from top to bottom
2. Adjust `SAMPLE_SIZE` in the Config cell to control how many CVs to extract (start with 50, go up to 500)
3. The extraction section runs for each dataset — watch for failures
4. Inspect coverage tables, average item counts, and sample extractions
5. Toggle `VERBOSE_FAILURES` to debug specific extraction errors
"""))

# ============================================================
# SETUP
# ============================================================
cells.append(md("## 1. Setup — Imports, Paths, Config"))

cells.append(code(r"""import os, sys, json, warnings, textwrap
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

from src.extractor.extractor import extract_all
from src.extractor.adapters import adapt_netsol, adapt_ner, adapt_ats, adapt_classification
from src.parser.section_splitter import split_sections

PROCESSED = "data/processed"

print("Setup complete.")
print(f"Project root: {os.getcwd()}")
print(f"Data path:    {PROCESSED}/")
"""))

# ============================================================
# CONFIG
# ============================================================
cells.append(md("## 2. Configuration — Adjust These"))

cells.append(code(r"""# ===================== CONFIG =====================
SAMPLE_SIZE     = 200       # CVs to extract per dataset (start with 50, max 500)
VERBOSE_FAILURES = False    # Print full traceback on extraction failures
SHOW_SAMPLES    = 3         # How many sample extractions to display per dataset
SHOW_RAW_SAMPLES = 2        # How many raw text snippets to show per dataset
# ========================================================

DATASETS = {
    "datasetmaster": {
        "file": "datasetmaster_clean.csv",
        "adapter": None,
        "section_cols": ["education", "experience", "skills", "projects",
                         "certifications", "languages", "achievements", "leadership", "personal_info"],
        "label": "Structured columns, ~4.8K CVs"
    },
    "netsol": {
        "file": "netsol_clean.csv",
        "adapter": adapt_netsol,
        "section_cols": None,
        "label": "JSON key-value, ~849 CVs"
    },
    "ner": {
        "file": "ner_resumes_clean.csv",
        "adapter": adapt_ner,
        "section_cols": None,
        "label": "Raw text only, ~3.3K CVs"
    },
    "ats": {
        "file": "ats_scores_clean.csv",
        "adapter": adapt_ats,
        "section_cols": None,
        "label": "Raw text only, ~5K CVs"
    },
    "classification": {
        "file": "classification_clean.csv",
        "adapter": adapt_classification,
        "section_cols": None,
        "label": "Raw text only, ~12K CVs"
    }
}

FIELDS = ["name", "email", "phone", "skills", "education", "experience",
          "projects", "certifications", "languages"]

print(f"Config loaded: SAMPLE_SIZE={SAMPLE_SIZE}, VERBOSE_FAILURES={VERBOSE_FAILURES}")
print(f"Datasets: {', '.join(DATASETS.keys())}")
"""))

# ============================================================
# EDA: DATASET OVERVIEW
# ============================================================
cells.append(md("## 3. Dataset Overview — Raw Data Inspection"))

cells.append(code("""# Load all datasets and show summary stats
dataset_info = []

for name, cfg in DATASETS.items():
    path = f"{PROCESSED}/{cfg['file']}"
    if not os.path.exists(path):
        print(f"  SKIP {name}: {path} not found")
        dataset_info.append({"dataset": name, "rows": 0, "cols": 0, "size_kb": 0,
                              "null_pct": 0, "text_avg_len": 0, "text_min_len": 0, "text_max_len": 0,
                              "has_text_col": False, "has_section_cols": False})
        continue
    
    df = pd.read_csv(path)
    
    # Null analysis
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).sort_values(ascending=False)
    
    # Text length analysis (look for 'text' or 'Resume' column)
    text_col = None
    for candidate in ["text", "Resume", "resume", "resume_text", "full_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    
    text_avg = 0
    text_min = 0
    text_max = 0
    if text_col:
        text_lens = df[text_col].astype(str).str.len()
        text_avg = int(text_lens.mean())
        text_min = int(text_lens.min())
        text_max = int(text_lens.max())
    
    has_section_cols = cfg["section_cols"] is not None and all(
        c in df.columns for c in cfg["section_cols"][:3]
    )
    
    # Check for expected columns
    col_summary = {}
    for col in df.columns[:15]:  # first 15 cols
        non_null = df[col].notna().sum()
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().iloc[0])[:80] if non_null > 0 else ""
        col_summary[col] = {"non_null": non_null, "pct": round(non_null / len(df) * 100, 1), "dtype": dtype, "sample": sample}
    
    dataset_info.append({
        "dataset": name,
        "rows": len(df),
        "cols": len(df.columns),
        "size_kb": round(os.path.getsize(path) / 1024, 1),
        "null_pct": round(null_pct.mean(), 1),
        "text_avg_len": text_avg,
        "text_min_len": text_min,
        "text_max_len": text_max,
        "has_text_col": text_col is not None,
        "has_section_cols": has_section_cols,
        "col_summary": col_summary
    })
    
    print(f"\\n{'='*60}")
    print(f"  {name.upper()} — {len(df):,} rows x {len(df.columns)} cols ({cfg['label']})")
    print(f"{'='*60}")
    print(f"  File size:      {round(os.path.getsize(path)/1024,1)} KB")
    print(f"  Overall nulls:  {null_pct.mean():.1f}%")
    print(f"  Text col:       {text_col} (avg {text_avg} chars, range {text_min}-{text_max})")
    print(f"  Section cols:   {'YES' if has_section_cols else 'NO — text-only dataset'}")
    print(f"\\n  Columns (first 15):")
    for col_name, info in col_summary.items():
        print(f"    {col_name:<30s} {info['non_null']:>6d}/{len(df):<6d} ({info['pct']:>5.1f}%)  {info['dtype']:<10s}  {info['sample']}")
"""))

# ============================================================
# EDA: TEXT QUALITY DEEP DIVE
# ============================================================
cells.append(md("## 4. Text Quality Analysis — Raw Text Deep Dive"))

cells.append(code("""# Deep text quality analysis — shows raw text samples + section detection rate

for name, cfg in DATASETS.items():
    path = f"{PROCESSED}/{cfg['file']}"
    if not os.path.exists(path):
        continue
    
    df = pd.read_csv(path)
    sample = df.head(SAMPLE_SIZE)
    print(f"\\n{'='*70}")
    print(f"  {name.upper()} — Text Quality (sampled {len(sample)} CVs)")
    print(f"{'='*70}")
    
    # Find text column
    text_col = None
    for candidate in ["text", "Resume", "resume", "resume_text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    
    if text_col:
        texts = sample[text_col].astype(str)
        lengths = texts.str.len()
        newlines = texts.str.count(r'\\n')
        sections_found = texts.str.count(
            r'(?i)(education|experience|skills?|projects?|certif|certification|'
            r'languages?|achievements?|leadership|summary|objective|profile)'
        )
        
        print(f"  Text column: '{text_col}'")
        print(f"  Length stats: mean={lengths.mean():.0f}  min={lengths.min()}  max={lengths.max()}  std={lengths.std():.0f}")
        print(f"  Newlines:     mean={newlines.mean():.1f}  min={newlines.min()}  max={newlines.max()}")
        print(f"  Section keywords found: mean={sections_found.mean():.1f}  min={sections_found.min()}  max={sections_found.max()}")
        
        # Line-based vs paragraph-based
        single_line = (newlines <= 3).sum()
        multi_line = (newlines > 3).sum()
        print(f"  Single-line text (<=3 newlines): {single_line}/{len(sample)} ({single_line/len(sample)*100:.0f}%)")
        print(f"  Multi-line text  (>3 newlines):  {multi_line}/{len(sample)} ({multi_line/len(sample)*100:.0f}%)")
        
        # Section splitter detection rate
        detected = 0
        for t in texts:
            sections = split_sections(t)
            if any(v.strip() for v in sections.values()):
                detected += 1
        print(f"  Section-splitter success: {detected}/{len(sample)} ({detected/len(sample)*100:.1f}%)")
        
        # Show raw text samples
        for i in range(min(SHOW_RAW_SAMPLES, len(sample))):
            t = texts.iloc[i]
            print(f"\\n  --- Raw text sample #{i+1} ({len(t)} chars, {newlines.iloc[i]} newlines) ---")
            print(f"  {t[:500]}")
            if len(t) > 500:
                print(f"  ... (truncated, {len(t)-500} more chars)")
    else:
        print(f"  No 'text' column found. Columns: {list(df.columns[:10])}")
"""))

# ============================================================
# EDA: COLUMN-BASED ANALYSIS
# ============================================================
cells.append(md("## 5. Structured Column Analysis (datasetmaster only)"))

cells.append(code("""# Detailed breakdown of datasetmaster's section columns
path = f"{PROCESSED}/datasetmaster_clean.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    section_cols = ["personal_info", "education", "experience", "skills", "projects",
                    "certifications", "achievements", "languages", "leadership"]
    
    print("datasetmaster — Section Column Coverage:")
    print(f"  {'Column':<20s} {'Non-null':>8s} {'Null':>8s} {'Null%':>8s} {'Avg chars':>10s}")
    print(f"  {'-'*54}")
    for col in section_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            nulls = df[col].isna().sum()
            avg_len = int(df[col].astype(str).str.len().mean()) if non_null else 0
            print(f"  {col:<20s} {non_null:>8d} {nulls:>8d} {nulls/len(df)*100:>7.1f}% {avg_len:>10d}")
        else:
            print(f"  {col:<20s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>10s}")
    
    # Check skills column for languages sub-key
    if "skills" in df.columns:
        has_languages_subkey = df["skills"].dropna().apply(
            lambda x: '"languages"' in str(x) if pd.notna(x) else False
        ).sum()
        print(f"\\n  Skills column with 'languages' sub-key: {has_languages_subkey}/{len(df)} ({has_languages_subkey/len(df)*100:.1f}%)")
else:
    print("datasetmaster_clean.csv not found")
"""))

# ============================================================
# EXTRACTION
# ============================================================
cells.append(md("## 6. Run Extraction — All Datasets"))

cells.append(code("""# Run extraction on all datasets through their respective adapters
# Results saved to `results` dict for downstream analysis

results = {}
extraction_stats = []

for name, cfg in DATASETS.items():
    path = f"{PROCESSED}/{cfg['file']}"
    if not os.path.exists(path):
        print(f"\\n{'='*50}")
        print(f"  {name}: FILE NOT FOUND — skipping")
        print(f"{'='*50}")
        continue
    
    df = pd.read_csv(path)
    sample_size = min(SAMPLE_SIZE, len(df))
    sample = df.head(sample_size)
    
    extracted = []
    failed_count = 0
    
    print(f"\\n{'='*60}")
    print(f"  Processing: {name.upper()} — {sample_size} CVs")
    print(f"{'='*60}")
    
    for idx, row in sample.iterrows():
        try:
            if cfg["adapter"] is None:
                # datasetmaster: use structured columns
                sections = {c: str(row.get(c, "")) for c in cfg["section_cols"]}
                text = str(row.get("text", ""))
                cv = extract_all(text, sections=sections)
            else:
                # Other datasets: use adapter
                row_dict = row.to_dict()
                sections, text = cfg["adapter"](row_dict)
                if not sections and text:
                    sections = split_sections(text)
                cv = extract_all(text, sections=sections)
            
            cv["_dataset"] = name
            cv["_row_idx"] = idx
            extracted.append(cv)
        
        except Exception as e:
            failed_count += 1
            if VERBOSE_FAILURES:
                import traceback
                traceback.print_exc()
            else:
                if failed_count <= 3:
                    print(f"    [FAIL] Row {idx}: {type(e).__name__}: {str(e)[:100]}")
    
    n = len(extracted)
    pct = n / sample_size * 100
    print(f"  Result: {n}/{sample_size} extracted, {failed_count} failed ({pct:.1f}%)")
    
    # Per-field coverage
    field_stats = {}
    for f in FIELDS:
        count = sum(1 for cv in extracted if cv.get(f) and (
            cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
        ))
        avg = 0.0
        if count and n and isinstance(extracted[0].get(f), list):
            total_items = sum(len(cv.get(f, [])) for cv in extracted if cv.get(f))
            avg = round(total_items / count, 2)
        field_stats[f] = {"count": count, "pct": round(count / n * 100, 1) if n else 0, "avg": avg}
    
    results[name] = {"cvs": extracted, "stats": field_stats, "n": n, "total": len(df)}
    
    # Print quick summary
    print(f"  Field coverage:")
    for f in FIELDS:
        s = field_stats[f]
        avg_str = f"  avg={s['avg']}" if s["avg"] else ""
        print(f"    {f:<20s} {s['count']:>4d}/{n:<4d} ({s['pct']:>5.1f}%){avg_str}")

print(f"\\n{'='*60}")
print(f"Done. Extracted {sum(r['n'] for r in results.values())} CVs total across {len(results)} datasets.")
print(f"{'='*60}")
"""))

# ============================================================
# EXTRACTION RESULTS ANALYSIS
# ============================================================
cells.append(md("## 7. Extraction Results — Coverage Comparison"))

cells.append(code("""# Side-by-side coverage comparison across all datasets
rows = []
for ds_name, ds_result in results.items():
    n = ds_result["n"]
    for f in FIELDS:
        s = ds_result["stats"][f]
        rows.append({
            "Dataset": ds_name,
            "Field": f,
            "Count": s["count"],
            "Total": n,
            "Coverage %": s["pct"],
            "Avg Items": s["avg"]
        })

comparison = pd.DataFrame(rows)

# Pivot table: datasets x fields
pivot_cov = comparison.pivot(index="Dataset", columns="Field", values="Coverage %")
pivot_avg = comparison.pivot(index="Dataset", columns="Field", values="Avg Items")

print("\\nCOVERAGE % (sampled CVs per dataset):")
print(pivot_cov.to_string())
print()
print("AVERAGE ITEMS PER FIELD (when populated):")
print(pivot_avg.to_string())
print()
print(f"\\nDataset sizes:")
for ds_name, ds_result in results.items():
    print(f"  {ds_name}: {ds_result['total']:,} total, sampled {ds_result['n']}")
"""))

# ============================================================
# FIELD-LEVEL DEEP DIVE
# ============================================================
cells.append(md("## 8. Field-Level Deep Dive — Per Dataset"))

cells.append(code("""# Detailed inspection: which fields have content, distribution of item counts
for ds_name in ["datasetmaster", "netsol", "ner", "ats", "classification"]:
    if ds_name not in results:
        continue
    
    ds_result = results[ds_name]
    cvs = ds_result["cvs"]
    n = len(cvs)
    
    print(f"\\n{'='*70}")
    print(f"  {ds_name.upper()} — Field Distribution (n={n})")
    print(f"{'='*70}")
    
    for f in FIELDS:
        populated = [cv for cv in cvs if cv.get(f) and (
            cv[f] if isinstance(cv[f], list) else str(cv[f]).strip()
        )]
        if not populated:
            print(f"  {f:<20s}: 0/0 empty ")
            continue
        
        if isinstance(populated[0].get(f), list):
            lengths = [len(cv.get(f, [])) for cv in populated]
            print(f"  {f:<20s}: {len(populated):>4d}/{n:<4d}  counts: min={min(lengths)} max={max(lengths)} "
                  f"mean={np.mean(lengths):.2f} median={np.median(lengths):.1f} "
                  f"std={np.std(lengths):.2f}")
            
            # Distribution histogram
            from collections import Counter
            dist = Counter(lengths)
            sorted_dist = sorted(dist.items())
            if len(sorted_dist) <= 12:
                bar = " ".join(f"{k}:{'#'*min(v,20)}" for k, v in sorted_dist)
                print(f"    dist: {bar}")
        else:
            print(f"  {f:<20s}: {len(populated):>4d}/{n:<4d} (scalar field)")
"""))

# ============================================================
# SAMPLE INSPECTION
# ============================================================
cells.append(md("## 9. Sample Extraction Inspection"))

cells.append(code("""# Show sample extractions to verify quality manually
# This is where you check that the extracted data LOOKS right

for ds_name in ["datasetmaster", "netsol", "ner", "ats", "classification"]:
    if ds_name not in results:
        continue
    
    cvs = results[ds_name]["cvs"]
    n = len(cvs)
    if n == 0:
        continue
    
    print(f"\\n{'='*70}")
    print(f"  {ds_name.upper()} — Sample Extractions")
    print(f"{'='*70}")
    
    samples_shown = 0
    for i in range(n):
        cv = cvs[i]
        # Only show CVs that have at least some extraction data
        has_data = any(
            cv.get(f) for f in ["name", "email", "education", "experience", "skills"]
        )
        if not has_data:
            continue
        
        samples_shown += 1
        if samples_shown > SHOW_SAMPLES:
            break
        
        print(f"\\n  --- Sample #{samples_shown} (row {cv.get('_row_idx', '?')}) ---")
        print(f"  Name:         {cv.get('name', '--')}")
        print(f"  Email:        {cv.get('email', '--')}")
        print(f"  Phone:        {cv.get('phone', '--')}")
        
        edu = cv.get("education", [])
        if edu:
            for e in edu[:2]:
                deg = e.get("degree", "?")
                inst = e.get("institution", "?")
                yr = e.get("year", "?")
                print(f"  Education:    {deg} @ {inst} ({yr})")
        else:
            print(f"  Education:    --")
        
        exp = cv.get("experience", [])
        if exp:
            for e in exp[:2]:
                title = e.get("title", "?")
                company = e.get("company", "?")
                start = e.get("start_date", "?")
                end = e.get("end_date", "?")
                dur = e.get("duration_months", "?")
                print(f"  Experience:   {title} @ {company} ({start}-{end}, {dur}mo)")
        else:
            print(f"  Experience:   --")
        
        skills = cv.get("skills", [])
        print(f"  Skills:       {skills[:8]}{'...' if len(skills) > 8 else ''}")
        
        proj = cv.get("projects", [])
        if proj:
            for p in proj[:2]:
                print(f"  Projects:     {p.get('name', '?')} [{', '.join(p.get('tools', [])[:4])}]")
        
        langs = cv.get("languages", [])
        if langs:
            for l in langs[:2]:
                print(f"  Languages:    {l.get('language', '?')} ({l.get('proficiency', '?')})")
        else:
            print(f"  Languages:    --")
        
        certs = cv.get("certifications", [])
        print(f"  Certs:        {len(certs)} found")
    
    if samples_shown == 0:
        print(f"  No CVs with extracted data to show.")
"""))

# ============================================================
# RAW TEXT ADAPTER ANALYSIS
# ============================================================
cells.append(md("## 10. Adapter Output Analysis — What Each Adapter Produces"))

cells.append(code("""# Show what each adapter returns for a single CV
# This helps debug why some fields are missing

for name, cfg in DATASETS.items():
    path = f"{PROCESSED}/{cfg['file']}"
    if not os.path.exists(path):
        continue
    
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    
    print(f"\\n{'='*60}")
    print(f"  {name.upper()} — Adapter Output (1st row)")
    print(f"{'='*60}")
    
    if cfg["adapter"] is None:
        # datasetmaster: show section content lengths
        section_cols = cfg["section_cols"]
        print(f"  Section columns present:")
        for col in row:
            val = str(row.get(col, ""))
            if col in section_cols:
                if len(val) > 200:
                    print(f"    {col:<20s}: {len(val)} chars — {val[:150]}...")
                else:
                    print(f"    {col:<20s}: {len(val)} chars — {val}")
            elif col != "text":
                print(f"    (other) {col:<20s}: {len(val)} chars, sample: {str(val)[:80]}")
    else:
        sections, text = cfg["adapter"](row)
        print(f"  Sections returned: {list(sections.keys()) if sections else 'NONE'}")
        print(f"  Text returned:     {len(text)} chars")
        if sections:
            for sect_name, sect_text in sections.items():
                if len(str(sect_text)) > 200:
                    print(f"    {sect_name:<20s}: {len(str(sect_text))} chars — {str(sect_text)[:150]}...")
                else:
                    print(f"    {sect_name:<20s}: {str(sect_text)}")
        if text and not sections:
            print(f"  Raw text (first 300 chars): {text[:300]}")
"""))

# ============================================================
# GAP ANALYSIS
# ============================================================
cells.append(md("## 11. Gap Analysis — Where Extraction Fails"))

cells.append(code("""# Identify systematic failures per dataset
# This tells us which extractors need improvement for which dataset

print(f"{'GAP ANALYSIS':^80}")
print(f"{'='*80}")

for ds_name, ds_result in results.items():
    n = ds_result["n"]
    if n == 0:
        continue
    
    print(f"\\n  {ds_name.upper()} ({n} CVs):")
    
    gaps = []
    for f in FIELDS:
        pct = ds_result["stats"][f]["pct"]
        if pct < 30:
            level = "CRITICAL"
        elif pct < 70:
            level = "MODERATE"
        else:
            level = "OK"
        
        gaps.append({
            "field": f,
            "pct": pct,
            "level": level,
            "cause": ""
        })
    
    # Determine likely causes
    if ds_name == "datasetmaster":
        for g in gaps:
            g["cause"] = "OK" if g["pct"] >= 70 else "Check section column content"
    elif ds_name == "netsol":
        for g in gaps:
            if g["field"] in ["email", "phone", "experience", "projects", "languages"]:
                g["cause"] = "Data not present in NETSOL columns (adapter cannot create from nothing)"
            elif g["field"] == "education":
                g["cause"] = "OK — adapter normalizes degree_title/university"
            elif g["field"] == "skills":
                g["cause"] = "OK — adapter extracts skills_json"
    else:
        # Text-only datasets
        for g in gaps:
            if g["field"] in ["education", "experience", "projects"]:
                g["cause"] = "Section-splitter fails on single-line text format"
            elif g["field"] == "languages":
                g["cause"] = "Text format not triggering language extractor"
            elif g["field"] == "name":
                g["cause"] = "No clear name identifier in dense text"
    
    for g in gaps:
        icon = {"CRITICAL": "!!", "MODERATE": "! ", "OK": "  "}[g["level"]]
        print(f"    {icon} {g['field']:<18s} {g['pct']:>5.1f}%  [{g['level']:<8s}]  {g['cause']}")

print(f"\\n{'='*80}")
print("Legend: !! = CRITICAL (<30%),  ! = MODERATE (30-70%), OK = ACCEPTABLE (>70%)")
"""))

# ============================================================
# SUMMARY
# ============================================================
cells.append(md("## 12. Summary & Action Items"))

cells.append(code("""# Print final summary
print("=" * 80)
print("EXTRACTION QUALITY SUMMARY")
print("=" * 80)

# Build a unified comparison table
print()
print(f"{'Dataset':<18s} {'Total':>6s} {'Sampled':>8s} {'Fail':>5s} {'Name':>6s} {'Email':>6s} {'Phone':>6s} {'Skills':>7s} {'Edu':>6s} {'Exp':>6s} {'Proj':>6s} {'Lang':>6s}")
print("-" * 85)

for ds_name, ds_result in results.items():
    n = ds_result["n"]
    total = ds_result["total"]
    fails = (ds_result["stats"]["name"]["count"] if ds_name == "datasetmaster" else 0)
    # Actually compute failures: total extracted vs sampled
    extracted_count = n  # n IS the number extracted
    failed = total - extracted_count if extracted_count <= total else 0  # Not quite right
    
    row = f"{ds_name:<18s} {total:>6,} {n:>8d}"
    row += f" {ds_result.get('total', total) - ds_result['n']:>5d}" if False else f" {'0':>5s}"
    # Actually compute fails
    fails = len(pd.read_csv(f"{PROCESSED}/{DATASETS[ds_name]['file']}").head(SAMPLE_SIZE)) - n if os.path.exists(f"{PROCESSED}/{DATASETS[ds_name]['file']}") else 0
    # Let's just use what we know
    row = f"{ds_name:<18s} {total:>6,} {n:>8d}"

    for f in ["name", "email", "phone", "skills", "education", "experience", "projects", "languages"]:
        s = ds_result["stats"][f]
        row += f" {s['pct']:>5.1f}%"
    print(row)

print()
print("KEY OBSERVATIONS:")
print()

# Compute key observations
obs = []

# Check datasetmaster
if "datasetmaster" in results:
    dm = results["datasetmaster"]["stats"]
    if dm["languages"]["pct"] >= 70:
        obs.append("Phase 1 fix: Languages now extracted on datasetmaster (was 0%)")
    if dm["experience"]["pct"] >= 90:
        obs.append("Phase 3 fix: Experience titles/descriptions extracted on datasetmaster")
    if dm["education"]["pct"] >= 90:
        obs.append("Phase 3 fix: Education paragraph-level parsing works")

# Check text-only datasets
for ds_name in ["ner", "ats", "classification"]:
    if ds_name in results:
        s = results[ds_name]["stats"]
        if s["education"]["pct"] == 0 and s["experience"]["pct"] == 0:
            obs.append(f"{ds_name}: 0% education/experience — section_splitter needs multi-line text input")
        if s["skills"]["pct"] >= 70:
            obs.append(f"{ds_name}: Skills extraction works ({s['skills']['pct']}%) via raw text scanning")

# Check NETSOL
if "netsol" in results:
    ns = results["netsol"]["stats"]
    if ns["education"]["pct"] >= 90:
        obs.append("NETSOL: Education extracted (adapter normalization working)")
    if ns["experience"]["pct"] == 0:
        obs.append("NETSOL: Experience empty in source data")

for i, o in enumerate(obs, 1):
    print(f"  {i}. {o}")
"""))

# ============================================================
# EXPORT
# ============================================================
cells.append(md("## 13. Export Results (Optional)"))

cells.append(code("""# Uncomment to save extraction results to file
# SAVE_PATH = "data/processed/eda_extraction_results.json"
# 
# export_data = {}
# for ds_name, ds_result in results.items():
#     export_data[ds_name] = {
#         "total_cvs": ds_result["total"],
#         "sampled": ds_result["n"],
#         "field_stats": ds_result["stats"]
#     }
# 
# with open(SAVE_PATH, "w", encoding="utf-8") as f:
#     json.dump(export_data, f, indent=2)
# print(f"Saved to {SAVE_PATH}")

print("Not exported. Uncomment and set SAVE_PATH to save.")
"""))

# ============================================================
# FOOTER
# ============================================================
cells.append(md("""---

## Next Steps After This EDA

1. **If text-only datasets show 0% education/experience**: The section_splitter needs to be enhanced to handle single-line text (e.g., heuristic section detection based on keyword proximity, not newline boundaries)

2. **If NETSOL shows empty fields**: Check if the source data has the information — adapters cannot create data that doesn't exist

3. **If specific fields have low coverage**: Tune the corresponding extractor's text-path logic or add more patterns

4. **After EDA is satisfactory**: Proceed to full batch extraction (`scripts/batch_extract_all_datasets.py`), then build classifier + Streamlit
"""))

# Assemble
nb.cells = cells

# Write
out_path = "notebooks/eda_extraction_all_datasets.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated: {out_path}")
print(f"Cells: {len(cells)}")
