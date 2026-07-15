"""
Test actual CV files through the parser + extractor pipeline.

Usage:
  python scripts/test_cv_files.py <folder_path>

Example:
  python scripts/test_cv_files.py demo/
  python scripts/test_cv_files.py C:/Users/Me/Downloads/test_cvs/

Output:
  - Prints extraction results per file
  - Saves results.json to the folder
"""
import os, sys, json, warnings, textwrap, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import pandas as pd
from src.parser.parser import parse_cv
from src.parser.section_splitter import split_sections
from src.extractor.extractor import extract_all
from src.scorer.scorer import score_cv
from src.suggester.suggester import generate_suggestions

def format_cv(cv, indent=4):
    """Pretty-print a CVSchema object."""
    lines = []
    
    def add(key, val):
        if val is None or val == "" or val == [] or val == {}:
            return
        if isinstance(val, list):
            if all(isinstance(v, str) for v in val):
                lines.append(f"{' '*indent}{key}: {', '.join(val[:10])}{'...' if len(val) > 10 else ''}")
            else:
                lines.append(f"{' '*indent}{key}:")
                for i, item in enumerate(val[:5]):
                    if isinstance(item, dict):
                        parts = []
                        for k, v in item.items():
                            if v and v not in ("", None, []):
                                parts.append(f"{k}={v}")
                        lines.append(f"{' '*indent}  [{i+1}] {' | '.join(parts)}")
                    else:
                        lines.append(f"{' '*indent}  [{i+1}] {item}")
                if len(val) > 5:
                    lines.append(f"{' '*indent}  ... ({len(val)} total)")
        elif isinstance(val, dict):
            parts = [f"{k}={v}" for k, v in val.items() if v is not None]
            lines.append(f"{' '*indent}{key}: {' | '.join(parts)}")
        else:
            val_str = str(val).replace('\n', ' ').strip()
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            lines.append(f"{' '*indent}{key}: {val_str}")
    
    add("Name", cv.get("name"))
    add("Email", cv.get("email"))
    add("Phone", cv.get("phone"))
    add("Skills", sorted(cv.get("skills", [])))
    add("Education", cv.get("education", []))
    add("Experience", cv.get("experience", []))
    add("Projects", cv.get("projects", []))
    add("Certifications", cv.get("certifications", []))
    add("Languages", cv.get("languages", []))
    add("Achievements", cv.get("achievements", []))
    add("Leadership", cv.get("leadership", []))
    
    # Scores
    section_scores = cv.get("section_scores", {})
    if section_scores:
        scores = {k: v for k, v in section_scores.items() if v is not None and v > 0}
        if scores:
            total = cv.get("total_score", 0)
            label = cv.get("label", "")
            lines.append(f"{' '*indent}Scores: {scores}")
            lines.append(f"{' '*indent}Total: {total} ({label})")
    
    return "\n".join(lines)


def process_file(filepath):
    """Process a single CV file and return results."""
    filename = os.path.basename(filepath)
    errors = []
    stats = {}
    
    t0 = time.time()
    
    # Step 1: Parse
    try:
        raw_text = parse_cv(filepath)
        if raw_text.strip():
            stats["text_len"] = len(raw_text)
            stats["parse_ok"] = True
        else:
            stats["parse_ok"] = False
            errors.append("Parser returned empty text")
    except Exception as e:
        stats["parse_ok"] = False
        errors.append(f"Parser error: {type(e).__name__}: {e}")
    
    if not stats.get("parse_ok"):
        return {"file": filename, "path": filepath, "ok": False, "errors": errors}
    
    # Step 2: Section split
    try:
        sections = split_sections(raw_text)
        sections_found = {k: bool(v.strip()) for k, v in sections.items()}
        stats["sections_found"] = sum(1 for v in sections_found.values() if v)
        stats["section_details"] = sections_found
    except Exception as e:
        sections = {}
        stats["sections_found"] = 0
        errors.append(f"Section split error: {e}")
    
    # Step 3: Extract
    try:
        cv = extract_all(raw_text, sections=sections)
        stats["extract_ok"] = True
        
        # Count extracted fields
        fields = ["name", "email", "phone", "skills", "education", "experience",
                  "projects", "certifications", "languages", "achievements", "leadership"]
        extracted_fields = {}
        for f in fields:
            val = cv.get(f)
            if val and (isinstance(val, list) or str(val).strip()):
                if isinstance(val, list):
                    extracted_fields[f] = len(val)
                else:
                    extracted_fields[f] = 1
            else:
                extracted_fields[f] = 0
        stats["extracted_fields"] = extracted_fields
        stats["total_extracted"] = sum(extracted_fields.values())
        
    except Exception as e:
        stats["extract_ok"] = False
        errors.append(f"Extractor error: {type(e).__name__}: {e}")
        cv = {}
    
    # Step 4: Score
    try:
        cv = score_cv(cv)
        stats["score"] = cv.get("total_score", 0)
        stats["label"] = cv.get("label", "")
        suggestions = generate_suggestions(cv)
        cv["suggestions"] = suggestions
        stats["suggestions"] = len(suggestions)
    except Exception as e:
        stats["score"] = 0
        stats["label"] = "error"
        errors.append(f"Scorer error: {e}")
    
    elapsed = time.time() - t0
    stats["time_sec"] = round(elapsed, 2)
    
    return {
        "file": filename,
        "path": filepath,
        "ok": True,
        "errors": errors,
        "stats": stats,
        "raw_text_preview": raw_text[:500] if raw_text else "",
        "cv": cv,
        "formatted": format_cv(cv)
    }


def main():
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_cv_files.py <folder_path>")
        print("       python scripts/test_cv_files.py demo/")
        sys.exit(1)
    
    folder = sys.argv[1]
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)
    
    # Find CV files
    extensions = ["*.pdf", "*.docx", "*.doc", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    
    files = sorted(set(files))
    
    if not files:
        print(f"No CV files found in {folder}")
        print("Supported: .pdf, .docx, .doc, .txt")
        sys.exit(1)
    
    print("=" * 80)
    print(f"CV PIPELINE TEST — {len(files)} files found in {folder}")
    print("=" * 80)
    
    all_results = []
    
    for i, filepath in enumerate(files, 1):
        print(f"\n{'#'*80}")
        print(f"  FILE #{i}: {os.path.basename(filepath)}")
        print(f"  Path: {os.path.abspath(filepath)}")
        print(f"{'#'*80}")
        
        result = process_file(filepath)
        all_results.append(result)
        
        if not result["ok"]:
            print(f"\n  FAILED: {result['errors'][0]}")
            continue
        
        s = result["stats"]
        print(f"  Parse:     OK ({s['text_len']} chars)")
        print(f"  Sections:  {s.get('sections_found', 0)} detected")
        print(f"  Extract:   OK ({s.get('total_extracted', 0)} items)")
        print(f"  Score:     {s.get('score', 0)} ({s.get('label', '')})")
        print(f"  Time:      {s.get('time_sec', 0)}s")
        
        print(f"\n  Extracted Fields:")
        for f_name, f_count in s.get("extracted_fields", {}).items():
            icon = "OK" if f_count > 0 else "--"
            val_str = f"{f_count} item(s)" if f_count > 0 else "empty"
            print(f"    [{icon:<2s}] {f_name:<20s} {val_str}")
        
        if result["errors"]:
            print(f"\n  Warnings:")
            for e in result["errors"]:
                print(f"    ! {e}")
        
        print(f"\n  Formatted Output:")
        print(result["formatted"] or "    No extraction data")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    headers = ["File", "Chars", "Sections", "Fields", "Score", "Label", "Time"]
    header_line = " | ".join(h.center(20 if i==0 else 12) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-'*82}")
    
    for r in all_results:
        if r["ok"]:
            s = r["stats"]
            fname = r["file"]
            if len(fname) > 18:
                fname = fname[:15] + "..."
            print(f"  {fname:<20s} {s['text_len']:<12d} {s['sections_found']:<12d} "
                  f"{s['total_extracted']:<8d} {s['score']:<6d} {s['label']:<8s} {s['time_sec']:<6.2f}")
        else:
            fname = r["file"]
            if len(fname) > 18:
                fname = fname[:15] + "..."
            print(f"  {fname:<20s} {'FAILED':<12s} {'':<12s} {'':<8s} {'':<6s} {'':<8s} {'':<6s}")
    
    # Save results
    out_path = os.path.join(folder, "_extraction_results.json")
    # Save a clean version without full CV objects (for easy reading)
    summary = []
    for r in all_results:
        entry = {
            "file": r["file"],
            "ok": r["ok"],
            "errors": r["errors"],
            "stats": r.get("stats", {}),
        }
        # Add formatted output
        if r.get("cv"):
            cv = r["cv"]
            entry["extracted"] = {
                "name": cv.get("name"),
                "email": cv.get("email"),
                "phone": cv.get("phone"),
                "skills": cv.get("skills", []),
                "education": cv.get("education", []),
                "experience": cv.get("experience", []),
                "projects": cv.get("projects", []),
                "certifications": cv.get("certifications", []),
                "languages": cv.get("languages", []),
                "achievements": cv.get("achievements", []),
                "leadership": cv.get("leadership", []),
                "total_score": cv.get("total_score"),
                "label": cv.get("label"),
                "suggestions": cv.get("suggestions", []),
            }
        summary.append(entry)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nDetailed results saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
