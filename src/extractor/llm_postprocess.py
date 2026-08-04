"""
src/extractor/llm_postprocess.py

Post-process an LLM-generated CVSchema dict so it can be scored fairly against
the rule pipeline. The regex extractor computes `duration_months` itself, but
decoder LLMs (few-shot or fine-tuned) usually emit it as null/0. That would
zero-out the experience score, so we reconstruct months from `start`/`end`.

Only fills `duration_months` when it is missing or zero and a parseable date
range exists; anything already set is left alone.
"""

import re
from datetime import datetime

_MONTH_NAME_TO_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_PRESENT = {"present", "current", "till date", "now"}

_MONTH_YEAR_RE = re.compile(
    r"(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[.,]?\s+(?P<year>\d{4})", re.IGNORECASE)
_YEAR_ONLY_RE = re.compile(r"(?<!\d)(?P<year>\d{4})(?!\d)")


def _parse_start(s: str):
    """Return (year, month) or None for a start date string."""
    if not s:
        return None
    m = _MONTH_YEAR_RE.search(str(s))
    if m:
        return int(m.group("year")), _MONTH_NAME_TO_NUM[m.group("month").lower()]
    y = _YEAR_ONLY_RE.search(str(s))
    if y:
        return int(y.group("year")), 1
    return None


def _parse_end(s: str, now=None):
    """Return (year, month) or the special token 'PRESENT'."""
    if not s:
        return None
    low = str(s).strip().lower()
    if low in _PRESENT:
        return "PRESENT"
    m = _MONTH_YEAR_RE.search(str(s))
    if m:
        return int(m.group("year")), _MONTH_NAME_TO_NUM[m.group("month").lower()]
    y = _YEAR_ONLY_RE.search(str(s))
    if y:
        return int(y.group("year")), 12
    return None


def _duration_months(start: str, end: str, now=None) -> int:
    now = now or datetime.now()
    st = _parse_start(start)
    en = _parse_end(end)
    if st is None:
        return 0
    if en == "PRESENT":
        ex, em = now.year, now.month
    elif en is None:
        return 0
    else:
        ex, em = en
    return max(0, (ex - st[0]) * 12 + (em - st[1]))


def fill_duration_months(cv: dict, now=None) -> dict:
    """Fill in missing duration_months for each experience entry (in place)."""
    for entry in cv.get("experience", []) or []:
        if not isinstance(entry, dict):
            continue
        cur = entry.get("duration_months")
        if cur and int(cur) > 0:
            continue
        start = entry.get("start") or ""
        end = entry.get("end") or ""
        entry["duration_months"] = _duration_months(str(start), str(end), now=now)
    return cv
