"""
src/extractor/grounding.py

Post-extraction verifiability filter for LLM-generated labels.

A fine-tuned generative LLM occasionally fabricates plausible values that are not
actually present in the resume text (skills are the main offender). This module
drops any candidate that cannot be verified against the source text, so the
scorer only sees information that is legitimately grounded.

We verify against the resume TEXT (not the taxonomy): a value invented by the
model is never in the source document, so requiring text-grounding removes the
hallucination even when the value happens to be a well-known / real skill name.

Containment is conservative: a value is kept only if all its tokens appear in a
contiguous run in the text (flexible separators). We prefer to drop a paraphrase
rather than admit an invented value; this biases toward under-claiming, which is
the safe direction for a resume scorer.
"""

import re

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_FLEX = r"[\s,./#+]"


def _tokens(value: str) -> list:
    return _TOKEN_RE.findall(str(value).lower())


def _whole(reg: str) -> str:
    return r"\b" + re.escape(reg) + r"\b"


def _present(skill: str, text_lower: str) -> bool:
    toks = _tokens(skill)
    if not toks:
        return True
    if len(toks) == 1:
        return re.search(_whole(toks[0]), text_lower) is not None
    pattern = _whole(toks[0])
    for t in toks[1:]:
        pattern += _FLEX + "{0,12}" + _whole(t)
    return re.search(pattern, text_lower) is not None


def filter_grounded_skills(skills, text: str) -> list:
    """Return only the skills whose full token run is present in `text`."""
    text_lower = (text or "").lower()
    return [s for s in (skills or []) if s and _present(str(s), text_lower)]


def ground_label(label: dict, text: str) -> dict:
    """Drop ungrounded skills in `label` (in place). Returns `label`."""
    label["skills"] = filter_grounded_skills(label.get("skills", []), text)
    return label