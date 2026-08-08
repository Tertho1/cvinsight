"""
src/extractor/bangla_ner.py

Bangla token-classification NER (csebuetnlp/banglabert fine-tune in
models/bangla-ner-v1) used as an additive entity source for the native Bangla
route. It finds skill/title/company/degree/institution/person/project/cert/
language spans in *original Bengali-script* text so the extractor can keep
Bangla-written entities that the Latin-script English engine would miss.

Design mirrors src/extractor/ner_tag.py: windowed token-classification with
lazy torch/transformers imports (so the rule-only fast path never loads torch),
and a graceful empty fallback when the model files are missing.

The label namespaces match the English ner-v1 schema:
PERSON, PROJECT, CERT, DEGREE, INSTITUTION, TITLE, COMPANY, SKILL, LANGUAGE.

Inference is per whitespace-token through the BERT subword tokenizer; spans are
reassembled across tokens with the same B/I/O grouping as ner_tag.py.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

WINDOW = 480
OVERLAP = 40
_MAX_LEN = 512

_DEFAULT_MODEL_PATH = "models/bangla-ner-v1"
_SCRIPT = re.compile(r"[\u0980-\u09FF]")


def load_bangla_ner(adapter=_DEFAULT_MODEL_PATH, device_name="cpu"):
    """Load the Bangla token-classification tagger.
    device_name: 'cpu' | 'gpu'."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model = AutoModelForTokenClassification.from_pretrained(adapter).to(device_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    return model, tokenizer, torch


def _labels_for_words(model, tokenizer, torch, device, words):
    """Run the model on a window and return per-word label ids (0 = O)."""
    from transformers import BatchEncoding
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=_MAX_LEN)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
    preds = torch.argmax(logits[0], dim=-1)
    word_ids = enc.word_ids(batch_index=0)
    wlabs = [0] * len(words)
    prev = None
    for bidx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            wlabs[wid] = preds[bidx].item()
        prev = wid
    return wlabs


def predict_spans(model, tokenizer, text):
    """Return {entity_key: [span strings]} for a Bengali text.

    Entity keys are lowercased (skill, degree, institution, title, company,
    project, cert, language, person). Only Bengali-heavy spans are returned to
    avoid muddying the English path's already-clean Latin spans.
    """
    import torch
    device = next(model.parameters()).device
    words = [m.group() for m in re.finditer(r"\S+", text)]
    if not words:
        return {}

    wlabs = [0] * len(words)
    start = 0
    while start < len(words):
        end = min(start + WINDOW, len(words))
        labs = _labels_for_words(model, tokenizer, torch, device, words[start:end])
        for gi, lab in enumerate(labs):
            i = start + gi
            if wlabs[i] == 0 or lab != 0:
                wlabs[i] = lab
        start = end - OVERLAP if end < len(words) else len(words)

    groups = {}
    cur, curtyp = [], None
    for w, lab in zip(words, wlabs):
        name = model.config.id2label[lab]
        if name == "O":
            if curtyp is not None:
                groups.setdefault(curtyp, []).append(" ".join(cur))
            cur, curtyp = [], None
            continue
        _b, etype = name.split("-", 1)
        k = etype.lower()
        if curtyp is None:
            curtyp, cur = k, [w]
        elif curtyp == k:
            cur.append(w)
        else:
            groups.setdefault(curtyp, []).append(" ".join(cur))
            curtyp, cur = k, [w]
    if curtyp is not None:
        groups.setdefault(curtyp, []).append(" ".join(cur))
    return groups


def _banglish_clean(span):
    """Return a cleaned version of a Latin script span (occasionally mixed into
    Bangla text). Keeps only alphanumerics, dots, + and # characters."""
    return re.sub(r"[^\w.+#]", " ", str(span)).strip()


class BanglaNER:
    """Lazy-loaded Bangla NER tagger singleton (mirrors BanglaSectionClassifier)."""

    def __init__(self, model_path: str = None, device_name: str = "auto"):
        self._model_path = model_path or os.environ.get(
            "BANGLA_NER_MODEL", ""
        ).strip() or _DEFAULT_MODEL_PATH
        self._device = device_name
        self._loaded = False
        self._model = self._tokenizer = self._torch = None

    def _resolve_device(self):
        if self._device != "auto":
            return self._device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._loaded:
            return True
        try:
            if not os.path.isdir(self._model_path):
                return False
            device = self._resolve_device()
            self._model, self._tokenizer, self._torch = load_bangla_ner(
                self._model_path, device)
            self._loaded = True
            return True
        except Exception as e:
            logger.error("Bangla NER load failed: %s", e)
            return False

    @property
    def loaded(self) -> bool:
        return self._load()

    def predict_spans(self, text: str) -> dict:
        if not text or not _SCRIPT.search(text):
            return {}
        if not self._load():
            return {}
        return predict_spans(self._model, self._tokenizer, text)


_ner = None


def get_bangla_ner(model_path: str = None, device_name: str = "auto") -> BanglaNER:
    """Module-level singleton, matching the lazy-load pattern in the English
    path. Device: 'auto' (CUDA if available else CPU), 'cpu', or 'gpu'."""
    global _ner
    if _ner is None:
        _ner = BanglaNER(model_path=model_path, device_name=device_name)
    return _ner