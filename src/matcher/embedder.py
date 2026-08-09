"""
src/matcher/embedder.py
Sentence-transformer embedder for CV and JD text.

Model is configurable via the environment variable CV_EMBEDDER (defaults to
`models/matcher-confit`, a ConFit-style contrastive fine-tune of
BAAI/bge-small-en-v1.5 that improves matcher Spearman ρ from 0.314 to 0.436 at
identical latency; `CV_EMBEDDER=BAAI/bge-small-en-v1.5` restores the base model).
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Model name may also differ in dimensionality (384 vs 512), so the dim helper
# reads it from the loaded model rather than assuming a constant.
_DEFAULT_MODEL = "models/matcher-confit"


def _model_name() -> str:
    env = os.environ.get("CV_EMBEDDER", "").strip()
    if env:
        return env
    # The ConFit fine-tune lives in a gitignored local dir; on a hosted/CI
    # deploy it is absent, so fall back to its public base model instead of
    # letting SentenceTransformer 401 against HF as if it were a repo id.
    if os.path.isdir(_DEFAULT_MODEL):
        return _DEFAULT_MODEL
    logger.info("Local %s not present; using public BAAI/bge-small-en-v1.5", _DEFAULT_MODEL)
    return "BAAI/bge-small-en-v1.5"


# Lazy-loaded singleton
_model = None
_model_name_loaded = None


def get_embedder():
    global _model, _model_name_loaded
    name = _model_name()
    if _model is not None and _model_name_loaded == name:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name, device="cpu")
        _model_name_loaded = name
        logger.info(f"Embedder loaded: {name}")
    except Exception as e:
        logger.error(f"Failed to load embedder {name}: {e}")
        _model = None
        _model_name_loaded = None
    return _model


def warm_up() -> bool:
    """Eagerly load the embedder so the first JD match has no ~10s cold start.

    Returns True if the embedder loaded successfully. Safe to call at app start;
    it reuses the module-level singleton (idempotent), so calling it again later
    is a no-op. Loads on CPU to match the app's runtime device.
    """
    model = get_embedder()
    if model is None:
        return False
    # Trigger a trivial encode so any lazy tokenizer/weight init is baked in
    # before a real CV/JD pair is scored.
    try:
        model.encode(["warm-up"], normalize_embeddings=True)
        return True
    except Exception as e:
        logger.error(f"Embedder warm-up encode failed: {e}")
        return False


def embed(text: str) -> np.ndarray | None:
    model = get_embedder()
    if model is None:
        return None
    if not text or not text.strip():
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim, dtype=np.float32)
    return model.encode(text, normalize_embeddings=True)


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Batch embed -> (n, 384) matrix."""
    model = get_embedder()
    if model is None:
        return None
    safe_texts = [t if t and t.strip() else " " for t in texts]
    return model.encode(
        safe_texts, normalize_embeddings=True, batch_size=16
    )
