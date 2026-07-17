"""
src/matcher/embedder.py
Sentence-transformer embedder for CV and JD text.
Uses multi-qa-MiniLM-L6-cos-v1 — lightweight, CPU-friendly.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_model = None


def get_embedder():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            "multi-qa-MiniLM-L6-cos-v1",
            device="cpu",
        )
        logger.info("Embedder loaded: multi-qa-MiniLM-L6-cos-v1")
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        _model = None
    return _model


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
