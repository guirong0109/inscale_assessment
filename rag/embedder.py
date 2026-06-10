"""
Thin wrapper around sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast CPU inference (~14k tokens/s)
  - No API key required — runs locally

Vectors are L2-normalised so FAISS inner-product search == cosine similarity.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of strings. Returns float32 array of shape (N, 384)."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns float32 array of shape (384,)."""
    return embed_texts([query])[0]
