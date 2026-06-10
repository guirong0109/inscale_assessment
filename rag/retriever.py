"""
FAISS-backed retriever.

Index type: IndexFlatIP (exact inner-product search on normalised vectors = cosine similarity).
Chosen over approximate methods (IVF, HNSW) because the dataset is small enough
that exact search is fast and avoids the quality loss of approximation.

Files:
  index/faiss.index   — FAISS binary
  index/metadata.json — parallel list of chunk dicts (text, source, chunk_index)
"""
import json
import os

import faiss
import numpy as np

INDEX_PATH = os.path.join("index", "faiss.index")
METADATA_PATH = os.path.join("index", "metadata.json")

_index: faiss.Index | None = None
_metadata: list[dict] | None = None


def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)


def load_index() -> bool:
    """Load FAISS index and metadata from disk. Returns True on success."""
    global _index, _metadata
    if not index_exists():
        return False
    _index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    return True


def reset_index():
    """Force reload on next retrieve() call (used after admin rebuilds index)."""
    global _index, _metadata
    _index = None
    _metadata = None


def retrieve(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Return up to top_k chunks ranked by cosine similarity.
    Each result is a chunk dict augmented with a 'score' key (0–1).
    """
    global _index, _metadata
    if _index is None:
        load_index()
    if _index is None or _index.ntotal == 0:
        return []

    query = query_embedding.reshape(1, -1).astype("float32")
    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(query, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and _metadata is not None:
            chunk = dict(_metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
    return results


def save_index(embeddings: np.ndarray, metadata: list[dict]):
    """Build and persist a new FAISS index from embeddings + metadata."""
    os.makedirs("index", exist_ok=True)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    reset_index()
