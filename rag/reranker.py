"""
Cross-encoder reranker — second-stage precision filter.

Pipeline position:
  FAISS (fast, approximate) → top-10 candidates
  Cross-encoder (slow, precise) → top-5 final results

Why two stages?
  FAISS scores by cosine similarity between independently-embedded vectors.
  It asks "are these texts in a similar region of vector space?"
  A cross-encoder reads the query AND the passage together in one forward pass,
  so it can model the interaction between them. It asks "does this passage
  actually answer this specific question?"

  Running a cross-encoder over the full index (~10k chunks) would be too slow.
  Running it over 10 FAISS candidates adds ~200ms and meaningfully re-orders them.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~85 MB, CPU-only, trained on MS MARCO passage ranking benchmark.
"""
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score `chunks` against `query` and return the `top_k` highest-scoring.
    Adds a 'rerank_score' key to each returned chunk dict.
    """
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
