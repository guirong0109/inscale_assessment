"""
Answer generation and contextual chunk enrichment via DeepSeek API.

Functions:
  generate_answer()        — streams the final answer to the user
  generate_chunk_context() — called at index time (ingest.py --contextual)
                             to prepend a document-position summary to each
                             chunk before embedding (Contextual Retrieval)

API key resolution order:
  1. DEEPSEEK_API_KEY environment variable  (set by .env locally, or injected
     from st.secrets in app.py at startup for Streamlit Cloud deployment)
  2. Raises ValueError if not found — prevents silent failures.
"""
import os
from openai import OpenAI

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """You are an expert assistant specialising in AI governance, \
policy, and regulation.

Your answers must be grounded in the document excerpts provided by the user.

Rules:
- Use ONLY the provided excerpts as your source of facts.
- After each key claim, cite the source document in [Source: <filename>] format.
- If the excerpts do not contain enough information to answer, say clearly:
  "The provided documents do not contain sufficient information to answer this question."
- Be concise, accurate, and professional.
- Do not speculate or add information not present in the context."""


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Add it to your .env file (local) or Streamlit secrets (production)."
        )
    return OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        text = chunk.get("text", "").strip()
        parts.append(f"[Excerpt {i} — Source: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]):
    """
    Yields streamed text tokens from DeepSeek.

    Usage:
        for token in generate_answer(question, chunks):
            buffer += token
    """
    client = _get_client()
    context = _format_context(chunks)

    user_message = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {question}"
    )

    stream = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
        temperature=0.1,
        max_tokens=1024,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Contextual Retrieval — called during ingestion, not at query time
# ---------------------------------------------------------------------------

_CONTEXT_SYSTEM = (
    "You situate document chunks for a search index. "
    "Given a document and one chunk from it, output 1–2 sentences that describe "
    "where this chunk sits in the document and what specific topic it covers. "
    "Output ONLY those sentences — no labels, no preamble."
)


def generate_chunk_context(document_text: str, chunk_text: str) -> str:
    """
    Generate a short positional context prefix for a chunk.

    This prefix is prepended to the chunk text BEFORE embedding so that
    each vector carries document-level meaning alongside local content.
    Technique: Anthropic Contextual Retrieval (2024).

    Called once per chunk during `python ingest.py --contextual`.
    Not called at query time.

    Args:
        document_text: Full (or truncated) text of the source document.
        chunk_text:    The chunk whose context we are generating.

    Returns:
        1–2 sentence context string, e.g.:
        "This excerpt is from the EU AI Act, Article 9, which defines
         risk management obligations for high-risk AI systems."
    """
    client = _get_client()
    # Truncate document to ~3 000 tokens to stay well within model limits
    doc_excerpt = document_text[:12_000]
    if len(document_text) > 12_000:
        doc_excerpt += "\n[Document truncated for brevity]"

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _CONTEXT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"<document>\n{doc_excerpt}\n</document>\n\n"
                    f"<chunk>\n{chunk_text}\n</chunk>"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=80,
        stream=False,
    )
    return response.choices[0].message.content.strip()
