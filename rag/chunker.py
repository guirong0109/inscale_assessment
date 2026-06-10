"""
Text chunking with paragraph-aware splitting and token overlap.

Strategy:
  1. Split on paragraph boundaries (double newlines) to preserve semantic units.
  2. If a paragraph alone exceeds max_tokens, fall back to sentence-level splits.
  3. Carry an overlap window of `overlap_tokens` tokens into each new chunk so
     that retrieval never misses a claim that straddles a boundary.
"""
import re
import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _get_overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    """Return the suffix of `parts` that fits within overlap_tokens."""
    tail: list[str] = []
    used = 0
    for part in reversed(parts):
        t = _token_len(part)
        if used + t > overlap_tokens:
            break
        tail.insert(0, part)
        used += t
    return tail


def split_into_chunks(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    source: str = "",
) -> list[dict]:
    """
    Returns a list of chunk dicts:
      {"text": str, "source": str, "chunk_index": int}
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[dict] = []
    current_parts: list[str] = []
    current_tokens = 0

    def flush():
        if current_parts:
            chunks.append(
                {
                    "text": "\n\n".join(current_parts),
                    "source": source,
                    "chunk_index": len(chunks),
                }
            )

    for para in paragraphs:
        para_tokens = _token_len(para)

        # Para is too large on its own — split at sentence level
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = _token_len(sent)
                if current_tokens + sent_tokens > max_tokens and current_parts:
                    flush()
                    overlap = _get_overlap_tail(current_parts, overlap_tokens)
                    current_parts = overlap + [sent]
                    current_tokens = _token_len("\n\n".join(current_parts))
                else:
                    current_parts.append(sent)
                    current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            flush()
            overlap = _get_overlap_tail(current_parts, overlap_tokens)
            current_parts = overlap + [para]
            current_tokens = _token_len("\n\n".join(current_parts))
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    flush()
    return chunks
