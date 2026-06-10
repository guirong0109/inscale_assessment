"""
Ingestion pipeline — run once before starting the Streamlit app.

Steps:
  1. Locate documents in data/ and data/uploaded/ (PDFs, TXT, MD)
  2. Optionally download the Kaggle dataset with --kaggle flag
  3. Extract plain text from each file
  4. Chunk text with paragraph-aware splitter (512 tokens, 50 overlap)
  5. [Optional] Enrich each chunk with a contextual prefix via DeepSeek
     (Contextual Retrieval — use --contextual flag, costs ~1 API call/chunk)
  6. Embed all chunks with sentence-transformers (all-MiniLM-L6-v2)
  7. Build and save a FAISS IndexFlatIP (cosine similarity)

Usage:
  python ingest.py                          # standard index
  python ingest.py --contextual             # + contextual retrieval enrichment
  python ingest.py --kaggle                 # download dataset first
  python ingest.py --kaggle --contextual    # both
  python ingest.py --data-dir /path/to/docs
"""
import argparse
import glob
import os

import numpy as np
import pdfplumber

from rag.chunker import split_into_chunks
from rag.embedder import embed_texts
from rag.retriever import save_index

DATA_DIR = "data"
UPLOADED_DIR = os.path.join(DATA_DIR, "uploaded")
SUPPORTED_EXTENSIONS = ("*.pdf", "*.txt", "*.md", "*.csv")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_pdf(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"    [warn] Could not read {path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def find_documents(dirs: list[str]) -> list[str]:
    paths: set[str] = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for pattern in SUPPORTED_EXTENSIONS:
            paths.update(glob.glob(os.path.join(d, "**", pattern), recursive=True))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_kaggle() -> str | None:
    try:
        import kagglehub
        print("Downloading Kaggle dataset 'umerhaddii/ai-governance-documents-data'...")
        path = kagglehub.dataset_download("umerhaddii/ai-governance-documents-data")
        print(f"  Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  [warn] Kaggle download failed: {e}")
        print(f"  Place documents manually in '{DATA_DIR}/' and re-run.")
        return None


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_index(doc_paths: list[str], contextual: bool = False) -> int:
    """
    Process documents, embed chunks, build FAISS index.

    Args:
        doc_paths:  list of file paths to process
        contextual: if True, enrich each chunk with a contextual prefix before
                    embedding (Contextual Retrieval). Costs ~1 DeepSeek API
                    call per chunk — expect 5–15 min extra for a large corpus.

    Returns the total number of chunks indexed.
    """
    if contextual:
        from rag.generator import generate_chunk_context

    all_chunks: list[dict] = []

    for path in doc_paths:
        name = os.path.basename(path)
        print(f"  → {name}")
        text = extract_text(path)
        if not text.strip():
            print(f"     [skip] no text extracted")
            continue
        chunks = split_into_chunks(text, max_tokens=512, overlap_tokens=50, source=name)
        print(f"     {len(chunks)} chunks", end="")

        if contextual:
            print(" — generating contextual prefixes...", end="", flush=True)
            enriched = 0
            for chunk in chunks:
                try:
                    context_prefix = generate_chunk_context(text, chunk["text"])
                    # Prepend context to chunk text; the original text is preserved
                    # in a separate key so the UI can display the clean excerpt.
                    chunk["context_prefix"] = context_prefix
                    chunk["text"] = f"{context_prefix}\n\n{chunk['text']}"
                    enriched += 1
                except Exception as e:
                    print(f"\n     [warn] context generation failed for chunk: {e}")
            print(f" ({enriched} enriched)")
        else:
            print()

        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nNo text extracted. Nothing to index.")
        return 0

    mode = "contextual" if contextual else "standard"
    print(f"\nEmbedding {len(all_chunks)} chunks [{mode} mode]...")
    texts = [c["text"] for c in all_chunks]
    batch = 64
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        end = min(i + batch, len(texts))
        parts.append(embed_texts(texts[i:end]))
        print(f"  {end}/{len(texts)}")

    embeddings = np.vstack(parts)
    save_index(embeddings, all_chunks)

    print(f"\nDone — {len(all_chunks)} chunks from {len(doc_paths)} documents indexed.")
    print("Index saved to index/faiss.index and index/metadata.json")
    return len(all_chunks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the document search index.")
    parser.add_argument(
        "--kaggle", action="store_true", help="Download dataset from Kaggle first"
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help=f"Local data directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--contextual",
        action="store_true",
        help=(
            "Enrich each chunk with a contextual prefix via DeepSeek before embedding "
            "(Contextual Retrieval). Improves retrieval quality at the cost of "
            "~1 API call per chunk. Requires DEEPSEEK_API_KEY in .env."
        ),
    )
    args = parser.parse_args()

    search_dirs = [args.data_dir, UPLOADED_DIR]

    if args.kaggle:
        kaggle_path = download_kaggle()
        if kaggle_path:
            search_dirs.append(kaggle_path)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(UPLOADED_DIR, exist_ok=True)

    docs = find_documents(search_dirs)
    if not docs:
        print(
            f"No documents found in {search_dirs}.\n"
            "Options:\n"
            "  • Run with --kaggle to auto-download\n"
            "  • Place PDF/TXT files in the data/ folder\n"
            "  • Use the Admin panel in the app to upload files"
        )
        return

    if args.contextual:
        print("Contextual Retrieval mode ON — each chunk will be enriched via DeepSeek.")
        print("This takes longer but produces higher-quality embeddings.\n")

    print(f"Found {len(docs)} document(s):")
    build_index(docs, contextual=args.contextual)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
