# AI Governance Document Q&A

A Retrieval-Augmented Generation (RAG) system for answering questions over AI governance policy documents, built with Streamlit and DeepSeek.

---

## How It Works

### Query pipeline (every question)

```
User question
     │
     ▼
[embed_query]     sentence-transformers all-MiniLM-L6-v2  — local, ~10ms
     │
     ▼
[FAISS search]    top-10 candidates by cosine similarity  — local, ~5ms
     │
     ▼
[Cross-encoder]   re-scores each (query, chunk) pair      — local, ~200ms
     │
     ▼
[DeepSeek API]    top-5 chunks as grounding context       — streams answer
     │
     ▼
Streamed answer with [Source: filename] citations
```

Only the final DeepSeek call touches the network. Embedding and reranking run locally with no API cost.

### Index pipeline (run once, via `ingest.py`)

```
PDF / TXT files
     │
     ▼
[pdfplumber]       extract plain text
     │
     ▼
[chunker]          paragraph-aware split, 512 tokens, 50-token overlap
     │
     ├── [--contextual flag]  call DeepSeek once per chunk to prepend a
     │                        1-2 sentence positional context before embedding
     │                        (Contextual Retrieval — paid once, zero query cost)
     ▼
[embedder]         all-MiniLM-L6-v2 → 384-dim float32 vectors
     │
     ▼
[FAISS IndexFlatIP] saved to index/faiss.index + index/metadata.json
```

### Key Components

| File | Responsibility |
|---|---|
| `rag/chunker.py` | Paragraph-aware splitter — 512 tokens, 50-token overlap |
| `rag/embedder.py` | `all-MiniLM-L6-v2` embeddings — local, no API key |
| `rag/retriever.py` | FAISS `IndexFlatIP` — exact cosine-similarity search |
| `rag/reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-ranks top-10 → top-5 |
| `rag/generator.py` | DeepSeek streaming answer + `generate_chunk_context()` for indexing |
| `ingest.py` | Parse → chunk → (optional contextual enrichment) → embed → save |
| `auth.py` | Session-based bcrypt auth with admin/user roles |
| `app.py` | Streamlit chat UI — auth-gated, reranking toggle |
| `pages/1_Admin.py` | Upload docs, remove docs, rebuild index (standard or contextual) |

### RAG Strategy Decisions

**Contextual Retrieval (Anthropic, 2024)**
Policy documents have numbered articles, cross-references, and defined terms. A bare chunk like *"systems referred to in Article 6 shall..."* loses all meaning without context. We optionally prepend a 1-2 sentence summary ("This excerpt is from the EU AI Act, Article 6, defining high-risk AI categories") before embedding. Cost: ~1 DeepSeek call per chunk at index time. Query-time cost: zero.

**Two-stage retrieval with cross-encoder reranking**
FAISS cosine similarity asks "are these vectors close?" — it's fast but only sees each text independently. A cross-encoder reads the query and passage together and asks "does this passage actually answer this question?" Running it on 10 FAISS candidates adds ~200ms with meaningfully better precision. Both models run locally, so there's no API cost.

**Why not HyDE / Multi-Query / Self-Reflective RAG?**
These all add one or more extra LLM API calls per query. For a demo corpus, the latency cost outweighs the quality gain. Contextual Retrieval solves the same vocabulary-gap problem as HyDE but pays the cost once at index time instead of on every query.

**Other design choices**
- `all-MiniLM-L6-v2` — 384-dim, fast CPU inference, no API key, no per-query cost.
- `IndexFlatIP` (exact search) over approximate methods — corpus is small enough that exact search is fast and avoids quality loss from approximation.
- Temperature 0.1 — maximises factual accuracy, minimises hallucination.
- No LangChain — each layer is implemented directly to keep logic transparent and auditable.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your DeepSeek API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# then edit .env and set DEEPSEEK_API_KEY=your-key-here
```

Get a key at: https://platform.deepseek.com

### 3. Create login credentials

```bash
python setup_credentials.py
```

Default accounts created:

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin@123` | admin |
| `user` | `User@123` | user |

**Change these passwords before deploying.**

### 4. Add documents and build the index

**Option A — Kaggle dataset (recommended):**

You need Kaggle API credentials (`~/.kaggle/kaggle.json`). Then:

```bash
python ingest.py --kaggle
```

**Option B — Local files:**

Place PDF or TXT files in the `data/` folder, then:

```bash
python ingest.py
```

**Option C — Upload via Admin panel:**

Start the app first, sign in as admin, go to Admin Panel → Upload, then Rebuild Index.

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deployment on Streamlit Cloud

1. Push the repo to GitHub (`.env` and `credentials.yaml` are gitignored — do not include them).
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud) pointing to `app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   DEEPSEEK_API_KEY = "your-key-here"
   ```
4. In `app.py` and `pages/1_Admin.py`, uncomment the two lines marked `TO ENABLE FOR DEPLOYMENT`.
5. To handle credentials on Streamlit Cloud, either:
   - Run `setup_credentials.py` locally, commit only the `credentials.yaml` (after removing from `.gitignore`), **or**
   - Store credential hashes directly in Streamlit Secrets and adapt `auth.py` to read from `st.secrets`.

---

## Example Queries

### Query 1
> What principles does the EU AI Act use to classify high-risk AI systems?

**Expected answer shape:** Cites Article 6 / Annex III categories from the EU AI Act document, lists criteria such as use in critical infrastructure, biometric identification, etc.

### Query 2
> How do different governance frameworks address algorithmic transparency and explainability?

**Expected answer shape:** Synthesises references from multiple documents (e.g. OECD AI Principles, EU AI Act, national strategies), citing each source.

### Query 3
> What are the recommended practices for AI incident reporting?

**Expected answer shape:** Grounds answer in governance documents that cover post-deployment monitoring, or states that the corpus does not contain enough information if absent.

---

## Limitations & Known Issues

- **Index must be rebuilt manually** when new documents are added via the Admin panel (click "Rebuild Index").
- **Kaggle credentials required** for auto-download; manual placement in `data/` is the fallback.
- **No query rewriting or HyDE** — complex multi-hop questions may miss relevant chunks. Adding a query-expansion step would improve recall.
- **No re-ranking** — results are ranked by raw cosine similarity. A cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) would improve precision.
- **Credentials stored on disk** — `credentials.yaml` is a simple bcrypt YAML file. For multi-user production use, replace with a proper database.

---

## What I Would Add Next (given more time)

1. **Conversation memory** — pass recent turns to DeepSeek so follow-up questions resolve pronouns correctly ("it", "that regulation", "the previous point").
2. **Metadata filters** — tag chunks by document type (regulation / guideline / national strategy) and let users filter by category before searching.
3. **Evaluation harness** — a small set of labelled QA pairs to measure retrieval recall@k and answer faithfulness automatically (e.g. with RAGAS).
4. **Hierarchical RAG** — build a two-level index (document summaries + chunks) to scale gracefully as the corpus grows beyond a few hundred documents.
5. **Streaming index updates** — currently the index must be fully rebuilt when documents are added. An incremental append to the FAISS index would make admin updates instant.
