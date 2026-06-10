"""
Main Streamlit app — AI Governance Document Q&A

Query pipeline:

  User question
    │
    ▼
  embed_query()  — sentence-transformers, local, ~10ms
    │
    ▼
  FAISS search   — top-10 candidates, local, ~5ms
    │
    ▼
  rerank()       — cross-encoder, local, ~200ms
    │
    ▼
  top-5 chunks passed to DeepSeek  — streams grounded answer with citations

All steps except the final DeepSeek call run locally with no API cost.
Contextual Retrieval improves what is in the index (index-time, not query-time).
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Inject DeepSeek key from Streamlit Secrets (works both locally via
# .streamlit/secrets.toml and on Streamlit Cloud via the Secrets panel).
# Falls back silently if the key is already set via .env.
# ---------------------------------------------------------------------------
try:
    if not os.getenv("DEEPSEEK_API_KEY") and "DEEPSEEK_API_KEY" in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass  # st.secrets unavailable outside Streamlit context (e.g. tests)

from auth import is_admin, logout, require_auth
from rag.embedder import embed_query
from rag.generator import generate_answer
from rag.reranker import rerank
from rag.retriever import index_exists, load_index, retrieve

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Governance Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------
require_auth()

# ---------------------------------------------------------------------------
# Load FAISS index once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading document index...")
def _load_index():
    return load_index()

_load_index()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ AI Governance Q&A")
    st.markdown(f"Signed in as **{st.session_state.get('name', '')}**")
    st.caption(f"Role: {st.session_state.get('role', 'user')}")
    st.divider()

    if is_admin():
        st.page_link("pages/1_Admin.py", label="Admin Panel", icon="⚙️")
        st.divider()

    # ------------------------------------------------------------------
    # Reranking toggle — lets the user observe the quality difference.
    # On by default. Adds ~200ms locally; no API cost.
    # ------------------------------------------------------------------
    use_rerank = st.toggle(
        "Cross-encoder reranking",
        value=True,
        help=(
            "After FAISS retrieves 10 candidates, a cross-encoder re-scores "
            "each (query, passage) pair together and re-orders them by true "
            "relevance. Adds ~200ms. No API call — runs locally."
        ),
    )

    st.divider()

    if st.button("Sign Out", use_container_width=True):
        logout()

    st.divider()
    st.caption(
        "Answers are grounded in the loaded document corpus. "
        "Always verify important claims against the original sources."
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("AI Governance Document Q&A")
st.caption(
    "Ask questions about AI governance policies, regulations, and frameworks. "
    "Answers are grounded in the document corpus with source citations."
)

if not index_exists():
    st.warning(
        "No document index found.\n\n"
        "**To get started:**\n"
        "1. Place PDF/TXT files in the `data/` folder, or use the Admin panel to upload them.\n"
        "2. Run `python ingest.py` in a terminal (or Admin → Rebuild Index)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    score_label = (
                        f"rerank score {src['rerank_score']:.2f}"
                        if "rerank_score" in src
                        else f"similarity {src['score'] * 100:.0f}%"
                    )
                    st.caption(f"📄 **{src['source']}** — {score_label}")
                    display_text = src.get("text", "")
                    if src.get("context_prefix"):
                        display_text = display_text.replace(
                            src["context_prefix"] + "\n\n", "", 1
                        )
                    st.text(display_text[:350] + ("..." if len(display_text) > 350 else ""))
                    st.divider()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
if query := st.chat_input("Ask about AI governance, regulations, or policies..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    with st.status("Searching documents...", expanded=False) as status:
        query_emb = embed_query(query)

        n_candidates = 10 if use_rerank else 5
        chunks = retrieve(query_emb, top_k=n_candidates)

        if use_rerank and chunks:
            chunks = rerank(query, chunks, top_k=5)

        status.update(
            label=f"Found {len(chunks)} relevant excerpt(s).",
            state="complete",
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    with st.chat_message("assistant"):
        if not chunks:
            reply = (
                "No relevant document excerpts were found for your question. "
                "Please try rephrasing or check that documents have been indexed."
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            placeholder = st.empty()
            full_response = ""
            try:
                for token in generate_answer(query, chunks):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except ValueError as e:
                placeholder.error(str(e))
                full_response = str(e)

            with st.expander("Sources", expanded=False):
                for src in chunks:
                    score_label = (
                        f"rerank score {src['rerank_score']:.2f}"
                        if "rerank_score" in src
                        else f"similarity {src['score'] * 100:.0f}%"
                    )
                    st.caption(f"📄 **{src['source']}** — {score_label}")
                    display_text = src.get("text", "")
                    if src.get("context_prefix"):
                        display_text = display_text.replace(
                            src["context_prefix"] + "\n\n", "", 1
                        )
                    st.text(display_text[:350] + ("..." if len(display_text) > 350 else ""))
                    st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": chunks}
            )

# ---------------------------------------------------------------------------
# Clear conversation
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.messages:
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
