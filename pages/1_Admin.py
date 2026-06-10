"""
Admin Panel — restricted to users with role='admin'.

Features:
  1. View all currently indexed documents
  2. Remove a document from disk
  3. Upload new PDF/TXT/MD files
  4. Rebuild the FAISS index from all documents on disk
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Inject DeepSeek key from Streamlit Secrets (mirrors app.py).
# ---------------------------------------------------------------------------
try:
    if not os.getenv("DEEPSEEK_API_KEY") and "DEEPSEEK_API_KEY" in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

from auth import require_admin, require_auth
from ingest import DATA_DIR, UPLOADED_DIR, build_index, find_documents

st.set_page_config(
    page_title="Admin Panel — AI Governance Q&A",
    page_icon="⚙️",
    layout="wide",
)

# Auth guard — must be admin
require_auth()
require_admin()

# ---------------------------------------------------------------------------
# Sidebar (mirrors main app)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Admin Panel")
    st.markdown(f"Signed in as **{st.session_state.get('name', '')}**")
    st.divider()
    st.page_link("app.py", label="Back to Chat", icon="💬")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Admin Panel")
st.caption("Manage the document corpus and rebuild the search index.")
st.divider()

os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Section 1: Current documents
# ---------------------------------------------------------------------------
st.subheader("1. Current Documents")

all_docs = find_documents([DATA_DIR, UPLOADED_DIR])

if not all_docs:
    st.info(
        "No documents found. Upload files below or place them in the `data/` folder."
    )
else:
    st.caption(f"{len(all_docs)} document(s) on disk.")
    for doc_path in all_docs:
        col_name, col_size, col_action = st.columns([5, 1, 1])
        name = os.path.basename(doc_path)
        size_kb = os.path.getsize(doc_path) // 1024
        col_name.write(f"📄 {name}")
        col_size.caption(f"{size_kb} KB")
        if col_action.button("Remove", key=f"rm_{doc_path}"):
            try:
                os.remove(doc_path)
                st.success(f"Removed **{name}**. Rebuild the index to apply changes.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not remove {name}: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Upload new documents
# ---------------------------------------------------------------------------
st.subheader("2. Upload New Documents")
st.caption("Supported formats: PDF, TXT, Markdown. Files are saved to `data/uploaded/`.")

uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    if st.button("Save uploaded files", type="secondary"):
        saved = []
        for uf in uploaded_files:
            dest = os.path.join(UPLOADED_DIR, uf.name)
            with open(dest, "wb") as out:
                out.write(uf.read())
            saved.append(uf.name)
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        st.info("Click **Rebuild Index** below to make the new documents searchable.")
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Rebuild index
# ---------------------------------------------------------------------------
st.subheader("3. Rebuild Search Index")
st.caption(
    "Re-processes all documents on disk, re-embeds them, and replaces the FAISS index. "
    "This can take a few minutes depending on corpus size."
)

docs_to_index = find_documents([DATA_DIR, UPLOADED_DIR])

if not docs_to_index:
    st.warning("No documents found. Upload files first.")
else:
    st.info(f"Will process **{len(docs_to_index)}** document(s).")

    use_contextual = st.toggle(
        "Contextual Retrieval (recommended, slower)",
        value=False,
        help=(
            "Before embedding each chunk, call DeepSeek to generate a 1-2 sentence "
            "context prefix that describes where the chunk sits in the document. "
            "Costs ~1 API call per chunk. Significantly improves retrieval quality "
            "for structured policy documents."
        ),
    )

    if use_contextual:
        st.warning(
            "Contextual mode will make ~1 DeepSeek API call per chunk. "
            "For a large corpus this may take 10–30 minutes and incur API costs."
        )

    if st.button("Rebuild Index", type="primary", use_container_width=False):
        with st.spinner("Building index — please wait..."):
            count = build_index(docs_to_index, contextual=use_contextual)
        st.cache_resource.clear()
        if count > 0:
            mode = "contextual" if use_contextual else "standard"
            st.success(
                f"Index rebuilt [{mode} mode]: **{count}** chunks "
                f"from **{len(docs_to_index)}** documents."
            )
        else:
            st.error("No chunks were created. Check that the documents contain readable text.")
