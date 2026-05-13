"""
Document Q&A RAG Application
Entry point — runs the Streamlit UI
"""

import streamlit as st
import os
from pathlib import Path
from rag.document_loader import load_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0d0d; color: #f0ede6; }

h1 { font-size: 3rem !important; font-weight: 800 !important; 
     background: linear-gradient(135deg, #f0ede6, #c8b89a);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.source-card {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-left: 3px solid #c8b89a; border-radius: 4px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
    font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #999;
}
.answer-box {
    background: #111; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 1.5rem; margin-top: 1rem;
    line-height: 1.7;
}
.metric-pill {
    display: inline-block; background: #1e1e1e;
    border: 1px solid #333; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.75rem;
    font-family: 'Space Mono', monospace; color: #c8b89a;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key", type="password",
        placeholder="sk-...",
        help="Your OpenAI key — never stored"
    )
    
    st.markdown("---")
    st.markdown("### 📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "PDF / TXT / MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    
    chunk_size = st.slider("Chunk size (tokens)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, 10)
    top_k = st.slider("Top-K retrieval", 1, 10, 4)
    
    st.markdown("---")
    
    if st.button("🚀 Index Documents", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Loading & chunking documents…"):
                docs = load_documents(uploaded_files, chunk_size, chunk_overlap)
            with st.spinner(f"Embedding {len(docs)} chunks…"):
                embedder = Embedder(api_key)
                vs = VectorStore()
                vs.add_documents(docs, embedder)
                st.session_state.vector_store = vs
                st.session_state.doc_count = len(docs)
                st.session_state.api_key = api_key
            st.success(f"✅ Indexed {len(uploaded_files)} files → {len(docs)} chunks")

    if st.session_state.vector_store:
        st.markdown(f"""
        <div class="metric-pill">📄 {st.session_state.doc_count} chunks</div>
        <div class="metric-pill">✅ Ready</div>
        """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🧠 DocMind")
st.markdown("*Retrieval-Augmented Generation — ask anything about your documents*")
st.markdown("---")

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f'<div class="source-card">{src}</div>', unsafe_allow_html=True)

# Query input
query = st.chat_input("Ask a question about your documents…")

if query:
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and index documents first.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve + Generate
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                retriever = Retriever(
                    st.session_state.vector_store,
                    Embedder(st.session_state.api_key),
                    top_k=top_k,
                )
                chunks = retriever.retrieve(query)
                
                generator = Generator(st.session_state.api_key)
                answer = generator.generate(query, chunks, st.session_state.chat_history)
            
            st.markdown(answer)
            sources = [f"[Chunk {c['id']}] {c['source']} — score: {c['score']:.3f}\n\n{c['text'][:200]}…" for c in chunks]
            with st.expander("📚 Sources used"):
                for src in sources:
                    st.markdown(f'<div class="source-card">{src}</div>', unsafe_allow_html=True)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
