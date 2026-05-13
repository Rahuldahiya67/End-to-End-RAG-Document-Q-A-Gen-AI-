# 🧠 DocMind — End-to-End RAG Document Q&A

A fully working **Retrieval-Augmented Generation (RAG)** application built in Python.
Upload PDFs or text files and ask natural-language questions — grounded answers, always cited.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                      app.py  (Streamlit UI)             │
└───────────────────────┬────────────────────────────────-┘
                        │
          ┌─────────────▼─────────────┐
          │      document_loader.py   │  ← Upload & chunk docs
          └─────────────┬─────────────┘
                        │  chunks[ ]
          ┌─────────────▼─────────────┐
          │        embedder.py        │  ← OpenAI text-embedding-3-small
          └─────────────┬─────────────┘
                        │  vectors[ ]
          ┌─────────────▼─────────────┐
          │       vector_store.py     │  ← FAISS in-memory index
          └─────────────┬─────────────┘
                        │
                  [User asks Q]
                        │
          ┌─────────────▼─────────────┐
          │        retriever.py       │  ← Embed query → top-K search
          └─────────────┬─────────────┘
                        │  top chunks
          ┌─────────────▼─────────────┐
          │        generator.py       │  ← GPT-4o with RAG prompt
          └─────────────┬─────────────┘
                        │
                   ✅ Answer
```

---

## 📁 Project Structure

```
rag_project/
├── app.py                  # Streamlit UI entry point
├── requirements.txt
├── README.md
└── rag/
    ├── __init__.py
    ├── document_loader.py  # File reading + text chunking
    ├── embedder.py         # OpenAI embeddings wrapper
    ├── vector_store.py     # FAISS index (add / search / persist)
    ├── retriever.py        # Query → top-K chunks
    └── generator.py        # RAG prompt + GPT-4o answer
```

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd rag_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Use it

1. Enter your **OpenAI API key** in the sidebar
2. Upload one or more **PDF / TXT / MD** files
3. Adjust chunk size & top-K as needed
4. Click **Index Documents**
5. Ask questions in the chat!

---

## ⚙️ Configuration

| Parameter      | Default | Description                              |
|----------------|---------|------------------------------------------|
| Chunk size     | 500     | Words per chunk                          |
| Chunk overlap  | 50      | Words shared between consecutive chunks  |
| Top-K          | 4       | Number of chunks retrieved per query     |

---

## 🔑 Key Concepts

### Chunking
Long documents are split into overlapping windows of words. Overlap preserves
context across chunk boundaries (avoids cutting sentences mid-thought).

### Embeddings
Each chunk is converted into a high-dimensional vector (1536-d) using OpenAI's
`text-embedding-3-small`. These capture semantic meaning, not just keywords.

### FAISS Index
All vectors are stored in a FAISS `IndexFlatIP` (inner-product / cosine similarity).
At query time, the query vector is compared against all chunk vectors in milliseconds.

### RAG Prompt
The top-K chunks are injected into the GPT-4o system prompt as context.
The model is instructed to answer ONLY from that context and cite chunk IDs.

---

## 🔧 Optional Enhancements

- **Cross-encoder re-ranking**: Uncomment `sentence-transformers` in `requirements.txt`
  and call `retriever.retrieve_with_rerank(query)` for higher accuracy.
- **Persistent index**: Call `vector_store.save("./index")` to save and
  `vector_store.load("./index")` to reload without re-embedding.
- **Swap LLM**: Replace GPT-4o in `generator.py` with any OpenAI-compatible API
  (Mistral, LLaMA via Ollama, etc.).
- **Swap vector DB**: Replace FAISS with Pinecone / Qdrant / Chroma for production.

---

## 📦 Dependencies

| Package      | Purpose                        |
|--------------|--------------------------------|
| `openai`     | Embeddings + GPT-4o            |
| `streamlit`  | Web UI                         |
| `faiss-cpu`  | Fast vector similarity search  |
| `numpy`      | Matrix operations              |
| `PyPDF2`     | PDF text extraction            |

---

## 🛡️ Security Notes

- Your OpenAI API key is **never stored** — it lives only in Streamlit session state.
- For production, use environment variables: `export OPENAI_API_KEY=sk-...`
  and read with `os.getenv("OPENAI_API_KEY")`.
