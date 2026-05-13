"""
rag/vector_store.py

In-memory vector store backed by FAISS (CPU, flat L2 index).
For production, swap this for Pinecone / Weaviate / Qdrant.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class VectorStore:
    """
    Stores chunk embeddings in a FAISS flat index and supports
    cosine-similarity search via L2 on normalised vectors.
    """

    def __init__(self):
        self._index = None          # FAISS index (lazy init)
        self._chunks: List[Dict[str, Any]] = []  # metadata parallel to index
        self._dim: int = 0

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict[str, Any]], embedder) -> None:
        """
        Embed all chunks and add them to the FAISS index.

        Args:
            chunks:   list of chunk dicts (from document_loader)
            embedder: Embedder instance
        """
        import faiss  # local import so the module loads without faiss installed

        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_documents(texts)

        matrix = np.array(vectors, dtype="float32")
        # L2-normalise for cosine similarity
        faiss.normalize_L2(matrix)

        self._dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)   # Inner-product on normed = cosine
        self._index.add(matrix)
        self._chunks = chunks

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(
        self, query_vector: List[float], top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Find the top-k most similar chunks for a query embedding.

        Returns a list of chunk dicts, each augmented with a `score` key
        (cosine similarity, 0–1).
        """
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._chunks[idx])   # shallow copy
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Persistence (optional) ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the FAISS index to disk."""
        import faiss, pickle, pathlib
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index."""
        import faiss, pickle, pathlib
        p = pathlib.Path(path)
        self._index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        self._dim = self._index.d
