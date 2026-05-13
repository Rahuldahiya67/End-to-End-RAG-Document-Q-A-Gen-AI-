"""
rag/retriever.py

Encapsulates query embedding + vector-store search.
Optionally re-ranks results with a cross-encoder (if installed).
"""

from typing import List, Dict, Any
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """
    Given a natural-language query, return the most relevant document chunks.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder, top_k: int = 4):
        self._vs = vector_store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Embed the query and return the top-k chunks from the vector store.
        Each chunk dict includes: id, text, source, chunk, score.
        """
        query_vec = self._embedder.embed_query(query)
        results = self._vs.search(query_vec, top_k=self._top_k)
        return results

    def retrieve_with_rerank(self, query: str) -> List[Dict[str, Any]]:
        """
        Optional: retrieve 2x candidates, then cross-encoder rerank.
        Requires: pip install sentence-transformers
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            # Fall back to standard retrieval
            return self.retrieve(query)

        candidates = self._vs.search(self._embedder.embed_query(query), top_k=self._top_k * 2)
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self._top_k]
