"""
rag/embedder.py

Thin wrapper around OpenAI's text-embedding-3-small model.
Batches requests to stay within API limits.
"""

from typing import List
import openai


class Embedder:
    """Convert text strings into dense vector embeddings."""

    MODEL = "text-embedding-3-small"
    BATCH_SIZE = 100          # OpenAI allows up to 2048, but 100 is safe

    def __init__(self, api_key: str):
        self._client = openai.OpenAI(api_key=api_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks.
        Returns a list of float vectors in the same order as `texts`.
        """
        all_vectors: List[List[float]] = []
        for batch in self._batched(texts):
            response = self._client.embeddings.create(
                model=self.MODEL,
                input=batch,
            )
            # response.data is sorted by index
            vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        response = self._client.embeddings.create(
            model=self.MODEL,
            input=[text],
        )
        return response.data[0].embedding

    # ── Private helpers ────────────────────────────────────────────────────────

    def _batched(self, items: list):
        for i in range(0, len(items), self.BATCH_SIZE):
            yield items[i : i + self.BATCH_SIZE]
