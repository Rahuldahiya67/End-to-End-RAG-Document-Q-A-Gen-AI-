"""
rag/generator.py

Builds a RAG prompt from retrieved chunks and calls GPT-4o to generate
a grounded, cited answer. Supports multi-turn conversation history.
"""

from typing import List, Dict, Any
import openai


SYSTEM_PROMPT = """You are DocMind, an expert document analyst.
Answer the user's question using ONLY the information in the provided context chunks.
Follow these rules strictly:
1. If the answer is not contained in the context, say: "I couldn't find relevant information in the documents."
2. Always cite which chunk(s) you used, referencing them as [Chunk N].
3. Be concise and precise. Prefer bullet points for multi-part answers.
4. Never hallucinate facts not present in the context.
"""


class Generator:
    """Generate an answer grounded in retrieved document chunks."""

    MODEL = "gpt-4o"
    MAX_CONTEXT_CHARS = 8_000   # ~2k tokens of context

    def __init__(self, api_key: str):
        self._client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        Build the prompt, call the model, return the answer string.

        Args:
            query:   The user's question.
            chunks:  Retrieved chunk dicts (must have 'id' and 'text').
            history: Prior chat turns [{"role": "user"|"assistant", "content": str}].
        """
        context = self._build_context(chunks)
        messages = self._build_messages(query, context, history or [])

        response = self._client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=0.2,          # Low temp for factual accuracy
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        total = 0
        for c in chunks:
            snippet = f"[Chunk {c['id']}] (source: {c['source']})\n{c['text']}"
            if total + len(snippet) > self.MAX_CONTEXT_CHARS:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include last 6 turns of history for multi-turn awareness
        recent_history = history[-6:] if len(history) > 6 else history
        # Filter out source metadata before sending to API
        for turn in recent_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        user_content = (
            f"Context from documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages
