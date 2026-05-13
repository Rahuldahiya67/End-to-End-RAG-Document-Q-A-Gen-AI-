"""
rag/document_loader.py

Loads uploaded files (PDF, TXT, MD) and splits them into overlapping chunks.
"""

import re
import io
from typing import List, Dict, Any
from pathlib import Path


def load_documents(
    uploaded_files,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Read uploaded Streamlit file objects, extract text, and split into chunks.

    Returns a list of chunk dicts:
        {
            "id":     int,
            "text":   str,
            "source": str,   # original filename
            "chunk":  int,   # chunk index within the file
        }
    """
    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for file in uploaded_files:
        ext = Path(file.name).suffix.lower()
        raw_text = _extract_text(file, ext)
        if not raw_text.strip():
            continue

        chunks = _split_text(raw_text, chunk_size, chunk_overlap)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id":     chunk_id,
                "text":   chunk_text,
                "source": file.name,
                "chunk":  idx,
            })
            chunk_id += 1

    return all_chunks


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_text(file, ext: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    if ext == ".pdf":
        return _extract_pdf(file)
    elif ext in (".txt", ".md"):
        return file.read().decode("utf-8", errors="replace")
    else:
        return ""


def _extract_pdf(file) -> str:
    """Extract text from a PDF using PyPDF2."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF support. "
            "Install it with: pip install PyPDF2"
        )


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple word-count-based sliding-window chunker.
    Splits on whitespace, then reassembles windows of `chunk_size` words
    with `overlap` words of context carried over.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks
