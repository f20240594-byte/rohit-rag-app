"""
utils/retriever.py
Retrieves the most relevant text chunks for a user query.

Two strategies are available:
  1. semantic  — cosine similarity on sentence-transformer embeddings (recommended).
  2. keyword   — simple word-overlap scoring (no extra dependencies, fast fallback).
"""

from __future__ import annotations

import numpy as np


# ─── Semantic retrieval ────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[str]:
    """
    Return the top-k chunks most semantically similar to the query.

    Embeddings must be L2-normalised (as produced by embedder.embed_chunks /
    embedder.embed_query with normalize_embeddings=True), so similarity is
    computed as a simple dot product.

    Args:
        query:             The user's question.
        chunks:            Original text chunks (parallel to chunk_embeddings).
        chunk_embeddings:  Pre-computed embeddings, shape (N, D).
        top_k:             Number of top chunks to return.
        model_name:        SentenceTransformer model used for the query embedding.

    Returns:
        List of up to top_k chunk strings, ordered by descending similarity.
    """
    from utils.embedder import embed_query  # local import to avoid circular deps

    query_vec = embed_query(query, model_name=model_name)  # shape (D,)

    # Cosine similarities (vectors are already normalised → dot product suffices)
    scores = chunk_embeddings @ query_vec  # shape (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ─── Keyword retrieval (fallback) ─────────────────────────────────────────────

def keyword_search(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[str]:
    """
    Return the top-k chunks with the highest keyword overlap with the query.

    Scoring: count of unique query words that appear in the chunk (case-insensitive).
    Ties are broken by original chunk order.

    Args:
        query:  The user's question.
        chunks: List of text chunks to search over.
        top_k:  Number of top chunks to return.

    Returns:
        List of up to top_k chunk strings, ordered by descending overlap score.
    """
    query_words = set(query.lower().split())
    scored: list[tuple[int, str]] = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ─── Unified retrieval entry-point ────────────────────────────────────────────

def find_relevant_chunks(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray | None = None,
    top_k: int = 5,
    strategy: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> list[str]:
    """
    High-level retrieval function used by the Streamlit app.

    Strategy selection:
      "semantic" — always use cosine-similarity search (requires embeddings).
      "keyword"  — always use keyword overlap search.
      "auto"     — use semantic if embeddings are provided, else fall back to keyword.

    Args:
        query:             User question.
        chunks:            Text chunks from the document.
        chunk_embeddings:  Pre-computed chunk embeddings (required for semantic).
        top_k:             Number of chunks to retrieve.
        strategy:          One of "auto", "semantic", "keyword".
        model_name:        Embedding model (used in semantic mode only).

    Returns:
        List of the most relevant text chunks.

    Raises:
        ValueError: If strategy is "semantic" but no embeddings are provided.
    """
    if strategy == "semantic" and chunk_embeddings is None:
        raise ValueError(
            "strategy='semantic' requires pre-computed chunk_embeddings. "
            "Call embedder.embed_chunks() first and pass the result here."
        )

    use_semantic = (
        strategy == "semantic"
        or (strategy == "auto" and chunk_embeddings is not None)
    )

    if use_semantic:
        return semantic_search(
            query,
            chunks,
            chunk_embeddings,
            top_k=top_k,
            model_name=model_name,
        )
    else:
        return keyword_search(query, chunks, top_k=top_k)
