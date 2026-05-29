"""
utils/embedder.py
Generates semantic embeddings for text chunks using sentence-transformers.
These embeddings enable similarity-based retrieval (vs. the keyword search
fallback in retriever.py).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Default lightweight model — fast and good enough for most RAG use cases.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Module-level cache so we don't reload the model on every call.
_model_cache: dict[str, SentenceTransformer] = {}


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load (or return cached) a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A loaded SentenceTransformer instance.
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_chunks(
    chunks: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of text chunks into dense vectors.

    Args:
        chunks:        List of text strings to embed.
        model_name:    SentenceTransformer model to use.
        batch_size:    Number of chunks to encode at once.
        show_progress: Show a tqdm progress bar during encoding.

    Returns:
        A 2-D NumPy array of shape (len(chunks), embedding_dim).
    """
    model = load_model(model_name)
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise so dot product == cosine sim
    )
    return embeddings  # shape: (N, D)


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Embed a single query string into a dense vector.

    Args:
        query:      The user's question or search string.
        model_name: SentenceTransformer model to use.

    Returns:
        A 1-D NumPy array of shape (embedding_dim,).
    """
    model = load_model(model_name)
    vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector  # shape: (D,)
