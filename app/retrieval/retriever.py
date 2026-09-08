"""
retrieval/retriever.py
─────────────────────
Three-stage retrieval pipeline:
  1. FAISS similarity search with over-fetch (L2-normalized cosine scores)
  2. Cosine score threshold filter (≥ 0.20 floor to drop noise)
  3. CrossEncoder reranking (ms-marco-MiniLM-L-6-v2) for joint attention ordering
"""

from __future__ import annotations

import os
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    FAISS_DB_DIR,
    COSINE_THRESHOLD,
    TOP_K,
)

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Single shared, cached embedding model for the entire session.
    Pinned to CPU with normalize_embeddings=True for true cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk. Cached for the session.
    Verifies that both index.faiss and index.pkl exist.
    """
    faiss_file = os.path.join(FAISS_DB_DIR, "index.faiss")
    pkl_file = os.path.join(FAISS_DB_DIR, "index.pkl")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        raise FileNotFoundError(
            f"FAISS index files not found in '{FAISS_DB_DIR}'. "
            "Please upload and index a PDF document first."
        )

    return FAISS.load_local(
        folder_path=FAISS_DB_DIR,
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def _get_reranker():
    """
    Load the CrossEncoder reranker. Cached for the session.
    Uses max_length=512 to truncate gracefully.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def retrieve(
    query: str,
    *,
    k: int = TOP_K,
    rerank_top_n: int = 5,
) -> list[tuple]:
    """
    Run three-stage retrieval:
      1. FAISS over-fetch candidates
      2. Filter by COSINE_THRESHOLD (≥ 0.20)
      3. CrossEncoder re-ranking
    Returns: list of (doc, cosine_score: float)
    """
    vs = get_vectorstore()
    fetch_k = max(k * 2, 12)

    try:
        raw_results = vs.similarity_search_with_relevance_scores(query, k=fetch_k)
    except Exception:
        return []

    if not raw_results:
        return []

    # Filter with cosine threshold (cosine scores are bounded [0, 1])
    passing = [(doc, float(score)) for doc, score in raw_results if float(score) >= COSINE_THRESHOLD]

    # Fallback to raw candidates if none pass threshold
    if not passing:
        passing = [(doc, float(score)) for doc, score in raw_results[:k]]

    candidate_docs = [doc for doc, _ in passing]
    cosine_lookup = {id(doc): score for doc, score in passing}

    try:
        reranker = _get_reranker()
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        ce_scores = reranker.predict(pairs)

        ranked = sorted(
            zip(ce_scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True,
        )[:rerank_top_n]

        final_docs = [doc for _, doc in ranked]
    except Exception:
        # CrossEncoder unavailable — sort by cosine similarity
        final_docs = [
            doc for doc, _ in sorted(passing, key=lambda x: x[1], reverse=True)
        ][:rerank_top_n]

    # Preserve original cosine scores for confidence scoring
    return [(doc, cosine_lookup.get(id(doc), 0.0)) for doc in final_docs]