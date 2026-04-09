"""
retrieval/retriever.py
─────────────────────
Full retrieval pipeline:

    query
      ↓
    FAISS similarity search  (k×3 over-fetch)
      ↓
    Score threshold filter   (≥ 0.30 cosine — drops weak / irrelevant chunks)
      ↓
    CrossEncoder reranker    (precise query-chunk relevance scoring)
      ↓
    Top-N docs               (returned to rag_chain)

Fixes applied vs original:
  • Single shared embedding cache — no duplicate 400 MB model load
  • Removed dead collection_name param (FAISS has no collections)
  • Guard clause if FAISS index doesn't exist yet → clear FileNotFoundError
  • model_kwargs: device pinned to cpu; encode_kwargs: normalise + batch
  • CrossEncoder now actually called inside retrieve()
  • Score thresholding prevents irrelevant chunks reaching the LLM
  • Over-fetch + rerank replaces a naive top-k call
  • Graceful reranker fallback — still returns filtered results if CE unavailable
"""

from __future__ import annotations

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL, FAISS_DB_DIR


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Minimum cosine similarity to keep a retrieved chunk.
# Cosine similarity on L2-normalised vectors lies in [-1, 1].
# 0.30 is a practical floor — anything below is almost certainly noise.
_SCORE_THRESHOLD: float = 0.30

# CrossEncoder model — tiny, fast, accurate for passage re-ranking.
_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hard cap on FAISS candidates before reranking, to bound latency.
_MAX_FETCH: int = 20


# ─────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Single shared embedding model for the whole app session.

    Changes vs original:
    - model_kwargs: pins device to 'cpu' (avoids silent CUDA fallback on machines
      with partial GPU support; swap to 'cuda' if you have a GPU).
    - encode_kwargs: normalise_embeddings=True is REQUIRED for cosine similarity
      scores returned by similarity_search_with_relevance_scores to be meaningful.
      Without normalisation the scores are dot products, not cosine values, and the
      0.30 threshold below becomes meaningless.
      batch_size=32 speeds up embedding during ingestion.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk.  Cached for the session.

    Changes vs original:
    - Removed unused `collection_name` parameter (ChromaDB concept, not FAISS).
    - Added existence check: raises a clear FileNotFoundError if the index has
      not been created yet, instead of crashing inside pickle with a confusing
      message.
    - show_spinner=False: the main app manages its own progress UI; a second
      spinner appearing mid-pipeline looks broken.
    """
    if not os.path.exists(FAISS_DB_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_DB_DIR}'. "
            "Upload and process a document first."
        )

    return FAISS.load_local(
        folder_path=FAISS_DB_DIR,
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def _get_reranker():
    """
    Load the CrossEncoder reranker.  Cached for the session.

    Changes vs original:
    - Moved to a private function — callers should use retrieve(), not load
      the reranker directly.
    - Added max_length=512: the model's context window; truncates long chunks
      gracefully instead of silently producing garbage scores on overflow.
    - show_spinner=False for the same reason as get_vectorstore.
    """
    from sentence_transformers import CrossEncoder  # lazy import — only load if used
    return CrossEncoder(_RERANKER_MODEL, max_length=512)


# ─────────────────────────────────────────────────────────────────
# Public retrieval API
# ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    *,
    k: int = 8,
    rerank_top_n: int = 4,
) -> list:
    """
    Run the full three-stage retrieval pipeline and return the top-N most
    relevant document chunks for `query`.

    Parameters
    ----------
    query       : The user's question / search string.
    k           : Number of final results to return (after reranking).
                  Defaults to 8 so the LLM gets enough context; rag_chain
                  can pass a smaller rerank_top_n to trim further.
    rerank_top_n: How many chunks the CrossEncoder should keep.  Must be ≤ k.

    Pipeline
    --------
    Stage 1 — FAISS over-fetch
        Retrieve min(k×3, _MAX_FETCH) candidates via approximate nearest-neighbour
        search.  Over-fetching gives the reranker enough material to work with.
        similarity_search_with_relevance_scores returns (doc, float) pairs where
        the float is cosine similarity on normalised embeddings.

    Stage 2 — Score threshold
        Drop any chunk whose cosine similarity is below _SCORE_THRESHOLD.
        This is the primary gatekeeper: it stops irrelevant content from ever
        reaching the LLM and causing hallucinations or spurious NOT_FOUND responses.
        If *nothing* passes the threshold (very short or off-topic document) we fall
        back to the raw top-k so the LLM can still attempt an answer.

    Stage 3 — CrossEncoder reranking
        The CrossEncoder reads the (query, chunk) pair jointly — unlike the
        bi-encoder used for FAISS, it attends to every token in both strings.
        This gives much more accurate relevance scores for the filtered candidates.
        If the CrossEncoder is unavailable (model download failed, CI environment,
        etc.) we skip gracefully and return the score-filtered results directly.
    """
    vs = get_vectorstore()

    # ── Stage 1: FAISS over-fetch ─────────────────────────────────
    fetch_k = min(k * 3, _MAX_FETCH)
    docs_and_scores: list[tuple] = vs.similarity_search_with_relevance_scores(
        query, k=fetch_k
    )

    if not docs_and_scores:
        return []

    # ── Stage 2: Score threshold ──────────────────────────────────
    above_threshold = [
        (doc, score)
        for doc, score in docs_and_scores
        if score >= _SCORE_THRESHOLD
    ]

    # Graceful fallback: if nothing passes, use raw results so the LLM
    # can still attempt a response (it may answer NOT_FOUND correctly).
    if not above_threshold:
        above_threshold = docs_and_scores[:k]

    docs = [doc for doc, _ in above_threshold]

    if len(docs) <= 1:
        # Nothing to rerank
        return docs[:rerank_top_n]

    # ── Stage 3: CrossEncoder rerank ──────────────────────────────
    try:
        reranker = _get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        ce_scores = reranker.predict(pairs)  # returns numpy array of floats

        ranked_pairs = sorted(
            zip(ce_scores, docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in ranked_pairs[:rerank_top_n]]

    except Exception:
        # CrossEncoder unavailable — return score-filtered results
        return docs[:rerank_top_n]