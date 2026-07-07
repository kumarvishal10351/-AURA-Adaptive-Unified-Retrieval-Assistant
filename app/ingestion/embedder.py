"""
ingestion/embedder.py
─────────────────────
Embeds document chunks and persists a FAISS index to disk.

Fixes vs original:
  • Imports _get_embeddings from retriever — single shared cached instance,
    no second 400 MB model load.
  • Removed dead `collection_name` parameter (FAISS has no collections).
  • Removed ThreadPoolExecutor wrapper — FAISS.from_documents is a single
    synchronous call; the thread + Future overhead added ~50 ms with zero
    benefit and a misleading 120 s timeout that could kill large-doc jobs.
  • Added empty-chunk guard with a clear error message.
  • Added MemoryError handling with a useful hint.
"""

from __future__ import annotations

import os
import time

from langchain_community.vectorstores import FAISS

from config.settings import FAISS_DB_DIR
from utils import mlflow_logger

# Re-use the single shared, cached embeddings model from retriever.
# Avoids loading the same ~400 MB HuggingFace model a second time.
from retrieval.retriever import _get_embeddings


def store_embeddings(chunks: list) -> FAISS:
    """
    Embed `chunks` with the shared HuggingFace model and persist the
    resulting FAISS index to ``FAISS_DB_DIR``.

    Parameters
    ----------
    chunks : list[Document]
        LangChain Document objects produced by the text splitter.
        Must be non-empty.

    Returns
    -------
    FAISS
        The newly created (and already saved) vectorstore.

    Raises
    ------
    ValueError
        If `chunks` is empty — an empty index is useless and would cause
        confusing errors at query time.
    MemoryError
        Re-raised with a helpful hint if the embedding run exhausts RAM.
    """
    if not chunks:
        raise ValueError(
            "No chunks provided to store_embeddings(). "
            "Check that your PDF loaded correctly and the splitter produced output."
        )

    embeddings = _get_embeddings()

    try:
        mlflow_logger.log_param(
            "embedding_model",
            "all-MiniLM-L6-v2"
        )

        mlflow_logger.log_param(
            "vector_store",
            "FAISS"
        )

        start_time = time.time()

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        embedding_time = time.time() - start_time

        mlflow_logger.log_metric(
            "embedding_time",
            embedding_time
        )

        mlflow_logger.log_metric(
            "embedded_chunks",
            len(chunks)
        )
    except MemoryError:
        raise MemoryError(
            f"Not enough memory to embed {len(chunks)} chunks. "
            "Try reducing chunk_size or chunk_overlap in your splitter settings, "
            "or process a smaller document."
        )

    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DB_DIR)
    return vectorstore