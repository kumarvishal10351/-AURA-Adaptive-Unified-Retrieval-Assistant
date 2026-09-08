"""
ingestion/embedder.py
─────────────────────
Embeds document chunks using sentence-transformers and persists FAISS index to disk.
"""

from __future__ import annotations

import os
import time

from langchain_community.vectorstores import FAISS

from config.settings import FAISS_DB_DIR, EMBEDDING_MODEL
from retrieval.retriever import _get_embeddings, get_vectorstore

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


def store_embeddings(chunks: list) -> FAISS:
    """
    Embed document chunks with normalized sentence-transformers embeddings
    and serialize the FAISS index to disk at FAISS_DB_DIR.
    """
    if not chunks:
        raise ValueError(
            "No chunks provided to store_embeddings(). "
            "Verify that document loaded correctly and text splitter produced output."
        )

    embeddings = _get_embeddings()

    try:
        mlflow_logger.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow_logger.log_param("vector_store", "FAISS")

        start_time = time.time()

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        embedding_time = time.time() - start_time

        mlflow_logger.log_metric("embedding_time", round(embedding_time, 4))
        mlflow_logger.log_metric("embedded_chunks", len(chunks))

    except MemoryError:
        raise MemoryError(
            f"Insufficient memory to generate embeddings for {len(chunks)} chunks. "
            "Reduce chunk_size or process a smaller document."
        )

    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DB_DIR)

    # Invalidate cached vectorstore so fresh index is immediately loaded
    try:
        get_vectorstore.clear()
    except Exception:
        pass

    return vectorstore