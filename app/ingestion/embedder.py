"""
ingestion/embedder.py
─────────────────────
Embeds document chunks using sentence-transformers and persists FAISS index to disk.
Supports cumulative indexing across multiple documents and uses COSINE distance strategy.
"""

from __future__ import annotations

import os
import time

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

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
    Accumulates chunks with any existing index.
    """
    if not chunks:
        raise ValueError(
            "No chunks provided to store_embeddings(). "
            "Verify that document loaded correctly and text splitter produced output."
        )

    embeddings = _get_embeddings()
    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    faiss_file = os.path.join(FAISS_DB_DIR, "index.faiss")

    try:
        mlflow_logger.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow_logger.log_param("vector_store", "FAISS")

        start_time = time.time()

        # If index already exists on disk, load and append new chunks
        if os.path.exists(faiss_file):
            try:
                vectorstore = FAISS.load_local(
                    folder_path=FAISS_DB_DIR,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
                vectorstore.add_documents(chunks)
            except Exception:
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    distance_strategy=DistanceStrategy.COSINE,
                )
        else:
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )

        embedding_time = time.time() - start_time

        mlflow_logger.log_metric("embedding_time", round(embedding_time, 4))
        mlflow_logger.log_metric("embedded_chunks", len(chunks))

    except MemoryError:
        raise MemoryError(
            f"Insufficient memory to generate embeddings for {len(chunks)} chunks. "
            "Reduce chunk_size or process a smaller document."
        )

    vectorstore.save_local(FAISS_DB_DIR)

    # Invalidate cached vectorstore so fresh index is immediately loaded
    try:
        get_vectorstore.cache_clear()
    except Exception:
        pass

    return vectorstore