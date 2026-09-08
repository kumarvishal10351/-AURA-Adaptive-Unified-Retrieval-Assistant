"""
ingestion/splitter.py
─────────────────────
Splits documents into overlapping chunks with semantic hierarchy preservation.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


def split_documents(documents):
    """
    Split documents into overlapping chunks using paragraph-aware separators.
    Preserves page numbers, source document paths, and semantic hierarchy.
    """
    if not documents:
        raise ValueError("No documents provided to split_documents().")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    mlflow_logger.log_param("chunk_size", CHUNK_SIZE)
    mlflow_logger.log_param("chunk_overlap", CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    # Clean and filter out empty or whitespace-only chunks
    cleaned_chunks = []
    for c in chunks:
        if c.page_content:
            text = c.page_content.strip()
            # Collapse any runs of 3+ newlines to standard paragraph break
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                c.page_content = text
                cleaned_chunks.append(c)

    if not cleaned_chunks:
        raise ValueError("Document splitting resulted in zero valid text chunks.")

    mlflow_logger.log_metric("chunk_count", len(cleaned_chunks))

    return cleaned_chunks