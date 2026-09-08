"""
ingestion/loader.py
───────────────────
PDF document loader with text cleaning, soft-wrap repair, and metadata preservation.
"""

import re
import os
import time
from langchain_community.document_loaders import PyMuPDFLoader

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


def load_pdf(file_path: str):
    """
    Load a PDF and clean page content with layout preservation.

    Validations:
    - Path must exist and have .pdf extension.
    - File must not be 0 bytes.
    - Document must contain at least one page with extractable text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Unsupported file format for '{file_path}'. Only PDF documents are supported.")

    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {file_path}")

    start_time = time.time()
    file_name = os.path.basename(file_path)
    file_size_kb = round(file_size_bytes / 1024, 2)

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    load_time = time.time() - start_time
    page_count = len(documents)

    mlflow_logger.log_param("document_name", file_name)
    mlflow_logger.log_metric("page_count", page_count)
    mlflow_logger.log_metric("document_size_kb", file_size_kb)
    mlflow_logger.log_metric("load_time", round(load_time, 4))

    has_extractable_text = False

    for doc in documents:
        text = doc.page_content or ""
        # Normalise line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive newlines to a paragraph break (\n\n)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse soft line wraps into spaces
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Collapse duplicate horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        cleaned = text.strip()
        doc.page_content = cleaned

        if cleaned:
            has_extractable_text = True

        # Ensure page index metadata is reliably preserved
        raw_page = doc.metadata.get("page", 0)
        try:
            doc.metadata["page"] = int(raw_page)
        except (ValueError, TypeError):
            doc.metadata["page"] = 0

        # Ensure source file name is recorded
        doc.metadata["source"] = file_path
        doc.metadata["file_name"] = file_name

    if not has_extractable_text:
        raise ValueError(
            f"The document '{file_name}' contains no extractable text. "
            "It may consist exclusively of scanned images or rasterized graphics."
        )

    return documents