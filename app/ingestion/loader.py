import re
import os
import time

from langchain_community.document_loaders import PyMuPDFLoader
from utils import mlflow_logger


def load_pdf(file_path: str):
    """
    Load a PDF and clean page content.

    Strategy:
    - Preserve paragraph breaks (double newlines → two spaces kept as a space).
    - Collapse runs of whitespace/single newlines into a single space so the
      RecursiveCharacterTextSplitter separator '\\n\\n' still has a chance to
      find real section boundaries that PyMuPDF emits as double newlines.
    - Strip leading/trailing whitespace from each page.
    """
    start_time = time.time()

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    load_time = time.time() - start_time

    file_name = os.path.basename(file_path)

    file_size_kb = round(
        os.path.getsize(file_path) / 1024,
        2
    )

    page_count = len(documents)

    mlflow_logger.log_param(
        "document_name",
        file_name
    )

    mlflow_logger.log_metric(
        "page_count",
        page_count
    )

    mlflow_logger.log_metric(
        "document_size_kb",
        file_size_kb
    )

    mlflow_logger.log_metric(
        "load_time",
        load_time
    )

    for doc in documents:
        text = doc.page_content
        # Normalise Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runs of 3+ newlines to a double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse single newlines (soft wraps) into a space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Collapse multiple spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)
        doc.page_content = text.strip()

    return documents