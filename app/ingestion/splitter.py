from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils import mlflow_logger


def split_documents(documents):
    """
    Split documents into overlapping chunks using paragraph-aware separators.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    mlflow_logger.log_param(
        "chunk_size",
        CHUNK_SIZE
    )

    mlflow_logger.log_param(
        "chunk_overlap",
        CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    mlflow_logger.log_metric(
        "chunk_count",
        len(chunks)
    )

    return chunks