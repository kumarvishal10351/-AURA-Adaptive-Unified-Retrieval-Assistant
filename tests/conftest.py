"""
tests/conftest.py
─────────────────
Shared test fixtures and utilities for the AURA test suite.
"""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_document():
    """A single-page document with known content for testing."""
    return Document(
        page_content="AURA uses Sentence Transformers for 384-dimensional dense embeddings. "
                     "The FAISS index is persisted to disk in the faiss_db directory. "
                     "The cosine similarity threshold is 0.20 for filtering candidates.",
        metadata={"page": 0, "source": "test_doc.pdf", "file_name": "test_doc.pdf"},
    )


@pytest.fixture
def multi_page_documents():
    """Multiple documents simulating a multi-page PDF extraction."""
    return [
        Document(
            page_content="Chapter 1: Introduction to Machine Learning. "
                         "Machine learning is a subset of artificial intelligence.",
            metadata={"page": 0, "source": "textbook.pdf", "file_name": "textbook.pdf"},
        ),
        Document(
            page_content="Chapter 2: Neural Networks. "
                         "A neural network consists of layers of interconnected nodes.",
            metadata={"page": 1, "source": "textbook.pdf", "file_name": "textbook.pdf"},
        ),
        Document(
            page_content="Chapter 3: Deep Learning. "
                         "Deep learning uses multiple layers to progressively extract features.",
            metadata={"page": 2, "source": "textbook.pdf", "file_name": "textbook.pdf"},
        ),
    ]


@pytest.fixture
def scored_results():
    """Retrieval results with cosine scores for confidence testing."""
    docs = [
        Document(page_content=f"Result {i}", metadata={"page": i})
        for i in range(5)
    ]
    scores = [0.75, 0.60, 0.45, 0.30, 0.20]
    return list(zip(docs, scores))
