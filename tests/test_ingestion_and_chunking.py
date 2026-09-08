"""
Tests for document ingestion, file validation, text normalization, and chunk splitting.
"""

import os
import pytest
from langchain_core.documents import Document
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def test_load_pdf_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("non_existent_file_path_12345.pdf")


def test_load_pdf_invalid_extension(tmp_path):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("This is not a PDF.", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_pdf(str(txt_file))


def test_load_pdf_empty_file(tmp_path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(ValueError, match="0 bytes"):
        load_pdf(str(empty_pdf))


def test_splitter_empty_input():
    with pytest.raises(ValueError):
        split_documents([])


def test_splitter_chunk_size_and_overlap():
    long_text = ("Artificial Intelligence and Machine Learning represent paradigms of modern computing. " * 30)
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})

    chunks = split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should respect max character size (+ minimal tolerance for single words)
        assert len(chunk.page_content) <= CHUNK_SIZE + 50
        # Provenance metadata must be preserved
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 0


def test_splitter_whitespace_normalization():
    dirty_text = "Paragraph one with normal text.\n\n\n\nParagraph two after excess newlines."
    doc = Document(page_content=dirty_text, metadata={"source": "clean.pdf"})
    chunks = split_documents([doc])
    assert len(chunks) >= 1
    # Check that chunks do not contain runs of 4+ newlines
    for c in chunks:
        assert "\n\n\n\n" not in c.page_content
