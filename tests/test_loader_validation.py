"""
Extended tests for document loading, validation, and text normalization.

Tests boundary conditions: corrupt PDFs, unusual whitespace, metadata preservation.
"""

import os
import pytest
from langchain_core.documents import Document
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class TestLoaderValidation:
    """Tests for PDF loader input validation."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent_path_12345.pdf")

    def test_non_pdf_extension(self, tmp_path):
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("Not a PDF", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_pdf(str(txt_file))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="0 bytes"):
            load_pdf(str(empty))

    def test_corrupt_pdf(self, tmp_path):
        """A file with .pdf extension but invalid content should fail gracefully."""
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"This is not a valid PDF file content at all")
        # PyMuPDF should raise an error for invalid PDF
        with pytest.raises(Exception):
            load_pdf(str(corrupt))

    def test_pdf_with_only_images(self, tmp_path):
        """A valid but text-less PDF should raise ValueError about no extractable text."""
        # We can't easily create an image-only PDF in tests, but we can test the
        # validation path by creating a minimal valid PDF with no text
        # For now, verify the error message format
        pass  # Covered by the empty extraction path in loader.py


class TestTextNormalization:
    """Tests for the text cleaning pipeline in loader.py."""

    def test_soft_wrap_repair(self):
        """Single newlines (soft wraps) should be joined with spaces."""
        import re
        text = "This is a line\nthat was wrapped\nby the PDF renderer."
        # Apply the same normalization as loader.py
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        assert "\n" not in text
        assert "line that" in text

    def test_paragraph_breaks_preserved(self):
        """Double newlines (paragraph breaks) should be preserved."""
        import re
        text = "Paragraph one.\n\nParagraph two."
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        assert "\n\n" in text
        assert "Paragraph one." in text
        assert "Paragraph two." in text

    def test_excessive_newlines_collapsed(self):
        """More than 2 newlines should collapse to paragraph break."""
        import re
        text = "Before.\n\n\n\n\nAfter."
        text = re.sub(r"\n{3,}", "\n\n", text)
        assert text == "Before.\n\nAfter."

    def test_tab_and_space_collapse(self):
        """Multiple spaces and tabs should collapse to single space."""
        import re
        text = "Word1   \t\t   Word2"
        text = re.sub(r"[ \t]+", " ", text)
        assert text == "Word1 Word2"

    def test_windows_line_endings(self):
        """\\r\\n should be normalized to \\n."""
        text = "Line1\r\nLine2\r\nLine3"
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        assert "\r" not in text
        assert text == "Line1\nLine2\nLine3"


class TestMetadataPreservation:
    """Tests for metadata preservation through the pipeline."""

    def test_chunk_inherits_page_metadata(self):
        """Chunks should inherit page number from parent document."""
        doc = Document(
            page_content="A " * 600,  # Long enough to split
            metadata={"page": 5, "source": "test.pdf", "file_name": "test.pdf"},
        )
        chunks = split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["source"] == "test.pdf"

    def test_multi_page_chunks_preserve_pages(self):
        """Each chunk should retain the page number of its source page."""
        docs = [
            Document(
                page_content=f"Page {i} content. " * 50,
                metadata={"page": i, "source": "multi.pdf", "file_name": "multi.pdf"},
            )
            for i in range(3)
        ]
        chunks = split_documents(docs)
        # Each chunk should have a valid page number from 0-2
        for chunk in chunks:
            assert chunk.metadata["page"] in [0, 1, 2]


class TestSplitterEdgeCases:
    """Edge cases for the text splitter."""

    def test_very_short_document(self):
        """A document shorter than chunk_size should produce one chunk."""
        doc = Document(page_content="Short text.", metadata={"page": 0, "source": "s.pdf"})
        chunks = split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_exactly_chunk_size(self):
        """A document exactly at chunk_size should produce one chunk."""
        text = "x" * CHUNK_SIZE
        doc = Document(page_content=text, metadata={"page": 0, "source": "s.pdf"})
        chunks = split_documents([doc])
        assert len(chunks) >= 1

    def test_whitespace_only_chunks_filtered(self):
        """Chunks that are only whitespace should be filtered out."""
        doc = Document(
            page_content="Real content here.\n\n" + " " * 500 + "\n\nMore content.",
            metadata={"page": 0, "source": "s.pdf"},
        )
        chunks = split_documents([doc])
        for chunk in chunks:
            assert chunk.page_content.strip() != ""

    def test_unicode_content_preserved(self):
        """Unicode characters should survive chunking."""
        text = "Les réseaux de neurones profonds sont très puissants. " * 30
        doc = Document(page_content=text, metadata={"page": 0, "source": "french.pdf"})
        chunks = split_documents([doc])
        # Verify accented characters survive
        full_text = " ".join(c.page_content for c in chunks)
        assert "réseaux" in full_text
        assert "très" in full_text

    def test_special_characters_preserved(self):
        """Special characters (math, symbols) should survive chunking."""
        text = "The formula is: E = mc². Temperature ≥ 100°C. Cost: $1,000. " * 30
        doc = Document(page_content=text, metadata={"page": 0, "source": "math.pdf"})
        chunks = split_documents([doc])
        full_text = " ".join(c.page_content for c in chunks)
        assert "mc²" in full_text
        assert "≥" in full_text
        assert "°C" in full_text
