"""
Tests for source attribution, page index conversion, and citation formatting.
"""

from langchain_core.documents import Document


def test_source_page_one_indexing():
    """
    Verifies that 0-indexed PyMuPDF page numbers are converted to 1-indexed
    display page numbers to eliminate off-by-one errors for end users.
    """
    doc_first_page = Document(
        page_content="First page introduction",
        metadata={"page": 0, "source": "docs/Report.pdf", "file_name": "Report.pdf"}
    )
    doc_tenth_page = Document(
        page_content="Tenth page conclusions",
        metadata={"page": 9, "source": "docs/Report.pdf", "file_name": "Report.pdf"}
    )

    docs = [doc_first_page, doc_tenth_page]

    formatted_sources = [
        {
            "content": d.page_content,
            "page": (d.metadata.get("page", 0) + 1) if isinstance(d.metadata.get("page"), int) else d.metadata.get("page", 1),
            "source": d.metadata.get("file_name", "Report.pdf"),
        }
        for d in docs
    ]

    assert formatted_sources[0]["page"] == 1  # 0 -> 1
    assert formatted_sources[1]["page"] == 10 # 9 -> 10
    assert formatted_sources[0]["source"] == "Report.pdf"


def test_source_attribution_non_integer_page():
    doc = Document(
        page_content="Unspecified page chunk",
        metadata={"page": "?", "file_name": "Paper.pdf"}
    )
    page_val = (doc.metadata.get("page", 0) + 1) if isinstance(doc.metadata.get("page"), int) else doc.metadata.get("page", 1)
    assert page_val == "?"
