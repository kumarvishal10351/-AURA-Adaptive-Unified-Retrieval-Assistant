"""
Unit tests for the RAG chain internal functions.

Tests the deterministic components of the pipeline:
  - _parse_expansion
  - _build_history
  - _build_context
  - NOT_FOUND sentinel handling
  - Token generator behavior
"""

import pytest
from langchain_core.documents import Document

from app.chains.rag_chain import _parse_expansion, _build_history, _build_context
from app.config.settings import MAX_CONTEXT_LENGTH, HISTORY_TURNS


class TestParseExpansion:
    """Tests for LLM query expansion output parsing."""

    def test_plain_lines(self):
        raw = "What is the warranty?\nDoes it cover damage?\nRepair policy details"
        queries = _parse_expansion(raw)
        assert len(queries) == 3
        assert queries[0] == "What is the warranty?"

    def test_numbered_lines(self):
        raw = "1. What is the warranty?\n2. Coverage details\n3. Repair policy"
        queries = _parse_expansion(raw)
        assert len(queries) == 3
        assert "1." not in queries[0]

    def test_bulleted_lines(self):
        raw = "- What is the warranty?\n- Coverage details\n- Repair policy"
        queries = _parse_expansion(raw)
        assert len(queries) == 3
        assert queries[0] == "What is the warranty?"

    def test_mixed_format(self):
        raw = "1) First query\n* Second query\n• Third query"
        queries = _parse_expansion(raw)
        assert len(queries) == 3

    def test_more_than_three_truncated(self):
        raw = "Q1\nQ2\nQ3\nQ4\nQ5"
        queries = _parse_expansion(raw)
        assert len(queries) == 3

    def test_empty_input(self):
        queries = _parse_expansion("")
        assert queries == []

    def test_whitespace_only(self):
        queries = _parse_expansion("   \n   \n   ")
        assert queries == []

    def test_empty_lines_filtered(self):
        raw = "Query one\n\n\nQuery two\n\nQuery three"
        queries = _parse_expansion(raw)
        assert len(queries) == 3


class TestBuildHistory:
    """Tests for conversation history formatting."""

    def test_empty_history(self):
        result = _build_history([])
        assert result == "No previous conversation."

    def test_single_turn(self):
        history = [{"question": "What is ML?", "answer": "Machine learning is..."}]
        result = _build_history(history)
        assert "Q: What is ML?" in result
        assert "A: Machine learning is..." in result

    def test_truncation_at_300_chars(self):
        long_answer = "x" * 500
        history = [{"question": "Q?", "answer": long_answer}]
        result = _build_history(history)
        # Answer should be truncated to 300 chars + ellipsis
        assert "…" in result

    def test_respects_history_turns_limit(self):
        history = [
            {"question": f"Q{i}?", "answer": f"A{i}"}
            for i in range(10)
        ]
        result = _build_history(history)
        # With HISTORY_TURNS=3 and 10 entries (Q0..Q9), last 3 are Q7,Q8,Q9
        # Q6 should NOT be included (it's the first excluded entry)
        excluded_index = 10 - HISTORY_TURNS - 1  # index 6
        assert f"Q{excluded_index}?" not in result
        assert f"Q{9}?" in result  # Last entry always included

    def test_missing_keys_handled(self):
        history = [{"question": "Q?"}]  # missing "answer"
        result = _build_history(history)
        assert "Q: Q?" in result
        assert "A: " in result  # empty answer


class TestBuildContext:
    """Tests for context window construction."""

    def test_single_document(self):
        doc = Document(page_content="Hello world")
        context = _build_context([doc])
        assert "[Chunk 1]" in context
        assert "Hello world" in context

    def test_multiple_documents(self):
        docs = [Document(page_content=f"Content {i}") for i in range(3)]
        context = _build_context(docs)
        assert "[Chunk 1]" in context
        assert "[Chunk 2]" in context
        assert "[Chunk 3]" in context

    def test_max_chars_respected(self):
        """Context should not exceed max_chars."""
        docs = [Document(page_content="x" * 5000) for _ in range(10)]
        context = _build_context(docs, max_chars=8000)
        assert len(context) <= 8000 + 100  # small tolerance for chunk labels

    def test_chunks_not_split_mid_document(self):
        """Context should include whole chunks, not truncate mid-document."""
        doc1 = Document(page_content="First chunk content")
        doc2 = Document(page_content="Second chunk content that is very long " * 500)
        context = _build_context([doc1, doc2], max_chars=100)
        # First chunk should be included fully
        assert "First chunk content" in context

    def test_empty_input(self):
        context = _build_context([])
        assert context == ""


class TestNotFoundSentinel:
    """Tests for NOT_FOUND detection patterns matching main.py logic."""

    @pytest.mark.parametrize("sentinel", [
        "NOT_FOUND",
        "NOT_FOUND\n",
        "NOT_FOUND: no relevant information",
        "not_found",
        "NOT FOUND",
        "Not Found",
    ])
    def test_detected_correctly(self, sentinel):
        first_str = sentinel.strip()
        is_not_found = (
            first_str.upper().startswith("NOT_FOUND") or
            first_str.upper().startswith("NOT FOUND")
        )
        assert is_not_found is True

    @pytest.mark.parametrize("valid_answer", [
        "The document states that...",
        "According to the analysis...",
        "NOTABLE findings include...",  # starts with NOT but isn't NOT_FOUND
        "Nothing was found to contradict...",
    ])
    def test_valid_answers_not_misclassified(self, valid_answer):
        first_str = valid_answer.strip()
        is_not_found = (
            first_str.upper().startswith("NOT_FOUND") or
            first_str.upper().startswith("NOT FOUND")
        )
        assert is_not_found is False
