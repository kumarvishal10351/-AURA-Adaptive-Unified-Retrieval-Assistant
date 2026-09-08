"""
Tests for UI helper functions and session state logic.

Tests the deterministic functions from main.py that can be tested
without Streamlit runtime.
"""

import re
import pytest


# Reproduce the exact functions from main.py for isolated testing
def strip_chunk_references(text: str) -> str:
    """Remove [Chunk X], [Chunks X, Y], and stray references from answer text."""
    text = re.sub(r"\[Chunks?\s*[\d,\s]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+and\s+(?=\s|\.|,|$)", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def avg_confidence(scores: list) -> int:
    valid = [x for x in scores if isinstance(x, (int, float)) and x > 0]
    return int(sum(valid) / len(valid)) if valid else 0


class TestStripChunkReferences:
    """Tests for chunk reference cleaning in displayed answers."""

    def test_single_chunk_ref(self):
        raw = "The result is 42 [Chunk 1]."
        assert "[Chunk 1]" not in strip_chunk_references(raw)
        assert "42" in strip_chunk_references(raw)

    def test_multi_chunk_ref(self):
        raw = "Revenue grew [Chunks 2, 3] and profit increased [Chunk 1]."
        cleaned = strip_chunk_references(raw)
        assert "[Chunks 2, 3]" not in cleaned
        assert "[Chunk 1]" not in cleaned
        assert "Revenue grew" in cleaned

    def test_case_insensitive(self):
        raw = "Data from [chunk 4] indicates growth."
        cleaned = strip_chunk_references(raw)
        assert "[chunk 4]" not in cleaned

    def test_no_refs_unchanged(self):
        raw = "A clean sentence with no references."
        assert strip_chunk_references(raw) == raw

    def test_empty_string(self):
        assert strip_chunk_references("") == ""

    def test_only_refs(self):
        result = strip_chunk_references("[Chunk 1]")
        assert result == ""


class TestAvgConfidence:
    """Tests for rolling average confidence calculation."""

    def test_empty_scores(self):
        assert avg_confidence([]) == 0

    def test_single_score(self):
        assert avg_confidence([80]) == 80

    def test_multiple_scores(self):
        assert avg_confidence([80, 90, 70]) == 80

    def test_zero_excluded(self):
        """Zero scores (from NOT_FOUND) should be excluded from average."""
        assert avg_confidence([100, 0, 80]) == 90

    def test_negative_excluded(self):
        assert avg_confidence([-10, 80]) == 80

    def test_all_zeros(self):
        assert avg_confidence([0, 0, 0]) == 0

    def test_mixed_types(self):
        assert avg_confidence([80, 90.5]) == 85  # int((80 + 90.5) / 2)


class TestSourceFormatting:
    """Tests for source attribution page indexing."""

    def test_zero_indexed_to_one_indexed(self):
        """PyMuPDF page 0 should display as page 1."""
        page_raw = 0
        page_display = (page_raw + 1) if isinstance(page_raw, int) else page_raw
        assert page_display == 1

    def test_page_nine_to_ten(self):
        page_raw = 9
        page_display = (page_raw + 1) if isinstance(page_raw, int) else page_raw
        assert page_display == 10

    def test_non_integer_page_preserved(self):
        """Non-integer page values should be passed through."""
        page_raw = "?"
        page_display = (page_raw + 1) if isinstance(page_raw, int) else page_raw
        assert page_display == "?"

    def test_none_page_default(self):
        metadata = {}
        page_raw = metadata.get("page", 0)
        page_display = (page_raw + 1) if isinstance(page_raw, int) else page_raw
        assert page_display == 1  # 0 + 1


class TestConfidenceBar:
    """Tests for the visual confidence bar formatting."""

    def test_confidence_bar_0(self):
        from app.utils.confidence import confidence_level
        value = 0
        filled = round(value / 10)
        empty = 10 - filled
        bar = "\u2588" * filled + "\u2591" * empty
        level = confidence_level(value)
        result = f"`[{bar}] {value}%` ({level})"
        assert "0%" in result
        assert "None" in result

    def test_confidence_bar_100(self):
        from app.utils.confidence import confidence_level
        value = 100
        filled = round(value / 10)
        empty = 10 - filled
        bar = "\u2588" * filled + "\u2591" * empty
        level = confidence_level(value)
        result = f"`[{bar}] {value}%` ({level})"
        assert "100%" in result
        assert "High" in result
        assert "\u2591" not in result  # All filled

    def test_confidence_bar_50(self):
        from app.utils.confidence import confidence_level
        value = 50
        filled = round(value / 10)
        assert filled == 5
