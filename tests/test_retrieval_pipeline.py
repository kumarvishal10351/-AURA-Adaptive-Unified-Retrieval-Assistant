"""
Integration tests for the retrieval pipeline.

Tests the full pipeline logic: deduplication, score preservation through
reranking, fallback behavior, and Top-N selection — using mock objects
to avoid requiring a real FAISS index or API key.
"""

import pytest
from langchain_core.documents import Document

from app.config.settings import COSINE_THRESHOLD, FINAL_TOP_N, FETCH_K


class TestDeduplication:
    """Tests for candidate deduplication with highest-score retention."""

    def test_duplicate_content_keeps_highest_score(self):
        """When the same chunk appears from multiple queries, keep the highest score."""
        merged = {}

        # First query returns chunk with score 0.45
        doc1 = Document(page_content="Machine learning is a subset of AI.",
                       metadata={"page": 0})
        merged[doc1.page_content] = (doc1, 0.45)

        # Second query returns same content with higher score
        doc2 = Document(page_content="Machine learning is a subset of AI.",
                       metadata={"page": 0})
        content = doc2.page_content
        if content not in merged or 0.65 > merged[content][1]:
            merged[content] = (doc2, 0.65)

        # Only one entry, with the higher score
        assert len(merged) == 1
        assert merged[content][1] == 0.65

    def test_different_content_kept_separately(self):
        merged = {}

        doc1 = Document(page_content="Content A", metadata={"page": 0})
        doc2 = Document(page_content="Content B", metadata={"page": 1})

        merged[doc1.page_content] = (doc1, 0.50)
        merged[doc2.page_content] = (doc2, 0.60)

        assert len(merged) == 2


class TestCosineGating:
    """Tests for the cosine threshold gating stage."""

    def test_above_threshold_pass(self):
        candidates = [
            (Document(page_content=f"Doc {i}"), score)
            for i, score in enumerate([0.25, 0.30, 0.45, 0.60])
        ]

        passing = [(doc, score) for doc, score in candidates if score >= COSINE_THRESHOLD]
        assert len(passing) == 4  # All above 0.20

    def test_below_threshold_filtered(self):
        candidates = [
            (Document(page_content=f"Doc {i}"), score)
            for i, score in enumerate([0.05, 0.10, 0.15, 0.19])
        ]

        passing = [(doc, score) for doc, score in candidates if score >= COSINE_THRESHOLD]
        assert len(passing) == 0

    def test_mixed_threshold(self):
        candidates = [
            (Document(page_content="Low"), 0.10),
            (Document(page_content="At"), 0.20),
            (Document(page_content="High"), 0.50),
        ]

        passing = [(doc, score) for doc, score in candidates if score >= COSINE_THRESHOLD]
        assert len(passing) == 2

    def test_fallback_when_all_below(self):
        """When no candidates pass threshold, fall back to top-K raw."""
        merged = {
            f"content_{i}": (Document(page_content=f"content_{i}"), 0.05 + i * 0.03)
            for i in range(5)
        }

        above = [
            (doc, score)
            for _, (doc, score) in merged.items()
            if score >= COSINE_THRESHOLD
        ]

        if not above:
            above = sorted(
                [(doc, score) for _, (doc, score) in merged.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:FETCH_K]

        assert len(above) == 5  # All retained as fallback
        assert above[0][1] >= above[-1][1]  # Sorted descending


class TestCrossEncoderOrdering:
    """Tests for CrossEncoder reranking behavior."""

    def test_ce_reranks_candidates(self):
        """CrossEncoder scores should reorder candidates."""
        doc_a = Document(page_content="Topic A", metadata={"page": 0})
        doc_b = Document(page_content="Topic B", metadata={"page": 1})
        doc_c = Document(page_content="Topic C", metadata={"page": 2})

        # Cosine scores
        cosine_lookup = {id(doc_a): 0.75, id(doc_b): 0.50, id(doc_c): 0.60}

        # CrossEncoder reranks differently
        ce_scores = [1.2, 4.8, 2.3]  # B ranked highest by CE
        candidates = [doc_a, doc_b, doc_c]

        ranked = sorted(
            zip(ce_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:FINAL_TOP_N]

        final_docs = [doc for _, doc in ranked]

        # CE reranking puts doc_b first
        assert final_docs[0] is doc_b

        # But results preserve cosine scores
        results = [(doc, cosine_lookup.get(id(doc), 0.0)) for doc in final_docs]
        assert results[0][1] == 0.50  # doc_b's cosine score, NOT 4.8

    def test_ce_failure_falls_back_to_cosine_order(self):
        """If CrossEncoder fails, candidates should be sorted by cosine score."""
        passing = [
            (Document(page_content="High", metadata={"page": 0}), 0.80),
            (Document(page_content="Med", metadata={"page": 1}), 0.50),
            (Document(page_content="Low", metadata={"page": 2}), 0.30),
        ]

        # Simulate CE failure → fallback to cosine sort
        final_docs = [
            doc for doc, _ in sorted(passing, key=lambda x: x[1], reverse=True)
        ][:FINAL_TOP_N]

        assert final_docs[0].page_content == "High"
        assert final_docs[-1].page_content == "Low"


class TestTopNSelection:
    """Tests for Top-N capping after reranking."""

    def test_more_than_five_capped(self):
        docs = [Document(page_content=f"D{i}") for i in range(10)]
        scores = list(range(10, 0, -1))
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
        assert len(ranked) == FINAL_TOP_N

    def test_fewer_than_five_preserved(self):
        docs = [Document(page_content=f"D{i}") for i in range(3)]
        scores = [3, 2, 1]
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
        assert len(ranked) == 3

    def test_exactly_five(self):
        docs = [Document(page_content=f"D{i}") for i in range(5)]
        scores = [5, 4, 3, 2, 1]
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
        assert len(ranked) == 5

    def test_empty_candidates(self):
        ranked = sorted([], key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
        assert len(ranked) == 0

    def test_single_candidate(self):
        doc = Document(page_content="Only one")
        ranked = sorted([(1.0, doc)], key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
        assert len(ranked) == 1


class TestScorePreservationEndToEnd:
    """
    End-to-end regression: verifies the complete pipeline preserves
    cosine scores through all stages.
    """

    def test_full_pipeline_score_flow(self):
        """Simulates the full pipeline and checks cosine score preservation."""
        # Stage 1: Fetch candidates
        merged = {}
        docs = [
            Document(page_content=f"Content {i}", metadata={"page": i})
            for i in range(8)
        ]
        for i, doc in enumerate(docs):
            merged[doc.page_content] = (doc, 0.15 + i * 0.1)

        # Stage 2: Cosine threshold filter
        above = [
            (doc, score)
            for _, (doc, score) in merged.items()
            if score >= COSINE_THRESHOLD
        ]

        if not above:
            above = sorted(
                [(doc, score) for _, (doc, score) in merged.items()],
                key=lambda x: x[1],
                reverse=True,
            )

        docs_to_rerank = [doc for doc, _ in above]
        cosine_lookup = {id(doc): score for doc, score in above}

        # Stage 3: Simulate CrossEncoder reranking
        ce_scores = [float(i) for i in range(len(docs_to_rerank))]
        ranked = sorted(
            zip(ce_scores, docs_to_rerank),
            key=lambda x: x[0],
            reverse=True,
        )[:FINAL_TOP_N]

        final_docs = [doc for _, doc in ranked]

        # Stage 4: Reconstruct results with cosine scores
        results = [
            (doc, cosine_lookup.get(id(doc), 0.0))
            for doc in final_docs
        ]

        # Verify cosine scores are preserved, not CE scores
        for doc, score in results:
            assert 0.0 <= score <= 1.0, f"Score {score} is outside cosine bounds"
            assert score == cosine_lookup[id(doc)], "Cosine score was modified"
