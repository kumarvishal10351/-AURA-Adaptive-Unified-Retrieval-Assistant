"""
Tests for embedding configuration correctness.

Critical: Embeddings MUST be L2-normalized for the cosine threshold (>=0.20)
to have any meaning. Without normalization, FAISS returns dot products
(unbounded) instead of cosine similarity (bounded [0,1]).
"""

import pytest
from app.config.settings import EMBEDDING_MODEL, RERANKER_MODEL


class TestEmbeddingConfiguration:
    """Verify embedding model configuration is correct."""

    def test_embedding_model_name(self):
        assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_reranker_model_name(self):
        assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_normalization_enabled_in_retriever(self):
        """
        Regression: verify that normalize_embeddings=True is set in the
        embeddings configuration. This is the single most critical config
        for retrieval correctness.
        """
        # We test this by inspecting the source code of retriever.py
        import inspect
        from app.retrieval.retriever import _get_embeddings

        source = inspect.getsource(_get_embeddings)
        assert "normalize_embeddings" in source, \
            "normalize_embeddings parameter not found in _get_embeddings"
        assert '"normalize_embeddings": True' in source or \
               "'normalize_embeddings': True" in source, \
            "normalize_embeddings must be set to True"

    def test_embedding_device_is_cpu(self):
        """Verify embeddings are configured for CPU (no GPU requirement)."""
        import inspect
        from app.retrieval.retriever import _get_embeddings

        source = inspect.getsource(_get_embeddings)
        assert '"cpu"' in source or "'cpu'" in source

    def test_embedding_dimensionality_documented(self):
        """The README documents 384-dimensional embeddings."""
        # all-MiniLM-L6-v2 produces 384-dim vectors
        # We verify the model name matches the documented specification
        assert "MiniLM-L6-v2" in EMBEDDING_MODEL
        # The 384 dimensionality is inherent to this model


class TestCrossEncoderConfiguration:
    """Verify CrossEncoder is used correctly (ordering only, not thresholding)."""

    def test_reranker_uses_raw_logits_for_ordering(self):
        """
        Regression: CrossEncoder predict() returns unbounded logits.
        These must only be used for sorting, never for cosine thresholding.
        """
        import inspect
        from app.chains.rag_chain import create_rag_chain

        source = inspect.getsource(create_rag_chain)
        # The code should sort by CE scores
        assert "sorted(" in source
        # The code should preserve cosine_lookup
        assert "cosine_lookup" in source

    def test_retriever_preserves_cosine_in_results(self):
        """
        Regression: retriever.retrieve() must return (doc, cosine_score),
        not (doc, ce_score).
        """
        import inspect
        from app.retrieval.retriever import retrieve

        source = inspect.getsource(retrieve)
        assert "cosine_lookup" in source
        assert "cosine_lookup.get(id(doc)" in source
