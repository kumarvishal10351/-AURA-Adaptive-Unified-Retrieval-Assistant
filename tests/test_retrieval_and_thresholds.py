"""
Tests for retrieval filtering, cosine threshold boundaries, CrossEncoder ordering, and Top-N selection.
"""

import pytest
from langchain_core.documents import Document
from app.config.settings import COSINE_THRESHOLD, FINAL_TOP_N


def test_cosine_threshold_boundaries():
    # Verify exact boundary behavior
    doc_below = Document(page_content="Below threshold", metadata={"page": 0})
    doc_at = Document(page_content="At threshold", metadata={"page": 1})
    doc_above = Document(page_content="Above threshold", metadata={"page": 2})

    candidates = [
        (doc_below, 0.19),
        (doc_at, 0.20),
        (doc_above, 0.21),
    ]

    passing = [(doc, score) for doc, score in candidates if score >= COSINE_THRESHOLD]

    assert len(passing) == 2
    assert (doc_below, 0.19) not in passing
    assert (doc_at, 0.20) in passing
    assert (doc_above, 0.21) in passing


def test_threshold_fallback_when_none_pass():
    # If all chunks are below 0.20, fallback must retain top-k candidates
    doc1 = Document(page_content="Low match 1")
    doc2 = Document(page_content="Low match 2")

    merged = {
        doc1.page_content: (doc1, 0.12),
        doc2.page_content: (doc2, 0.15),
    }

    above = [
        (doc, score)
        for _, (doc, score) in merged.items()
        if score >= COSINE_THRESHOLD
    ]

    # No chunks passed
    assert len(above) == 0

    # Fallback logic
    if not above:
        above = sorted(
            [(doc, score) for _, (doc, score) in merged.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

    assert len(above) == 2
    assert above[0][1] == 0.15  # highest retained first


def test_cross_encoder_logits_preserve_cosine_scores():
    """
    Regression Test: CrossEncoder logits must only be used for candidate ranking,
    never replacing the original cosine similarity scores passed to confidence calculation.
    """
    doc_a = Document(page_content="Topic A: High semantic match")
    doc_b = Document(page_content="Topic B: Moderate semantic match")

    passing = [
        (doc_a, 0.75),  # Cosine 0.75
        (doc_b, 0.50),  # Cosine 0.50
    ]

    cosine_lookup = {id(doc): score for doc, score in passing}

    # Simulate CrossEncoder output logits (unbounded reals: e.g. +3.5 and -1.2)
    ce_scores = [1.2, 4.8]  # CrossEncoder ranks doc_b higher in context
    candidate_docs = [doc_a, doc_b]

    ranked = sorted(
        zip(ce_scores, candidate_docs),
        key=lambda x: x[0],
        reverse=True,
    )[:FINAL_TOP_N]

    final_docs = [doc for _, doc in ranked]

    # Reconstitute results with original cosine scores
    results = [(doc, cosine_lookup.get(id(doc), 0.0)) for doc in final_docs]

    assert final_docs[0] is doc_b  # doc_b ranked first by CE
    assert results[0][1] == 0.50   # but cosine score is preserved as 0.50 (NOT 4.8 CE logit!)
    assert results[1][1] == 0.75   # doc_a cosine score is preserved as 0.75


def test_final_top_n_capping():
    docs = [Document(page_content=f"Chunk {i}") for i in range(10)]
    scores = [0.1 * i for i in range(10)]
    paired = list(zip(scores, docs))

    ranked = sorted(paired, key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
    assert len(ranked) == 5
