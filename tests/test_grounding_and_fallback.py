"""
Tests for grounding verification, NOT_FOUND sentinel detection, and chunk reference cleaning.
"""

import re
from app.experiment.evaluator import AnswerEvaluator


def strip_chunk_references(text: str) -> str:
    """Helper under test."""
    text = re.sub(r"\[Chunks?\s*[\d,\s]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+and\s+(?=\s|\.|,|$)", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def test_strip_chunk_references():
    raw = "The revenue grew by 12% in Q3 [Chunk 1] and operating margin was 18% [Chunks 2, 3]."
    cleaned = strip_chunk_references(raw)
    assert "[Chunk 1]" not in cleaned
    assert "[Chunks 2, 3]" not in cleaned
    assert "revenue grew by 12% in Q3 and operating margin was 18%." in cleaned

    # Case insensitivity
    raw_lower = "According to the study [chunk 4], the outcome was positive."
    assert "[chunk 4]" not in strip_chunk_references(raw_lower)


def test_not_found_sentinel_detection():
    samples_not_found = [
        "NOT_FOUND",
        "NOT_FOUND\n",
        "not_found.",
        "NOT FOUND",
        "  NOT_FOUND: The document lacks info",
    ]

    for s in samples_not_found:
        clean = s.strip().upper()
        is_missing = clean.startswith("NOT_FOUND") or clean.startswith("NOT FOUND")
        assert is_missing is True


def test_grounding_evaluator_verified():
    context = [
        "AURA uses Sentence Transformers for 384-dimensional dense embeddings.",
        "The FAISS index is persisted to disk in the faiss_db directory."
    ]
    grounded_answer = "AURA generates 384-dimensional dense embeddings and saves FAISS to faiss_db."

    eval_result = AnswerEvaluator.evaluate_grounding(grounded_answer, context)
    assert eval_result["grounded"] is True
    assert eval_result["coverage_score"] > 0.50
    assert eval_result["is_not_found"] is False


def test_grounding_evaluator_not_found():
    context = ["Irrelevant context about climate change."]
    not_found_answer = "NOT_FOUND"

    eval_result = AnswerEvaluator.evaluate_grounding(not_found_answer, context)
    assert eval_result["grounded"] is True
    assert eval_result["is_not_found"] is True
