"""
experiment/evaluator.py
───────────────────────
Evaluation harness for RAG response grounding, faithfulness, and retrieval relevance.
"""

from __future__ import annotations
from typing import Sequence


class AnswerEvaluator:
    """
    Assesses answer grounding against retrieved source documents.
    """

    @staticmethod
    def evaluate_grounding(answer: str, context_chunks: Sequence[str]) -> dict[str, float | bool]:
        """
        Check whether answer keywords and n-grams are attested within the context chunks.
        """
        if not answer or not context_chunks:
            return {
                "grounded": False,
                "coverage_score": 0.0,
                "is_not_found": answer.strip().upper().startswith("NOT_FOUND"),
            }

        if answer.strip().upper().startswith("NOT_FOUND"):
            return {
                "grounded": True,
                "coverage_score": 1.0,
                "is_not_found": True,
            }

        full_context = " ".join(context_chunks).lower()
        words = [w.lower().strip(".,;:()[]\"'") for w in answer.split() if len(w) > 3]

        if not words:
            return {"grounded": True, "coverage_score": 1.0, "is_not_found": False}

        matched = sum(1 for w in words if w in full_context)
        coverage = round(matched / len(words), 3)

        return {
            "grounded": coverage >= 0.50,
            "coverage_score": coverage,
            "is_not_found": False,
        }