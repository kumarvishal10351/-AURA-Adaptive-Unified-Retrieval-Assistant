"""
utils/confidence.py
────────────────────
Converts retrieval results into an intuitive 0–100 integer confidence score.

Formula:
  confidence = floor(CONFIDENCE_FLOOR + (top_cosine / COSINE_FULL_CONFIDENCE) * (100 - CONFIDENCE_FLOOR))
  clamped strictly to [0, 100].

Results format:
  results: list of (doc, cosine_score: float) where cosine_score ∈ [0, 1].
"""

from __future__ import annotations

from config.settings import COSINE_FULL_CONFIDENCE, CONFIDENCE_FLOOR


def calculate_confidence(results: list) -> int:
    """
    Compute a 0–100 integer confidence score from retrieval results.

    Parameters
    ----------
    results : list of (doc, cosine_score: float)
        Output of rag_pipeline — always cosine scores in [0, 1].
        Empty list or missing score → returns 0.

    Returns
    -------
    int
        Confidence percentage in [0, 100].
    """
    if not results:
        return 0

    try:
        _, top_score = results[0]
        top_cosine = float(top_score)
    except (TypeError, ValueError, IndexError):
        return 0

    # Ensure cosine is bounded [0, 1]
    top_cosine = max(0.0, min(1.0, top_cosine))

    scale_range = 100.0 - CONFIDENCE_FLOOR
    raw = CONFIDENCE_FLOOR + (top_cosine / COSINE_FULL_CONFIDENCE) * scale_range
    return max(0, min(100, int(raw)))


def confidence_level(score: int) -> str:
    """
    Return qualitative confidence band for user interface and metrics.

    Thresholds:
      ≥ 80 → High (Green)
      60–79 → Medium (Amber)
      1–59 → Low (Red)
      ≤ 0 → None
    """
    if score <= 0:
        return "None"
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def average_confidence(scores: list[int]) -> int:
    """Return the integer average of a list of confidence scores."""
    if not scores:
        return 0
    valid = [s for s in scores if isinstance(s, (int, float)) and s > 0]
    return int(sum(valid) / len(valid)) if valid else 0