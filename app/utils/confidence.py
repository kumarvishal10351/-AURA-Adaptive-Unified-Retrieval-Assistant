"""
utils/confidence.py
────────────────────
Converts retrieval results into a 0–100 integer confidence score.

Root cause of the "1% confidence" bug
────────────────────────────────────────
The previous implementation (inferred from the screenshot) was passing
CrossEncoder logit scores directly into the confidence calculation.
CrossEncoder logits from ms-marco-MiniLM-L-6-v2 are unbounded reals —
a perfectly good match can score -1 to +3, a poor match scores -5 to -10.
Passing sigmoid(-4.6) * 100 = 1% is mathematically correct for the raw logit
but completely misleading as a user-facing confidence percentage.

The fix
─────────
rag_chain.py now always returns results = [(doc, cosine_score: float)]
where cosine_score ∈ [0, 1] (FAISS relevance score on L2-normalised embeddings).
This file no longer needs to guess the score type.

Scaling strategy
─────────────────
Raw cosine similarity has a counter-intuitive range for this use case:
  - 0.95–1.00 = nearly identical text (only happens for exact re-queries)
  - 0.50–0.80 = good, relevant match
  - 0.20–0.50 = partial / related match
  - 0.00–0.20 = noise / off-topic

Reporting 0.55 as "55% confidence" is too pessimistic — users see a
misleadingly low number for what is actually a strong result.

We rescale the top-chunk cosine score to a more intuitive range:
  - cosine 0.80+ → 90–100%   (excellent match)
  - cosine 0.60  → ~75%      (good match)
  - cosine 0.40  → ~55%      (partial match)
  - cosine 0.20  → ~35%      (weak match — should show "Low")
  - cosine 0.00  → 20%       (floor — something is always better than nothing)

Formula: confidence = 20 + (top_cosine / 0.80) * 80, clamped to [0, 100].
The 0.80 divisor treats a cosine of 0.80 as "full confidence" (100 before clamp).
"""

from __future__ import annotations


def calculate_confidence(results: list) -> int:
    """
    Compute a 0–100 integer confidence score from retrieval results.

    Parameters
    ----------
    results : list of (doc, cosine_score: float)
        Output of rag_pipeline — always cosine scores in [0, 1].
        Empty list → returns 0.

    Returns
    -------
    int
        Confidence in [0, 100].  Thresholds used by the UI:
          ≥ 80 → High (green)
          60–79 → Medium (amber)
          < 60 → Low (red)
    """
    if not results:
        return 0

    # Use the top-ranked chunk's cosine score as the signal.
    # The reranker has already put the most relevant chunk first.
    try:
        _, top_score = results[0]
        top_cosine   = float(top_score)
    except (TypeError, ValueError, IndexError):
        return 0

    # Sanity-clamp: cosine scores should be 0–1 with normalised embeddings.
    # Guard against any unexpected values without crashing.
    top_cosine = max(0.0, min(1.0, top_cosine))

    # Rescale so cosine=0.80 maps to 100% confidence.
    # Floor at 20 so even weak (but non-zero) matches show something.
    raw = 20.0 + (top_cosine / 0.80) * 80.0
    return min(100, int(raw))


def average_confidence(scores: list[int]) -> int:
    """Return the integer average of a list of confidence scores."""
    if not scores:
        return 0
    return int(sum(scores) / len(scores))