from config.settings import CONFIDENCE_DIVISOR


def calculate_confidence(results) -> int:
    """
    Convert ChromaDB similarity scores into a confidence percentage.

    ChromaDB returns L2 distances where lower = better match.
    Typical range: 0 (perfect) to 2 (very different).

    Improvement over previous version:
    - Averages the top results instead of using only the best score.
    - This gives a more realistic picture of overall retrieval quality.
    """
    if not results:
        return 0

    # Weight towards the best scores: use top-3 (or fewer if less available)
    top_results = results[:3]
    scores = [score for _, score in top_results]
    avg_score = sum(scores) / len(scores)

    # Normalise: 0 distance → 100%, 2 distance → 0%
    confidence = max(0.0, min(1.0, 1.0 - (avg_score / CONFIDENCE_DIVISOR)))
    return round(confidence * 100)