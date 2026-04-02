def calculate_confidence(results):
    """
    Convert similarity score into confidence %
    Lower score = higher confidence
    """

    if not results:
        return 0

    best_score = results[0][1]

    # Convert score → confidence (simple normalization)
    confidence = max(0, 1 - (best_score / 5))

    return round(confidence * 100, 2)