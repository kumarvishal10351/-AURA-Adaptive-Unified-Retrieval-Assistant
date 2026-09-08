"""
chains/router.py
────────────────
Hybrid relevance gate combining fast vector similarity thresholding
with an optional LLM semantic judge.
"""

from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from config.settings import COSINE_THRESHOLD


def is_relevant(llm, question: str, context: str, score: float, threshold: float = COSINE_THRESHOLD) -> bool:
    """
    Hybrid relevance gate:
      1. Fast cosine similarity filter (rejects immediately if score < threshold).
      2. LLM semantic judge (checks if context explicitly answers question).

    Parameters
    ----------
    llm : LangChain LLM instance.
    question : User query.
    context : Retrieved context string.
    score : Cosine similarity score [0, 1].
    threshold : Minimum acceptable cosine score. Default 0.20.

    Returns
    -------
    bool
        True if the context is relevant to the question.
    """
    # Fast filter: skip LLM call if score is below minimum threshold
    if score < threshold:
        return False

    prompt = ChatPromptTemplate.from_template(
        "You are a strict relevance evaluator.\n\n"
        "Does the context below contain a clear answer to the question?\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        'Reply with exactly "YES" or "NO".'
    )

    try:
        response = llm.invoke(
            prompt.format(context=context[:2000], question=question)
        )
        content = getattr(response, "content", str(response)).strip().upper()
        return content.startswith("YES")
    except Exception:
        # If LLM check fails, fall back to cosine score
        return score >= threshold