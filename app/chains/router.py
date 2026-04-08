from langchain_core.prompts import ChatPromptTemplate


def is_relevant(llm, question: str, context: str, score: float, threshold: float = 1.4) -> bool:
    """
    Hybrid relevance gate:
      1. Fast embedding-distance filter  (no LLM call needed if score is bad)
      2. LLM semantic judge              (accurate but costs one API call)

    Returns True if the context is relevant to the question.

    Args:
        llm:       LangChain LLM instance.
        question:  The user's question.
        context:   Retrieved context text.
        score:     Best L2 distance from ChromaDB (lower = more similar).
        threshold: Maximum acceptable distance before skipping LLM check.
                   Default 1.4 (tighter than previous 2.0 to reduce false positives).
    """
    # Quick filter — skip LLM call if distance is clearly too large
    if score > threshold:
        return False

    prompt = ChatPromptTemplate.from_template(
        "You are a strict relevance evaluator.\n\n"
        "Does the context below contain a clear answer to the question?\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        'Reply with exactly "YES" or "NO".'
    )

    response = llm.invoke(
        prompt.format(context=context[:2000], question=question)
    )
    return response.content.strip().upper().startswith("YES")