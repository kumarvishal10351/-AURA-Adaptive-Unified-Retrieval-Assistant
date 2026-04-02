from langchain_core.prompts import ChatPromptTemplate

def is_relevant(llm, question, context, score, threshold=2.0):
    """
    Hybrid decision:
    - embedding score
    - LLM semantic check
    """

    # Quick filter (cheap)
    if score > threshold:
        return False

    # LLM semantic judge (accurate)
    prompt = ChatPromptTemplate.from_template("""
    You are a strict evaluator.

    Determine if the answer to the question exists in the given context.

    Context:
    {context}

    Question:
    {question}

    Answer ONLY "YES" or "NO".
    """)

    response = llm.invoke(
        prompt.format(context=context[:2000], question=question)
    )

    decision = response.content.strip().upper()

    return decision == "YES"