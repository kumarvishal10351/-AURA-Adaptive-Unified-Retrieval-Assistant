from langchain_core.prompts import ChatPromptTemplate
from config.settings import TOP_K, MAX_CONTEXT_LENGTH


def create_rag_chain(llm, vectorstore):

    # ─── Prompt ───────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful, accurate assistant.\n\n"
        "Answer the question using ONLY the provided context below.\n"
        "If the answer is not present in the context, respond with exactly: NOT_FOUND\n"
        "Do NOT guess or make up information.\n\n"
        "Conversation History (for context):\n"
        "{history}\n\n"
        "Context from document:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def format_docs(docs) -> str:
        return "\n\n".join(
            f"[Chunk {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    def build_history(history: list) -> str:
        """
        Build a compact history string.
        Truncate each previous answer to 300 chars to avoid wasting context.
        """
        if not history:
            return "No previous conversation."
        lines = []
        for h in history[-3:]:          # last 3 turns only
            ans_preview = h["answer"][:300].rstrip()
            if len(h["answer"]) > 300:
                ans_preview += "…"
            lines.append(f"Q: {h['question']}\nA: {ans_preview}")
        return "\n\n".join(lines)

    # ─── Pipeline ─────────────────────────────────────────────────────────────
    def rag_pipeline(question: str, history: list):
        # Retrieve with scores
        results = vectorstore.similarity_search_with_score(question, k=TOP_K)

        if not results:
            return "NOT_FOUND", [], []

        docs = [doc for doc, _ in results]

        # Build context with chunk labels (easier for the LLM to reason over)
        context = format_docs(docs)
        context = context[:MAX_CONTEXT_LENGTH]

        history_text = build_history(history)

        try:
            response = llm.invoke(
                prompt.format(
                    context=context,
                    question=question,
                    history=history_text,
                )
            )
            return response.content, docs, results
        except TimeoutError as e:
            raise TimeoutError(f"LLM request timed out after 30 seconds. The API may be overloaded. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error while invoking LLM: {str(e)}")

    return rag_pipeline