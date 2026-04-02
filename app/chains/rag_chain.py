from langchain_core.prompts import ChatPromptTemplate

def create_rag_chain(llm, vectorstore):

    # -------------------------------
    # Prompt (STRICT + MEMORY AWARE)
    # -------------------------------
    prompt = ChatPromptTemplate.from_template("""
You are a strict assistant.

You MUST answer ONLY using the provided context.

Use the conversation history to understand the question better.

If the answer is NOT clearly present in the context,
you MUST respond with exactly:

NOT_FOUND

Do NOT explain.
Do NOT guess.

Conversation History:
{history}

Context:
{context}

Question:
{question}
""")

    # -------------------------------
    # Format Documents
    # -------------------------------
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # -------------------------------
    # RAG Pipeline
    # -------------------------------
    def rag_pipeline(question, history):

        # 🔍 Retrieve documents with scores
        results = vectorstore.similarity_search_with_score(question, k=5)

        docs = [doc for doc, score in results]

        # 📄 Build context (limit size for safety)
        context = format_docs(docs)
        context = context[:3000]

        # 🧠 Build conversation history (last 3 turns)
        history_text = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in history[-3:]]
        )

        # 🤖 LLM call
        response = llm.invoke(
            prompt.format(
                context=context,
                question=question,
                history=history_text
            )
        )

        return response.content, docs, results

    # -------------------------------
    return rag_pipeline