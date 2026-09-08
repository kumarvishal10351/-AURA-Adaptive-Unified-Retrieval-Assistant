"""
chains/rag_chain.py
────────────────────
Core RAG Pipeline:
  Stage 1: Parallel original FAISS fetch + LLM multi-query expansion
  Stage 2: Score merge, deduplication, and cosine threshold filter (≥ 0.20)
  Stage 3: CrossEncoder rerank (ms-marco-MiniLM-L-6-v2) for Top-5 ordering
  Stage 4: 16,000-character context assembly with 3-turn conversation history
  Stage 5: Strict grounded token streaming via Mistral Nemo with NOT_FOUND sentinel
"""

from __future__ import annotations

import concurrent.futures
from concurrent.futures import as_completed

from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
    TOP_K,
    FETCH_K,
    FINAL_TOP_N,
    EXPAND_TIMEOUT,
    HISTORY_TURNS,
    MAX_CONTEXT_LENGTH,
    COSINE_THRESHOLD,
)
from retrieval.retriever import get_vectorstore, _get_reranker

# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict document analyst. You answer ONLY from the provided context "
    "chunks extracted from the user's uploaded document. You are NOT a general "
    "knowledge assistant.\n\n"
    "ABSOLUTE RULES (never break these):\n"
    "1. Use ONLY the information explicitly stated in the context chunks below.\n"
    "2. NEVER use your pre-training knowledge, general knowledge, or any information "
    "   not present in the context chunks. This is the most important rule.\n"
    "3. Do NOT add explanations, definitions, or background information from outside "
    "   the document — even if you know the answer from your training data.\n"
    "4. If the context partially answers the question, provide ONLY what the "
    "   context contains and explicitly state: 'The document does not provide "
    "   further details on this topic.'\n"
    "5. Respond with ONLY the word 'NOT_FOUND' (nothing else) when the context "
    "   contains absolutely zero relevant information about the question.\n"
    "6. Do NOT include chunk references like [Chunk 1] or [Chunk 2] in your answer. "
    "   Write naturally as if summarizing the document.\n"
    "7. Never invent, infer, or extrapolate facts beyond what is explicitly written "
    "   in the context chunks.\n"
    "8. Structure your response with clear formatting (bullet points, headings) "
    "   when the content warrants it.\n\n"
    "Conversation history (last {history_turns} turns for context):\n"
    "{history}\n\n"
    "Context chunks from the user's uploaded document:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer (from document only):"
)

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "You are generating search queries for a vector database.\n"
    "Produce 3 alternative phrasings of the question below to maximise recall.\n"
    "Use conversation history to resolve pronouns (it, they, this, the document, etc.).\n"
    "Return ONLY the 3 queries, one per line. No numbering, bullets, or commentary.\n\n"
    "History: {history}\n"
    "Question: {question}"
)


def _build_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for h in history[-HISTORY_TURNS:]:
        ans = h.get("answer", "")
        preview = ans[:300].rstrip()
        if len(ans) > 300:
            preview += "…"
        lines.append(f"Q: {h.get('question', '')}\nA: {preview}")
    return "\n\n".join(lines)


def _parse_expansion(raw: str) -> list[str]:
    queries = []
    for line in raw.split("\n"):
        clean = line.strip()
        for prefix in ("1.", "2.", "3.", "1)", "2)", "3)", "-", "*", "•"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
                break
        if clean:
            queries.append(clean)
    return queries[:3]


def _fetch_candidates(vs, query: str, k: int = FETCH_K) -> dict[str, tuple]:
    """
    FAISS search returning {content: (doc, cosine_score)}.
    """
    try:
        results = vs.similarity_search_with_relevance_scores(query, k=k)
        return {doc.page_content: (doc, float(score)) for doc, score in results}
    except Exception:
        return {}


def _build_context(docs: list, max_chars: int = MAX_CONTEXT_LENGTH) -> str:
    """Build labelled chunk context, bounded at chunk borders within max_chars."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs):
        chunk = f"[Chunk {i + 1}]\n{doc.page_content}"
        if parts and total + len(chunk) + 2 > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts)


def create_rag_chain(llm, vectorstore=None):
    """
    Returns rag_pipeline(question, history) -> (generator, docs, results).
    results is always List[Tuple[doc, cosine_score: float]] in [0, 1].
    """

    def rag_pipeline(question: str, history: list):
        vs = vectorstore or get_vectorstore()
        history_text = _build_history(history)

        # ── Stage 1: Parallel original fetch + query expansion ────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            orig_future = ex.submit(_fetch_candidates, vs, question)
            expand_future = ex.submit(
                llm.invoke,
                _REWRITE_PROMPT.format_messages(
                    history=history_text, question=question
                ),
            )

            try:
                merged: dict[str, tuple] = orig_future.result(timeout=30)
            except Exception:
                merged = {}

            extra_queries: list[str] = []
            try:
                rewrite_res = expand_future.result(timeout=EXPAND_TIMEOUT)
                extra_queries = _parse_expansion(getattr(rewrite_res, "content", str(rewrite_res)))
            except Exception:
                pass  # Fallback to original query candidates

            if extra_queries:
                extra_futures = {
                    ex.submit(_fetch_candidates, vs, q): q
                    for q in extra_queries
                }
                for fut in as_completed(extra_futures, timeout=25):
                    try:
                        partial = fut.result()
                        for content, (doc, score) in partial.items():
                            if content not in merged or score > merged[content][1]:
                                merged[content] = (doc, score)
                    except Exception:
                        continue

        if not merged:
            def _empty():
                yield "NOT_FOUND"
            return _empty(), [], []

        # ── Stage 2: Cosine threshold filter ──────────────────────────────
        above = [
            (doc, score)
            for _, (doc, score) in merged.items()
            if score >= COSINE_THRESHOLD
        ]

        if not above:
            above = sorted(
                [(doc, score) for _, (doc, score) in merged.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:FETCH_K]

        docs_to_rerank = [doc for doc, _ in above]
        cosine_lookup = {id(doc): score for doc, score in above}

        # ── Stage 3: CrossEncoder rerank (ordering only) ───────────────────
        try:
            reranker = _get_reranker()
            pairs = [[question, doc.page_content] for doc in docs_to_rerank]
            ce_scores = reranker.predict(pairs)

            ranked = sorted(
                zip(ce_scores, docs_to_rerank),
                key=lambda x: x[0],
                reverse=True,
            )[:FINAL_TOP_N]

            final_docs = [doc for _, doc in ranked]

        except Exception:
            final_docs = [
                doc for doc, _ in sorted(above, key=lambda x: x[1], reverse=True)
            ][:FINAL_TOP_N]

        results = [
            (doc, cosine_lookup.get(id(doc), 0.0))
            for doc in final_docs
        ]

        # ── Stage 4: Build context + stream answer ────────────────────────
        context = _build_context(final_docs)

        formatted_prompt = _ANSWER_PROMPT.format_messages(
            context=context,
            question=question,
            history=history_text,
            history_turns=HISTORY_TURNS,
        )

        def token_generator():
            has_yielded = False
            try:
                for chunk in llm.stream(formatted_prompt):
                    c = getattr(chunk, "content", str(chunk))
                    if c:
                        has_yielded = True
                        yield c
            except Exception as exc:
                yield f"\n\n[Generation error]: {exc}"
                has_yielded = True

            if not has_yielded:
                yield "NOT_FOUND"

        return token_generator(), final_docs, results

    return rag_pipeline
