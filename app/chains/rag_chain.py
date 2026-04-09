"""
chains/rag_chain.py
────────────────────
Fixes vs previous version:
  • Two separate thresholds: COSINE_THRESHOLD on FAISS scores (0–1 range),
    CrossEncoder used for ranking ONLY — no CE-score gating.
    The root bug was using _SCORE_THRESHOLD (calibrated for cosine) as a
    post-CE gate; CE logits are unbounded so it wiped valid chunks for any
    broad / summary question.
  • results is now always List[Tuple[doc, cosine_score: float]] — consistent
    regardless of whether CE rerank ran or fell back, so confidence.py
    always receives 0–1 values it can scale correctly.
  • Prompt rewritten: less aggressive NOT_FOUND trigger; instructs LLM to
    give partial answers rather than giving up on broad questions.
  • NOT_FOUND detection hardened in app.py (see note at bottom of file).
  • Parallel architecture and expansion timeout unchanged.
"""

from __future__ import annotations

import concurrent.futures
from concurrent.futures import as_completed

from langchain_core.prompts import ChatPromptTemplate

from config.settings import TOP_K, MAX_CONTEXT_LENGTH
from retrieval.retriever import get_vectorstore, _get_reranker

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Applied to FAISS cosine scores (0–1 range with normalize_embeddings=True).
# 0.20 is intentionally lower than the old 0.30 to prevent over-filtering
# on broad questions like "summarise the document" or "what are the conclusions".
COSINE_THRESHOLD: float = 0.20

FETCH_K: int        = max(TOP_K * 2, 12)  # candidates per query variation
FINAL_TOP_N: int    = 5                   # chunks sent to the LLM
EXPAND_TIMEOUT: int = 18                  # seconds for LLM expansion call
HISTORY_TURNS: int  = 3                   # recent turns included in prompt

# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise document analyst. Your job is to answer the user's question "
    "using the context chunks retrieved from their uploaded document.\n\n"
    "RULES:\n"
    "1. Use ONLY information present in the provided context chunks.\n"
    "2. Cite every chunk you draw from, e.g. [Chunk 2] or [Chunks 1, 3].\n"
    "3. If a chunk is partially relevant, use what is relevant and note it.\n"
    "4. If the chunks relate to the question but don't answer it fully, give the "
    "   best partial answer you can and explain what context is missing.\n"
    "5. Respond with ONLY the word 'NOT_FOUND' (nothing else, no punctuation) "
    "   ONLY when the context contains absolutely zero information on the topic. "
    "   This should be very rare — prefer a partial answer with caveats.\n"
    "6. Never invent or infer facts not present in the context chunks.\n\n"
    "Conversation history (last {history_turns} turns for context):\n"
    "{history}\n\n"
    "Context chunks from document:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "You are generating search queries for a vector database.\n"
    "Produce 3 alternative phrasings of the question below to maximise recall.\n"
    "Use conversation history to resolve pronouns (it, they, this, the document, etc.).\n"
    "Return ONLY the 3 queries, one per line. No numbering, bullets, or commentary.\n\n"
    "History: {history}\n"
    "Question: {question}"
)

# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _build_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for h in history[-HISTORY_TURNS:]:
        preview = h["answer"][:300].rstrip()
        if len(h["answer"]) > 300:
            preview += "…"
        lines.append(f"Q: {h['question']}\nA: {preview}")
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
    FAISS search. Returns {content: (doc, cosine_score)}.
    Keyed by content for dedup; callers keep highest score on collision.
    """
    try:
        results = vs.similarity_search_with_relevance_scores(query, k=k)
        return {doc.page_content: (doc, float(score)) for doc, score in results}
    except Exception:
        return {}


def _build_context(docs: list, max_chars: int = MAX_CONTEXT_LENGTH) -> str:
    """Build labelled chunk context, truncating at a chunk boundary."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs):
        chunk = f"[Chunk {i + 1}]\n{doc.page_content}"
        if parts and total + len(chunk) + 2 > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────

def create_rag_chain(llm, vectorstore):
    """
    Returns rag_pipeline(question, history) -> (generator, docs, results).

    results is always List[Tuple[doc, cosine_score: float]] so that
    calculate_confidence() in utils/confidence.py always receives 0–1
    values and can produce a meaningful percentage.
    """

    def rag_pipeline(question: str, history: list):
        vs           = get_vectorstore()
        history_text = _build_history(history)

        # ── Stage 1: parallel original fetch + query expansion ────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:

            # Kick off FAISS search for the raw question immediately
            orig_future = ex.submit(_fetch_candidates, vs, question)

            # Simultaneously ask LLM to expand the question
            expand_future = ex.submit(
                llm.invoke,
                _REWRITE_PROMPT.format_messages(
                    history=history_text, question=question,
                ),
            )

            merged: dict[str, tuple] = orig_future.result(timeout=30)

            extra_queries: list[str] = []
            try:
                rewrite_res = expand_future.result(timeout=EXPAND_TIMEOUT)
                extra_queries = _parse_expansion(rewrite_res.content)
            except Exception:
                pass  # expansion failed; original question is enough

            # Fetch for each expanded query in parallel
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

        # ── Stage 2: cosine threshold filter ──────────────────────────────
        # COSINE_THRESHOLD applies only to FAISS scores (0–1 range).
        # If nothing passes, keep the raw top candidates so broad questions
        # ("explain the document", "what are the conclusions") are never
        # silently dropped — let the LLM decide from whatever context exists.
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
        # Keep a cosine-score lookup so results tuple stays consistent
        cosine_lookup  = {doc.page_content: score for doc, score in above}

        # ── Stage 3: CrossEncoder rerank (ordering only) ─────────────────
        # CE logit scores are unbounded (-∞ to +∞).  We use them ONLY to
        # reorder candidates — we never threshold on them or expose them as
        # confidence.  Confidence always comes from cosine scores (0–1).
        try:
            reranker  = _get_reranker()
            pairs     = [[question, doc.page_content] for doc in docs_to_rerank]
            ce_scores = reranker.predict(pairs)

            ranked = sorted(
                zip(ce_scores, docs_to_rerank),
                key=lambda x: x[0],
                reverse=True,
            )[:FINAL_TOP_N]

            final_docs = [doc for _, doc in ranked]

        except Exception:
            # Reranker unavailable — order by cosine score
            final_docs = [
                doc for doc, _ in sorted(above, key=lambda x: x[1], reverse=True)
            ][:FINAL_TOP_N]

        # results: (doc, cosine_score) — always 0–1, always consistent
        results = [
            (doc, cosine_lookup.get(doc.page_content, 0.0))
            for doc in final_docs
        ]

        # ── Stage 4: build context + stream answer ────────────────────────
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
                    if chunk.content:
                        has_yielded = True
                        yield chunk.content
            except Exception as exc:
                yield f"\n\n[Generation error]: {exc}"
                has_yielded = True
            if not has_yielded:
                yield "NOT_FOUND"

        return token_generator(), final_docs, results

    return rag_pipeline


