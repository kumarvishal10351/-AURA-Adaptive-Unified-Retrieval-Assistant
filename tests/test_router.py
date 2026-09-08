"""
Tests for chains/router.py hybrid relevance gating.
"""

from app.chains.router import is_relevant


class MockLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, prompt):
        class Resp:
            content = self.response_text
        return Resp()


def test_router_score_below_threshold():
    llm = MockLLM("YES")
    # Score 0.15 is below default 0.20 -> rejected immediately without LLM call
    assert is_relevant(llm, "question", "context", score=0.15, threshold=0.20) is False


def test_router_llm_approves():
    llm = MockLLM("YES, this context answers the question.")
    assert is_relevant(llm, "question", "context", score=0.50, threshold=0.20) is True


def test_router_llm_rejects():
    llm = MockLLM("NO, context does not answer.")
    assert is_relevant(llm, "question", "context", score=0.50, threshold=0.20) is False


def test_router_llm_exception_fallback():
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("API unavailable")

    # When LLM fails, falls back to score >= threshold
    assert is_relevant(BrokenLLM(), "q", "ctx", score=0.30, threshold=0.20) is True
    assert is_relevant(BrokenLLM(), "q", "ctx", score=0.10, threshold=0.20) is False
