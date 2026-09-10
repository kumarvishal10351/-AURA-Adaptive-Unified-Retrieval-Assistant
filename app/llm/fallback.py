from functools import lru_cache
from langchain_mistralai import ChatMistralAI
from config.settings import get_api_key, PRIMARY_LLM_MODEL


@lru_cache(maxsize=1)
def get_fallback_llm():
    """High-performance cached fallback LLM client with strict token ceiling for low-latency synthesis."""
    return ChatMistralAI(
        api_key=get_api_key(),
        model=PRIMARY_LLM_MODEL,  # open-mistral-nemo (fast, low-latency 12B model)
        temperature=0.3,
        max_tokens=550,
        timeout=15,
        max_retries=1,
    )