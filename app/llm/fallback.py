from functools import lru_cache
from langchain_mistralai import ChatMistralAI
from config.settings import get_api_key, PRIMARY_LLM_MODEL


@lru_cache(maxsize=1)
def get_fallback_llm():
    """Cached fallback LLM client with higher temperature (0.7) for general knowledge synthesis."""
    return ChatMistralAI(
        api_key=get_api_key(),
        model=PRIMARY_LLM_MODEL,  # open-mistral-nemo is universally supported across tiers
        temperature=0.7,
        timeout=30,
        max_retries=2,
    )