from functools import lru_cache
from langchain_mistralai import ChatMistralAI
from config.settings import get_api_key


@lru_cache(maxsize=1)
def get_fallback_llm():
    """Cached fallback LLM client. Higher temperature for general knowledge."""
    return ChatMistralAI(
        api_key=get_api_key(),
        model="mistral-large-latest",
        temperature=0.7,
        timeout=30,
        max_retries=2,
    )