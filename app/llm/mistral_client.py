import streamlit as st
from langchain_mistralai import ChatMistralAI
from config.settings import get_api_key


@st.cache_resource
def get_mistral_llm():
    """Cached Mistral LLM client. Created once per session."""
    return ChatMistralAI(
        api_key=get_api_key(),
        model="open-mistral-nemo",
        temperature=0.1,
        timeout=30,  # 30 second timeout for API calls
        max_retries=2,  # Retry up to 2 times on failure
    )