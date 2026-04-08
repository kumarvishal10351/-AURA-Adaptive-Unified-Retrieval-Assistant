import streamlit as st
from langchain_mistralai import ChatMistralAI
from config.settings import get_api_key


@st.cache_resource
def get_fallback_llm():
    """Cached fallback LLM client. Higher temperature for general knowledge."""
    return ChatMistralAI(
        api_key=get_api_key(),
        model="mistral-large-latest",
        temperature=0.7,
    )