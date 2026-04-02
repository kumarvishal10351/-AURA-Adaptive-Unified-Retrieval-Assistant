from langchain_mistralai import ChatMistralAI
from config.settings import MISTRAL_API_KEY

def get_fallback_llm():
    return ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=0.7
    )