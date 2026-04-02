from langchain_mistralai import ChatMistralAI
from app.config.settings import MISTRAL_API_KEY

def get_mistral_llm():
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=0.2
    )
    return llm