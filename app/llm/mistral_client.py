import streamlit as st
from langchain_mistralai import ChatMistralAI

def get_mistral_llm():
    llm = ChatMistralAI(
        api_key=st.secrets["MISTRAL_API_KEY"],  # ✅ FIX
        model="mistral-large-latest",
        temperature=0.2
    )
    return llm