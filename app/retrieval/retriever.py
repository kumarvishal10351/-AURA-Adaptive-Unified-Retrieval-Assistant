import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, CHROMA_DB_DIR


@st.cache_resource
def _get_embeddings():
    """Shared, cached embeddings model. Identical to embedder._get_embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_vectorstore(collection_name: str = "rag_docs"):
    """Cached ChromaDB vectorstore connection. Created once per session."""
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )