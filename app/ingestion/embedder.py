import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config.settings import EMBEDDING_MODEL, CHROMA_DB_DIR


@st.cache_resource
def _get_embeddings():
    """Shared, cached embeddings model. Downloaded once per session."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def store_embeddings(chunks, collection_name: str = "rag_docs"):
    """
    Embed chunks and persist to ChromaDB.

    Uses a fixed collection name so re-indexing a new document replaces the
    previous collection cleanly.  The collection is deleted first to avoid
    stale data accumulating from previous uploads.
    """
    embeddings = _get_embeddings()

    # Delete existing collection so we start fresh for the new document
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )
    return vectorstore