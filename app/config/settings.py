import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Unified API key resolver: tries st.secrets first, falls back to .env."""
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        key = os.getenv("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY not found. "
                "Set it in .streamlit/secrets.toml or .env"
            )
        return key


# ── Vector store ──────────────────────────────────────────────────────────────
# Use absolute path resolved from this file's location so ChromaDB is always
# created in the project root, regardless of which directory Streamlit runs from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DB_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
# Mistral Small supports 32k tokens; 12 000 chars ≈ 3 000 tokens — much more
# useful than the previous 3 000-char limit while still leaving room for the
# prompt and the model's reply.
MAX_CONTEXT_LENGTH = 12_000

# ── Confidence scoring ────────────────────────────────────────────────────────
# ChromaDB L2 distances typically range 0-2; divisor normalises to 0-100 %.
CONFIDENCE_DIVISOR = 2.0