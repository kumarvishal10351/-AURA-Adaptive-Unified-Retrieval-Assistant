import os
import streamlit as st
from dotenv import load_dotenv

# Environment configuration for thread-safety and tokenizer performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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


# ── Project directories ───────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAISS_DB_DIR = os.path.join(_PROJECT_ROOT, "faiss_db")
DATA_DOCS_DIR = os.path.join(_PROJECT_ROOT, "data", "docs")

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PRIMARY_LLM_MODEL = "open-mistral-nemo"
FALLBACK_LLM_MODEL = "mistral-large-latest"

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Retrieval parameters ──────────────────────────────────────────────────────
TOP_K = 12
MAX_CONTEXT_LENGTH = 16_000
COSINE_THRESHOLD: float = 0.20
FINAL_TOP_N: int = 5
FETCH_K: int = max(TOP_K * 2, 12)
EXPAND_TIMEOUT: int = 18
HISTORY_TURNS: int = 3

# ── Confidence scoring parameters ─────────────────────────────────────────────
COSINE_FULL_CONFIDENCE: float = 0.80
CONFIDENCE_FLOOR: float = 20.0

# ── MLflow tracking ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-assistant")