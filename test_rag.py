import os
import sys
import logging

# Fix PyTorch / FAISS deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath("app"))

from config.settings import FAISS_DB_DIR
from retrieval.retriever import get_vectorstore
from llm.mistral_client import get_mistral_llm
from chains.rag_chain import create_rag_chain

print("Loading Vectorstore...")
vectorstore = None
try:
    vectorstore = get_vectorstore()
    print("Vectorstore loaded:", vectorstore)
except Exception as e:
    print("Failed to load vectorstore:", repr(e))

print("Loading LLM...")
try:
    llm = get_mistral_llm()
    print("LLM loaded:", llm)
except Exception as e:
    print("Failed to load LLM:", repr(e))

print("Invoking RAG chain...")
try:
    rag_chain = create_rag_chain(llm, vectorstore)
    answer, docs, results = rag_chain("What are the technical skills?", [])
    print("Answer:", answer)
except Exception as e:
    print("RAG chain failed:", repr(e))
