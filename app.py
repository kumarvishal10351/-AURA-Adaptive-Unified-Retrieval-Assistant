"""
app.py
──────
Root entrypoint for Hugging Face Spaces (Gradio SDK).
Provides a web interface for PDF upload, FAISS vector indexing,
and interactive grounded RAG chat powered by Mistral AI.
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure project root and app directory are in sys.path
_ROOT = Path(__file__).resolve().parent
_APP_DIR = _ROOT / "app"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import gradio as gr
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator

from config.settings import DATA_DOCS_DIR, FAISS_DB_DIR
from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from ingestion.embedder import store_embeddings
from retrieval.retriever import get_vectorstore
from llm.mistral_client import get_mistral_llm
from chains.rag_chain import create_rag_chain
from utils.confidence import calculate_confidence


@spaces.GPU
def upload_pdf_file(files):
    if not files:
        return "⚠️ No file selected. Please choose a PDF file to upload.", get_indexed_docs_summary()

    os.makedirs(DATA_DOCS_DIR, exist_ok=True)
    results = []

    file_list = files if isinstance(files, list) else [files]

    for f in file_list:
        file_path = f.name if hasattr(f, "name") else str(f)
        filename = os.path.basename(file_path)
        dest_path = os.path.join(DATA_DOCS_DIR, filename)

        try:
            shutil.copyfile(file_path, dest_path)
            docs = load_pdf(dest_path)
            chunks = split_documents(docs)
            store_embeddings(chunks)
            results.append(f"✅ **{filename}**: Indexed {len(chunks)} chunks across {len(docs)} pages.")
        except Exception as exc:
            results.append(f"❌ **{filename}**: Ingestion failed ({exc})")

    return "\n\n".join(results), get_indexed_docs_summary()


def clear_library():
    """Clear all uploaded documents and reset the FAISS vector index."""
    if os.path.exists(DATA_DOCS_DIR):
        for f in os.listdir(DATA_DOCS_DIR):
            if f.lower().endswith(".pdf"):
                try:
                    os.remove(os.path.join(DATA_DOCS_DIR, f))
                except Exception:
                    pass
    if os.path.exists(FAISS_DB_DIR):
        for f in os.listdir(FAISS_DB_DIR):
            try:
                os.remove(os.path.join(FAISS_DB_DIR, f))
            except Exception:
                pass
    try:
        get_vectorstore.clear()
    except Exception:
        pass
    return "🗑️ Library cleared. Upload a new PDF to begin.", get_indexed_docs_summary()


def get_indexed_docs_summary():
    if not os.path.exists(DATA_DOCS_DIR):
        return "No documents currently indexed."
    pdfs = [f for f in os.listdir(DATA_DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        return "No documents currently indexed."
    return f"📚 **Indexed Documents ({len(pdfs)}):**\n" + "\n".join(f"• {name}" for name in pdfs)


def format_history(history):
    formatted = []
    if not history:
        return formatted
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                formatted.append({"question": content, "answer": ""})
            elif role == "assistant" and formatted:
                formatted[-1]["answer"] = content
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            formatted.append({"question": item[0] or "", "answer": item[1] or ""})
    return formatted


@spaces.GPU
def rag_chat_response(message, history):
    if not message or not message.strip():
        return

    faiss_file = os.path.join(FAISS_DB_DIR, "index.faiss")
    doc_count = 0
    if os.path.exists(DATA_DOCS_DIR):
        doc_count = len([f for f in os.listdir(DATA_DOCS_DIR) if f.lower().endswith(".pdf")])

    if doc_count == 0 or not os.path.exists(faiss_file):
        yield "⚠️ **No documents are indexed yet.** Please upload a PDF in the 'Document Library' section above to start asking questions!"
        return

    try:
        llm = get_mistral_llm()
    except Exception as exc:
        yield f"⚠️ **Mistral LLM Configuration Error**: {exc}. Please verify your MISTRAL_API_KEY secret in Space settings."
        return

    formatted_history = format_history(history)

    try:
        rag_pipeline = create_rag_chain(llm)
        token_gen, final_docs, ranked_results = rag_pipeline(message, formatted_history)

        accumulated_text = ""
        for token in token_gen:
            accumulated_text += token
            yield accumulated_text

        # Format sources and confidence
        if ranked_results:
            scores = [sc for _, sc in ranked_results]
            conf = calculate_confidence(scores)

            source_lines = []
            for i, (doc, sc) in enumerate(ranked_results):
                meta = getattr(doc, "metadata", {}) or {}
                page = meta.get("page", 0) + 1
                fname = meta.get("file_name", "Document")
                source_lines.append(f"- **{fname}** (Page {page}, Match Score: `{sc:.2f}`)")

            footer = f"\n\n---\n📊 **Grounding Confidence**: `{conf}%`\n"
            if source_lines:
                footer += "🔍 **Sources Retrieved**:\n" + "\n".join(source_lines[:4])

            accumulated_text += footer
            yield accumulated_text

    except Exception as exc:
        yield f"❌ Encountered an error generating response: {exc}"


# ── Gradio Blocks Layout ─────────────────────────────────────────
custom_css = """
#container { max-width: 900px; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
"""

try:
    demo = gr.Blocks(title="Viora Research Workspace", css=custom_css)
except Exception:
    demo = gr.Blocks(title="Viora Research Workspace")

with demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """
            # 🧠 Viora Research Workspace
            ### Grounded RAG Assistant powered by Mistral AI & FAISS
            Upload your documents, explore grounded synthesis, and ask questions with strict zero-speculation citations.
            """,
            elem_classes=["header-box"]
        )

        with gr.Accordion("📂 Document Library (Upload & Manage PDFs)", open=True):
            with gr.Row():
                pdf_files = gr.File(
                    label="Select PDF file(s)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    scale=3
                )
                with gr.Column(scale=1):
                    upload_btn = gr.Button("🚀 Index Document", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Library", variant="secondary")

            upload_status = gr.Markdown("Ready for document upload.")
            docs_summary = gr.Markdown(value=get_indexed_docs_summary())

            upload_btn.click(
                upload_pdf_file,
                inputs=[pdf_files],
                outputs=[upload_status, docs_summary]
            )
            clear_btn.click(
                clear_library,
                inputs=[],
                outputs=[upload_status, docs_summary]
            )

        gr.Markdown("---")
        gr.Markdown("### 💬 Research Consultation")
        try:
            gr.ChatInterface(
                fn=rag_chat_response,
                examples=[
                    "Summarize the key findings in this document.",
                    "What are the technical qualifications mentioned?",
                    "What are the main risks or limitations discussed?",
                ],
                cache_examples=False,
            )
        except TypeError:
            gr.ChatInterface(
                fn=rag_chat_response,
                type="messages",
                examples=[
                    "Summarize the key findings in this document.",
                    "What are the technical qualifications mentioned?",
                    "What are the main risks or limitations discussed?",
                ],
                cache_examples=False,
            )

# ── Mount FastAPI REST API Routes with Gradio UI & CORS ──────────────
from gradio.routes import App
from fastapi.middleware.cors import CORSMiddleware
from app.api import app as fastapi_app

app = App.create_app(demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
initial_count = len(app.router.routes)
app.include_router(fastapi_app.router)
new_routes = app.router.routes[initial_count:]
app.router.routes = new_routes + app.router.routes[:initial_count]

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, _app=app)
