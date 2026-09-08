import streamlit as st
import os
import re
import time
import html as html_module
from datetime import datetime

# Initialize environment and paths
from config.settings import (
    FAISS_DB_DIR,
    DATA_DOCS_DIR,
    COSINE_THRESHOLD,
)

from chains.rag_chain import create_rag_chain
from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from ingestion.embedder import store_embeddings
from retrieval.retriever import get_vectorstore
from llm.mistral_client import get_mistral_llm
from llm.fallback import get_fallback_llm
from utils.confidence import calculate_confidence, confidence_level

st.set_page_config(
    page_title="AURA",
    page_icon="A",
    layout="centered",
)


def inject_css():
    st.markdown("""
    <style>
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stStatusWidget"], [data-testid="stDecoration"],
    .stDeployButton {display: none !important;}
    .main .block-container {padding-top: 1.5rem !important;}
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    faiss_file = os.path.join(FAISS_DB_DIR, "index.faiss")
    pkl_file = os.path.join(FAISS_DB_DIR, "index.pkl")
    index_persisted = os.path.exists(faiss_file) and os.path.exists(pkl_file)

    existing_docs = 0
    last_file_name = None
    if os.path.exists(DATA_DOCS_DIR):
        doc_files = [f for f in os.listdir(DATA_DOCS_DIR) if f.lower().endswith(".pdf")]
        existing_docs = len(doc_files)
        if doc_files:
            last_file_name = doc_files[-1]

    defaults = {
        "db_ready":      index_persisted,
        "last_file":     last_file_name,
        "chat_history":  [],
        "total_queries": 0,
        "total_docs":    max(existing_docs, 1 if index_persisted else 0),
        "conf_scores":   [],
        "input_key":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def avg_confidence() -> int:
    s = st.session_state.conf_scores
    valid = [x for x in s if isinstance(x, (int, float)) and x > 0]
    return int(sum(valid) / len(valid)) if valid else 0


def strip_chunk_references(text: str) -> str:
    """Remove [Chunk X], [Chunks X, Y], and stray references from answer text."""
    text = re.sub(r"\[Chunks?\s*[\d,\s]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+and\s+(?=\s|\.|,|$)", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def confidence_bar(value: int) -> str:
    filled = round(value / 10)
    empty = 10 - filled
    bar = "\u2588" * filled + "\u2591" * empty
    level = confidence_level(value)
    return f"`[{bar}] {value}%` ({level})"


def greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


def render_header():
    left, middle, right = st.columns([2.5, 1.2, 1.3])

    with left:
        st.markdown("#### AURA")

    with middle:
        if st.session_state.chat_history:
            if st.button("Clear chat", use_container_width=True, icon=":material/delete:"):
                st.session_state.chat_history = []
                st.rerun()

    with right:
        with st.popover("Attach document", icon=":material/attach_file:", use_container_width=True):
            uploaded_file = st.file_uploader(
                "PDF file", type=["pdf"], label_visibility="collapsed"
            )

            if uploaded_file:
                is_new = st.session_state.last_file != uploaded_file.name
                if is_new:
                    st.session_state.db_ready = False
                    st.session_state.last_file = uploaded_file.name
                    get_vectorstore.clear()

                os.makedirs(DATA_DOCS_DIR, exist_ok=True)
                file_path = os.path.join(DATA_DOCS_DIR, uploaded_file.name)
                if is_new or not os.path.exists(file_path):
                    with open(file_path, "wb") as fh:
                        fh.write(uploaded_file.getbuffer())

                size_str = f"{round(uploaded_file.size / 1024, 1)} KB"
                st.caption(f"{uploaded_file.name} — {size_str}")

                if not st.session_state.db_ready:
                    if st.button("Load document", use_container_width=True, key="proc_btn"):
                        with st.status("Processing document...", expanded=True) as status:
                            from utils import mlflow_logger
                            mlflow_logger.start_experiment()
                            try:
                                st.write("Parsing PDF with PyMuPDF...")
                                docs_raw = load_pdf(file_path)
                                st.write("Splitting into semantic chunks...")
                                chunks = split_documents(docs_raw)
                                st.write("Generating normalized dense embeddings...")
                                store_embeddings(chunks)
                                st.write("Indexing into FAISS complete.")
                                time.sleep(0.3)
                                status.update(label="Document ready", state="complete")
                                st.session_state.db_ready = True
                                st.session_state.total_docs += 1
                                # Clear previous chat history on new document indexing
                                if is_new:
                                    st.session_state.chat_history = []
                                time.sleep(0.4)
                                st.rerun()
                            except Exception as exc:
                                status.update(label="Processing failed", state="error")
                                st.error(f"Error: {exc}")
                            finally:
                                mlflow_logger.end_run()

            if st.session_state.db_ready:
                active_name = st.session_state.last_file or "Persisted Document"
                st.success(f"Active: {active_name}", icon=":material/check_circle:")

    if st.session_state.db_ready:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Documents", st.session_state.total_docs)
        with c2:
            st.metric("Queries", st.session_state.total_queries)
        with c3:
            conf_val = f"{avg_confidence()}%" if st.session_state.conf_scores else "—"
            st.metric("Avg confidence", conf_val)

    st.divider()


def render_conversation():
    # Empty state
    if not st.session_state.chat_history:
        st.markdown("")
        st.markdown("")
        st.markdown(f"##### {greeting()}. What would you like to know?")
        if not st.session_state.db_ready:
            st.caption("Attach a PDF using the button above to get started.")
        else:
            active_name = st.session_state.last_file or "the indexed document"
            st.caption(f"Ready to answer questions from **{active_name}**.")
        st.markdown("")
        st.markdown("")

    # Chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])

        mode = chat.get("mode", "fallback")
        with st.chat_message("assistant"):
            clean_answer = strip_chunk_references(chat["answer"])
            st.markdown(clean_answer)

            # Confidence and sources in a single expander
            if mode == "rag" and (chat.get("confidence", 0) > 0 or chat.get("docs")):
                with st.expander("Confidence & sources", icon=":material/info:"):
                    if chat.get("confidence", 0) > 0:
                        st.markdown(f"Confidence: {confidence_bar(chat['confidence'])}")

                    chat_docs = chat.get("docs", [])
                    if chat_docs:
                        st.caption(f"{len(chat_docs)} source chunk{'s' if len(chat_docs) != 1 else ''}")
                        for idx, doc in enumerate(chat_docs):
                            page = doc.get("page", 1)
                            source_file = doc.get("source", "")
                            snippet = doc.get("content", "")[:420]
                            source_label = f"Source {idx + 1} — {source_file} (Page {page})" if source_file else f"Source {idx + 1} — Page {page}"
                            st.caption(source_label)
                            st.code(snippet, language=None)

            if mode in ("not_found", "fallback"):
                label = "Evidence not in document" if mode == "not_found" else "General knowledge (Mistral Large fallback)"
                st.caption(label)

    # Fallback offer
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last.get("mode") == "not_found" and last.get("allow_fallback"):
            col_info, col_btn = st.columns([3, 1.2])
            with col_info:
                st.info("The required context was not found in the uploaded document.")
            with col_btn:
                if st.button(
                    "Use general model",
                    use_container_width=True,
                    key="fallback_trigger_btn",
                    icon=":material/language:",
                ):
                    with st.spinner("Querying general model..."):
                        try:
                            fallback_llm = get_fallback_llm()
                            fb_answer = fallback_llm.invoke(last["question"]).content
                            last["answer"] = fb_answer
                            last["mode"] = "fallback"
                            last["allow_fallback"] = False
                            last["confidence"] = 0
                            last["docs"] = []
                            st.session_state.chat_history[-1] = last
                        except Exception as exc:
                            st.error(f"Fallback failed: {exc}")
                    st.rerun()


def handle_input():
    placeholder = "Ask about your document..." if st.session_state.db_ready else "Attach a document first..."

    query_input = st.chat_input(
        placeholder=placeholder,
        disabled=not st.session_state.db_ready,
        key=f"query_input_{st.session_state.input_key}",
    )

    if not query_input or not query_input.strip():
        return

    query_to_run = query_input.strip()

    if not st.session_state.db_ready:
        st.warning("Attach and process a document first.")
        return

    answer, docs, mode, confidence = "", [], "fallback", 0

    with st.chat_message("user"):
        st.markdown(query_to_run)

    with st.chat_message("assistant"):
        from utils import mlflow_logger
        mlflow_logger.start_experiment()
        try:
            vectorstore = get_vectorstore()
            llm = get_mistral_llm()
            rag_chain = create_rag_chain(llm, vectorstore)

            with st.spinner("Retrieving relevant context..."):
                answer_gen, docs, results = rag_chain(query_to_run, st.session_state.chat_history)
                confidence = calculate_confidence(results)

            # Peek first chunk to check for NOT_FOUND sentinel
            first_chunk = next(answer_gen, None)
            first_str = str(first_chunk or "").strip()

            if not first_str or first_str.upper().startswith("NOT_FOUND") or first_str.upper().startswith("NOT FOUND"):
                mode = "not_found"
                docs = []
                confidence = 0
                answer = "The required context is not present in the uploaded document."
                st.markdown(answer)
            else:
                mode = "rag"
                def token_stream():
                    yield first_str
                    for chunk in answer_gen:
                        if chunk:
                            yield str(chunk)

                raw_streamed = st.write_stream(token_stream())
                answer = strip_chunk_references(raw_streamed or "")

        except TimeoutError as exc:
            st.error(f"**Request timeout**: {str(exc)}")
            answer = ""
        except Exception as exc:
            st.error(f"Error: {str(exc)}")
            answer = ""
        finally:
            mlflow_logger.end_run()

    if answer:
        st.session_state.total_queries += 1
        if mode == "rag" and confidence > 0:
            st.session_state.conf_scores.append(confidence)

        st.session_state.chat_history.append({
            "question":       query_to_run,
            "answer":         answer,
            "mode":           mode,
            "confidence":     confidence,
            "allow_fallback": mode == "not_found",
            "docs": [
                {
                    "content": d.page_content,
                    "page":    (d.metadata.get("page", 0) + 1) if isinstance(d.metadata.get("page"), int) else d.metadata.get("page", 1),
                    "source":  d.metadata.get("file_name", os.path.basename(d.metadata.get("source", ""))),
                }
                for d in docs
            ],
        })
        st.session_state.input_key += 1
        st.rerun()


def main():
    inject_css()
    init_session_state()
    render_header()
    render_conversation()
    handle_input()


if __name__ == "__main__":
    main()