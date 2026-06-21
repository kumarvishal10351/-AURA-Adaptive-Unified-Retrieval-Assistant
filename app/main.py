import streamlit as st
import os
import re
import time
import html as html_module
from datetime import datetime

# Fix tokenizer deadlock on Streamlit hot-reloads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chains.rag_chain import create_rag_chain
from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from ingestion.embedder import store_embeddings
from retrieval.retriever import get_vectorstore
from llm.mistral_client import get_mistral_llm
from llm.fallback import get_fallback_llm
from utils.confidence import calculate_confidence


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
    defaults = {
        "db_ready":      False,
        "last_file":     None,
        "chat_history":  [],
        "total_queries": 0,
        "total_docs":    0,
        "conf_scores":   [],
        "input_key":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def avg_confidence() -> int:
    s = st.session_state.conf_scores
    return int(sum(s) / len(s)) if s else 0


def strip_chunk_references(text: str) -> str:
    """Remove [Chunk X], [Chunks X, Y], and stray 'and [Chunk X]' from answer text."""
    text = re.sub(r"\[Chunks?\s*[\d,\s]+\]", "", text)
    text = re.sub(r"\s+and\s+(?=\s|\.|,|$)", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def confidence_bar(value: int) -> str:
    filled = round(value / 10)
    empty = 10 - filled
    bar = "\u2588" * filled + "\u2591" * empty
    return f"`[{bar}] {value}%`"


def greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


def render_header():
    left, right = st.columns([3, 1])

    with left:
        st.markdown("#### AURA")

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

                os.makedirs("data/docs", exist_ok=True)
                file_path = os.path.join("data/docs", uploaded_file.name)
                if is_new or not os.path.exists(file_path):
                    with open(file_path, "wb") as fh:
                        fh.write(uploaded_file.getbuffer())

                size_str = f"{round(uploaded_file.size / 1024, 1)} KB"
                st.caption(f"{uploaded_file.name} — {size_str}")

                if not st.session_state.db_ready:
                    if st.button("Load documents", use_container_width=True, key="proc_btn"):
                        with st.status("Processing...", expanded=True) as status:
                            try:
                                st.write("Parsing PDF...")
                                docs_raw = load_pdf(file_path)
                                st.write("Splitting into chunks...")
                                chunks = split_documents(docs_raw)
                                st.write("Generating embeddings...")
                                store_embeddings(chunks)
                                st.write("Indexing complete.")
                                time.sleep(0.3)
                                status.update(label="Document ready", state="complete")
                                st.session_state.db_ready = True
                                st.session_state.total_docs += 1
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as exc:
                                status.update(label="Processing failed", state="error")
                                st.error(f"Error: {exc}")

                if st.session_state.db_ready:
                    st.success("Ready to query.", icon=":material/check_circle:")

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
                            page = doc.get("page", "?")
                            snippet = doc.get("content", "")[:420]
                            st.caption(f"Source {idx + 1} — Page {page}")
                            st.code(snippet, language=None)

            if mode in ("not_found", "fallback"):
                label = "not in document" if mode == "not_found" else "general knowledge"
                st.caption(label)

    # Fallback offer
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last.get("mode") == "not_found" and last.get("allow_fallback"):
            col_info, col_btn = st.columns([3, 1])
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
        with st.spinner("Retrieving..."):
            try:
                vectorstore = get_vectorstore()
                llm = get_mistral_llm()
                rag_chain = create_rag_chain(llm, vectorstore)
                answer_gen, docs, results = rag_chain(query_to_run, st.session_state.chat_history)
                confidence = calculate_confidence(results)
                answer = "".join(list(answer_gen))

                if answer.strip().startswith("NOT_FOUND"):
                    mode = "not_found"
                    docs = []
                    confidence = 0
                    answer = "The required context is not present in the uploaded document."
                else:
                    mode = "rag"

            except TimeoutError as exc:
                st.error(f"**Request timeout**: {str(exc)}")
                answer = ""
            except Exception as exc:
                st.error(f"Error: {str(exc)}")
                answer = ""

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
                    "page":    d.metadata.get("page", "?") if hasattr(d, "metadata") else "?",
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