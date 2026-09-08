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
    page_title="AURA — Adaptive Unified Retrieval Assistant",
    page_icon="A",
    layout="wide",
)


# ── Stitch Design System: Archival Research Workspace ─────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════════════════════════════════════════
       AURA — Stitch "Archival Research Workspace" Design System
       Fonts: EB Garamond (Headlines) · Manrope (Body) · JetBrains Mono (Code)
       ═══════════════════════════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=JetBrains+Mono:wght@400;500;600&family=Manrope:wght@300;400;500;600;700&display=swap');

    /* ── CSS Custom Properties (Design Tokens) ──────────────────────────────── */
    :root {
        --surface:                 #FAF9FB;
        --surface-dim:             #DBD9DC;
        --surface-container-lowest:#FFFFFF;
        --surface-container-low:   #F5F3F5;
        --surface-container:       #EFEDF0;
        --surface-container-high:  #E9E8EA;
        --surface-container-highest:#E3E2E4;
        --on-surface:              #1B1C1E;
        --on-surface-variant:      #3F4943;
        --outline:                 #6F7973;
        --outline-variant:         #BFC9C1;
        --primary:                 #005239;
        --on-primary:              #FFFFFF;
        --primary-container:       #1F6B4F;
        --on-primary-container:    #9FE9C5;
        --secondary:               #46645A;
        --on-secondary:            #FFFFFF;
        --secondary-container:     #C8EADD;
        --on-secondary-container:  #4C6B60;
        --tertiary:                #773300;
        --tertiary-container:      #9C4500;
        --on-tertiary-container:   #FFCFB6;
        --error:                   #BA1A1A;
        --error-container:         #FFDAD6;
        --on-error-container:      #93000A;
        --surface-tint:            #1E6B4F;

        --font-headline: 'EB Garamond', Georgia, 'Times New Roman', serif;
        --font-body:     'Manrope', -apple-system, 'Segoe UI', sans-serif;
        --font-code:     'JetBrains Mono', 'Fira Code', 'Consolas', monospace;

        --radius-sm:  2px;
        --radius-md:  4px;
        --radius-lg:  6px;
        --radius-xl:  8px;
    }

    /* ── Hide Streamlit Chrome ──────────────────────────────────────────────── */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stStatusWidget"], [data-testid="stDecoration"],
    .stDeployButton, header[data-testid="stHeader"] {
        display: none !important;
    }

    /* ── Global Typography & Canvas ─────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    .stApp, .main, [data-testid="stMainBlockContainer"] {
        background-color: var(--surface) !important;
        color: var(--on-surface) !important;
        font-family: var(--font-body) !important;
        font-size: 0.9375rem !important;
        line-height: 1.5rem !important;
        letter-spacing: 0em !important;
    }

    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    /* ── Headings → EB Garamond ─────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeading"] {
        font-family: var(--font-headline) !important;
        color: var(--on-surface) !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Markdown body ──────────────────────────────────────────────────────── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"] {
        font-family: var(--font-body) !important;
        color: var(--on-surface) !important;
        line-height: 1.6 !important;
    }

    [data-testid="stCaptionContainer"] span,
    [data-testid="stCaptionContainer"] p {
        color: var(--outline) !important;
        font-size: 0.8125rem !important;
    }

    /* ── Code blocks → JetBrains Mono ───────────────────────────────────────── */
    code, pre, [data-testid="stCode"],
    .stCodeBlock, [data-testid="stCodeBlock"] {
        font-family: var(--font-code) !important;
        font-size: 0.8125rem !important;
        line-height: 1.25rem !important;
        background-color: var(--surface-container-low) !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: var(--radius-md) !important;
        color: var(--on-surface-variant) !important;
    }

    /* ── Metrics (Telemetry) ─────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--surface-container-lowest) !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: var(--radius-xl) !important;
        padding: 0.75rem 1rem !important;
        box-shadow: none !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: var(--font-code) !important;
        font-size: 0.6875rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: var(--outline) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: var(--font-headline) !important;
        font-size: 1.75rem !important;
        font-weight: 500 !important;
        color: var(--on-surface) !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 0.8125rem !important;
        letter-spacing: 0.01em !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--outline-variant) !important;
        background: var(--surface-container-lowest) !important;
        color: var(--on-surface) !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
        padding: 0.4rem 1rem !important;
    }

    .stButton > button:hover {
        background: var(--surface-container-low) !important;
        border-color: var(--outline) !important;
        color: var(--primary) !important;
    }

    /* Primary button variant (using Streamlit's primary type) */
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        background: var(--primary) !important;
        color: var(--on-primary) !important;
        border-color: var(--primary) !important;
    }

    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: var(--primary-container) !important;
        border-color: var(--primary-container) !important;
        color: var(--on-primary) !important;
    }

    /* ── Chat Input ──────────────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: var(--radius-xl) !important;
        border: 1px solid var(--outline-variant) !important;
        background: var(--surface-container-lowest) !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea {
        font-family: var(--font-body) !important;
        font-size: 0.9375rem !important;
        color: var(--on-surface) !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--outline) !important;
        font-style: italic !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--primary-container) !important;
        outline: none !important;
    }

    /* Chat send button */
    [data-testid="stChatInput"] button {
        background: var(--primary) !important;
        color: var(--on-primary) !important;
        border-radius: var(--radius-md) !important;
    }

    /* ── Chat Messages ───────────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: var(--surface-container-lowest) !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: var(--radius-xl) !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: none !important;
        margin-bottom: 0.75rem !important;
    }

    /* User messages get a subtle left accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid var(--secondary) !important;
        background: var(--surface-container-low) !important;
    }

    /* Assistant messages get the primary accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid var(--primary-container) !important;
    }

    /* Avatar styling */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        background: var(--primary) !important;
        color: var(--on-primary) !important;
    }

    /* ── Expander (Source Attribution) ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--surface-container-low) !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] summary {
        font-family: var(--font-code) !important;
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        color: var(--on-surface-variant) !important;
        letter-spacing: 0.005em !important;
    }

    /* ── Divider ──────────────────────────────────────────────────────────────── */
    [data-testid="stHorizontalRule"],
    hr {
        border-color: var(--outline-variant) !important;
        opacity: 0.4 !important;
    }

    /* ── File Uploader ───────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        font-family: var(--font-body) !important;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed var(--outline-variant) !important;
        border-radius: var(--radius-md) !important;
        background: var(--surface-container-low) !important;
    }

    /* ── Popover ──────────────────────────────────────────────────────────────── */
    [data-testid="stPopover"] > div {
        border: 1px solid var(--outline-variant) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: 0px 2px 4px rgba(24,25,27,0.04),
                    0px 8px 16px rgba(24,25,27,0.06) !important;
        background: var(--surface-container-lowest) !important;
    }

    /* ── Status / Spinner ────────────────────────────────────────────────────── */
    [data-testid="stStatusWidget"],
    .stSpinner {
        font-family: var(--font-code) !important;
        font-size: 0.8125rem !important;
    }

    /* ── Alerts ───────────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        font-family: var(--font-body) !important;
        border: 1px solid var(--outline-variant) !important;
    }

    /* Info alert styling */
    .stAlert[data-baseweb] {
        background: var(--surface-container-low) !important;
    }

    /* ── Scrollbar ────────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--outline-variant);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--outline);
    }

    /* ── Custom Component Classes ─────────────────────────────────────────────── */

    /* Masthead */
    .aura-masthead {
        background: var(--surface-container-lowest);
        border-bottom: 1px solid var(--outline-variant);
        padding: 1rem 2rem;
        margin: -1rem -1rem 0 -1rem;
    }

    .aura-wordmark {
        font-family: var(--font-headline);
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--on-surface);
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .aura-wordmark-sub {
        font-family: var(--font-code);
        font-size: 0.6875rem;
        color: var(--outline);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* Telemetry Bar */
    .aura-telemetry {
        background: var(--surface-container-low);
        border-bottom: 1px solid var(--outline-variant);
        padding: 0.5rem 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0 -1rem;
    }

    .aura-telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.6rem;
        background: var(--surface-container-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-md);
        font-family: var(--font-code);
        font-size: 0.6875rem;
        font-weight: 500;
        color: var(--on-surface-variant);
        letter-spacing: 0em;
    }

    .aura-telemetry-pill .pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--primary-container);
        animation: pulse-dot 2s infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .aura-telemetry-pill .pill-value {
        font-weight: 600;
        color: var(--on-surface);
    }

    /* Greeting Section */
    .aura-greeting {
        text-align: center;
        padding: 3rem 1rem 1.5rem;
    }

    .aura-greeting-headline {
        font-family: var(--font-headline);
        font-size: 2.25rem;
        font-weight: 500;
        color: var(--on-surface);
        letter-spacing: -0.015em;
        line-height: 2.75rem;
        margin-bottom: 0.5rem;
    }

    .aura-greeting-sub {
        font-family: var(--font-body);
        font-size: 0.9375rem;
        color: var(--on-surface-variant);
        max-width: 40rem;
        margin: 0 auto;
        line-height: 1.5;
    }

    .aura-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        background: transparent;
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-md);
        font-family: var(--font-code);
        font-size: 0.6875rem;
        color: var(--secondary);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .aura-status-badge .badge-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--primary-container);
    }

    /* Confidence Badges */
    .conf-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.15rem 0.5rem;
        border-radius: var(--radius-md);
        font-family: var(--font-code);
        font-size: 0.6875rem;
        font-weight: 600;
        letter-spacing: 0em;
    }

    .conf-badge-high {
        color: var(--primary-container);
        background: rgba(31, 107, 79, 0.08);
        border: 1px solid rgba(31, 107, 79, 0.3);
    }

    .conf-badge-medium {
        color: #B45309;
        background: rgba(180, 83, 9, 0.08);
        border: 1px solid rgba(180, 83, 9, 0.3);
    }

    .conf-badge-low {
        color: #991B1B;
        background: rgba(153, 27, 27, 0.08);
        border: 1px solid rgba(153, 27, 27, 0.3);
    }

    /* Confidence Progress Bar */
    .conf-bar-track {
        width: 100%;
        height: 4px;
        background: var(--surface-container-high);
        border-radius: 2px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s ease;
    }

    .conf-bar-fill-high { background: var(--primary-container); }
    .conf-bar-fill-medium { background: #B45309; }
    .conf-bar-fill-low { background: #991B1B; }

    /* Source Citation Chip */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.1rem 0.4rem;
        background: var(--surface-container-lowest);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-md);
        font-family: var(--font-code);
        font-size: 0.6875rem;
        font-weight: 500;
        color: var(--on-surface-variant);
        margin-right: 0.35rem;
    }

    /* Evidence Snippet */
    .evidence-block {
        border-left: 2px solid var(--primary-container);
        background: rgba(31, 107, 79, 0.04);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        font-family: var(--font-body);
        font-size: 0.8125rem;
        color: var(--on-surface-variant);
        line-height: 1.4;
    }

    .evidence-header {
        font-family: var(--font-code);
        font-size: 0.6875rem;
        font-weight: 600;
        color: var(--primary-container);
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
    }

    /* Footer */
    .aura-footer {
        border-top: 1px solid var(--outline-variant);
        padding: 0.75rem 2rem;
        margin: 2rem -1rem 0 -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        font-family: var(--font-code);
        font-size: 0.6875rem;
        color: var(--outline);
        letter-spacing: 0.01em;
    }

    /* Horizontal rule for separators */
    .aura-rule {
        border: none;
        border-top: 1px solid var(--outline-variant);
        opacity: 0.5;
        margin: 0.5rem 0;
    }

    /* Mode label for non-RAG responses */
    .mode-label {
        font-family: var(--font-code);
        font-size: 0.6875rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        padding: 0.15rem 0.5rem;
        border-radius: var(--radius-md);
    }

    .mode-label-notfound {
        color: #B45309;
        background: rgba(180, 83, 9, 0.08);
        border: 1px solid rgba(180, 83, 9, 0.2);
    }

    .mode-label-fallback {
        color: var(--outline);
        background: var(--surface-container);
        border: 1px solid var(--outline-variant);
    }

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


def confidence_badge_html(value: int) -> str:
    """Return a styled confidence badge with progress bar."""
    level = confidence_level(value)
    if value >= 70:
        badge_class = "conf-badge-high"
        bar_class = "conf-bar-fill-high"
    elif value >= 40:
        badge_class = "conf-badge-medium"
        bar_class = "conf-bar-fill-medium"
    else:
        badge_class = "conf-badge-low"
        bar_class = "conf-bar-fill-low"

    return f"""<div style="margin: 0.5rem 0;">
        <span class="conf-badge {badge_class}">{value}% CONF · {level.upper()}</span>
        <div class="conf-bar-track">
            <div class="conf-bar-fill {bar_class}" style="width: {min(value, 100)}%;"></div>
        </div>
    </div>"""


def confidence_bar(value: int) -> str:
    """Markdown-compatible confidence display for chat history."""
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


def render_masthead():
    """Render the scholarly masthead with wordmark."""
    st.markdown("""
    <div class="aura-masthead">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;">
            <div>
                <div class="aura-wordmark">A U R A</div>
                <div class="aura-wordmark-sub">Adaptive Unified Retrieval Assistant</div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="aura-status-badge">
                    <span class="badge-dot"></span>
                    Research Workspace
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_telemetry():
    """Render the telemetry bar with session metrics."""
    doc_count = st.session_state.total_docs
    query_count = st.session_state.total_queries
    conf_avg = avg_confidence()
    conf_display = f"{conf_avg}%" if st.session_state.conf_scores else "—"
    active_file = st.session_state.last_file or "No document"

    ready_status = "READY" if st.session_state.db_ready else "AWAITING INDEX"

    st.markdown(f"""
    <div class="aura-telemetry">
        <span class="aura-telemetry-pill">
            <span class="pill-dot"></span>
            <span style="color: var(--outline);">{ready_status}</span>
        </span>
        <span class="aura-telemetry-pill">
            <span style="color: var(--outline);">Docs:</span>
            <span class="pill-value">{doc_count}</span>
        </span>
        <span class="aura-telemetry-pill">
            <span style="color: var(--outline);">Queries:</span>
            <span class="pill-value">{query_count}</span>
        </span>
        <span class="aura-telemetry-pill">
            <span style="color: var(--outline);">Avg Conf:</span>
            <span class="pill-value">{conf_display}</span>
        </span>
        <span class="aura-telemetry-pill" style="margin-left: auto;">
            <span style="color: var(--outline);">Target:</span>
            <span class="pill-value" style="max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{html_module.escape(active_file)}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_header():
    render_masthead()
    render_telemetry()

    # Document attachment controls
    col_spacer, col_actions = st.columns([3, 2])

    with col_actions:
        action_left, action_right = st.columns(2)
        with action_left:
            if st.session_state.chat_history:
                if st.button("Clear Session", use_container_width=True, icon=":material/delete:"):
                    st.session_state.chat_history = []
                    st.rerun()

        with action_right:
            with st.popover("Attach Document", icon=":material/attach_file:", use_container_width=True):
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
                        if st.button("Index Document", use_container_width=True, key="proc_btn"):
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

    st.markdown('<hr class="aura-rule">', unsafe_allow_html=True)


def render_conversation():
    # ── Empty State: Scholarly Greeting ────────────────────────────────────
    if not st.session_state.chat_history:
        active_name = st.session_state.last_file or "the indexed document"

        if not st.session_state.db_ready:
            status_line = "Attach a PDF using the button above to begin your research session."
        else:
            status_line = f'Ready to interrogate <strong style="color: var(--primary-container);">{html_module.escape(active_name)}</strong> with continuous evidentiary citation.'

        st.markdown(f"""
        <div class="aura-greeting">
            <div class="aura-status-badge">
                <span class="badge-dot"></span>
                Archival Research Engine
            </div>
            <div class="aura-greeting-headline">{greeting()}. What would you like to explore?</div>
            <div class="aura-greeting-sub">{status_line}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat History ──────────────────────────────────────────────────────
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])

        mode = chat.get("mode", "fallback")
        with st.chat_message("assistant"):
            clean_answer = strip_chunk_references(chat["answer"])
            st.markdown(clean_answer)

            # Confidence and sources
            if mode == "rag" and (chat.get("confidence", 0) > 0 or chat.get("docs")):
                conf_val = chat.get("confidence", 0)

                # Render confidence badge
                if conf_val > 0:
                    st.markdown(confidence_badge_html(conf_val), unsafe_allow_html=True)

                chat_docs = chat.get("docs", [])
                if chat_docs:
                    with st.expander(f"Source Evidence — {len(chat_docs)} chunk{'s' if len(chat_docs) != 1 else ''}", icon=":material/description:"):
                        for idx, doc in enumerate(chat_docs):
                            page = doc.get("page", 1)
                            source_file = doc.get("source", "")
                            snippet = doc.get("content", "")[:420]
                            source_label = source_file if source_file else "Document"

                            st.markdown(f"""
                            <div class="evidence-block">
                                <div class="evidence-header">
                                    <span class="source-chip">§{idx + 1}</span>
                                    {html_module.escape(source_label)} · Page {page}
                                </div>
                                <div>{html_module.escape(snippet)}</div>
                            </div>
                            """, unsafe_allow_html=True)

            if mode == "not_found":
                st.markdown(
                    '<span class="mode-label mode-label-notfound">EVIDENCE NOT IN DOCUMENT</span>',
                    unsafe_allow_html=True
                )
            elif mode == "fallback":
                st.markdown(
                    '<span class="mode-label mode-label-fallback">GENERAL KNOWLEDGE · MISTRAL LARGE</span>',
                    unsafe_allow_html=True
                )

    # ── Fallback Offer ────────────────────────────────────────────────────
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


def render_footer():
    """Render the scholarly footer with system metadata."""
    st.markdown(f"""
    <div class="aura-footer">
        <div>AURA Research Workbench · Epistemic Grounding Engine</div>
        <div style="display: flex; gap: 1.5rem;">
            <span>Cosine Threshold: {COSINE_THRESHOLD}</span>
            <span>© 2025 AURA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
    render_footer()


if __name__ == "__main__":
    main()