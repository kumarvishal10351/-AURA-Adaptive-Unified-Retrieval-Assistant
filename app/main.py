import streamlit as st
import os
import re
import time
import html as html_module

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


# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG Assistant — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #060b16;
        --surface: #0d1528;
        --surface-2: #111b31;
        --surface-3: #17223c;
        --primary: #38bdf8;
        --primary-2: #2563eb;
        --primary-3: #22d3ee;
        --primary-dim: rgba(56,189,248,0.14);
        --success: #34d399;
        --warning: #f59e0b;
        --danger: #f43f5e;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        --border: rgba(148,163,184,0.16);
        --border-hover: rgba(148,163,184,0.28);
        --shadow-sm: 0 10px 22px rgba(0,0,0,0.35);
        --shadow-md: 0 16px 34px rgba(0,0,0,0.45);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-full: 9999px;
    }

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text) !important;
    }
    body { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
    #MainMenu, footer { visibility: hidden; }
    header {
        background: transparent !important;
        border-bottom: none !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    /* Hide Streamlit toolbar items like "Deploy" without removing sidebar toggle */
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 0.9rem !important;
        left: 0.85rem !important;
        z-index: 1000 !important;
        background: rgba(10,18,35,0.85) !important;
        border: 1px solid rgba(148,163,184,0.16) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        color: white !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: white !important;
        color: white !important;
    }

    [data-testid="stSidebarCollapseButton"] {
        color: white !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
        color: white !important;
    }
    .stDeployButton { display: none; }

    html, body { background: var(--bg) !important; }
    .stApp {
        background:
            radial-gradient(900px 500px at 10% -10%, rgba(37,99,235,0.24) 0%, transparent 60%),
            radial-gradient(900px 520px at 95% 8%, rgba(56,189,248,0.16) 0%, transparent 62%),
            linear-gradient(180deg, #070d1b 0%, #060b16 100%) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(109,40,217,0.45), rgba(37,99,235,0.35));
        border-radius: 99px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1223 0%, #080e1e 100%) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 320px !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1.15rem !important; }

    /* ── Main container ── */
    .main .block-container {
        padding: 1.35rem 1.75rem 2.5rem !important;
        max-width: 1260px !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4 { color: var(--text) !important; font-weight: 800 !important; letter-spacing: -0.025em; }

    /* ── App title ── */
    .app-title {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 55%, var(--primary-3) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        letter-spacing: -0.04em;
    }

    /* ── Logo ── */
    .logo-box {
        width: 44px; height: 44px; border-radius: 13px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 60%, var(--primary-3) 100%);
        display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
        box-shadow: 0 10px 30px rgba(109,40,217,0.18);
        flex-shrink: 0;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 6px 14px; border-radius: var(--radius-full);
        background: rgba(109,40,217,0.06); border: 1px solid rgba(109,40,217,0.16);
        font-size: 0.68rem; font-weight: 700;
    }
    .status-dot {
        width: 6px; height: 6px; border-radius: 50%;
        animation: blink 2.5s ease-in-out infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem 1.2rem 1.2rem !important;
        transition: all 0.25s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(15,23,42,0.08) !important;
        border-color: var(--border-hover) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important; font-size: 0.6rem !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 800 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important; font-size: 1.55rem !important;
        font-weight: 900 !important; letter-spacing: -0.04em !important;
    }

    /* ── Panel ── */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin-bottom: 0.85rem;
        box-shadow: var(--shadow-sm);
    }
    .panel-header {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid var(--border);
        display: flex; align-items: center; gap: 9px;
        background: rgba(148,163,184,0.05);
    }
    .panel-body { padding: 1rem; }
    .p-icon {
        width: 28px; height: 28px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center; font-size: 0.8rem;
    }
    .p-icon-teal   { background: rgba(37,99,235,0.10); }
    .p-icon-gold   { background: rgba(217,119,6,0.10); }
    .p-icon-violet { background: rgba(109,40,217,0.10); }
    .p-title { font-size: 0.78rem !important; font-weight: 800 !important; color: var(--text) !important; }
    .p-sub   { font-size: 0.59rem; color: var(--text-muted); margin-top: 1px; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: radial-gradient(ellipse at 50% 80%, rgba(56,189,248,0.07) 0%, transparent 72%) !important;
        border: 2px dashed rgba(56,189,248,0.30) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.2rem !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(109,40,217,0.40) !important;
        box-shadow: 0 0 28px rgba(109,40,217,0.08) !important;
    }
    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 65%, #1d4ed8 100%) !important;
        color: #fff !important;
        border: 1px solid rgba(109,40,217,0.22) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 800 !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 12px 26px rgba(109,40,217,0.18) !important;
    }
    /* Streamlit adds an "upload another" icon button after upload; hide it (looks like a random + button) */
    [data-testid="stFileUploader"] button[aria-label="Upload another file"] { display: none !important; }
    [data-testid="stFileUploader"] button[title="Upload another file"] { display: none !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 60%, var(--primary-3) 100%) !important;
        color: white !important;
        border: 1px solid rgba(109,40,217,0.20) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 10px 26px rgba(109,40,217,0.16), inset 0 1px 0 rgba(255,255,255,0.18) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 14px 34px rgba(109,40,217,0.22), 0 0 0 1px rgba(109,40,217,0.20) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(148,163,184,0.08) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        color: var(--text) !important; box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(109,40,217,0.06) !important;
        border-color: rgba(109,40,217,0.18) !important;
        color: var(--primary) !important; transform: none !important;
    }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background: rgba(15,23,42,0.75) !important;
        border: 1px solid rgba(148,163,184,0.24) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.72rem 1rem !important;
        caret-color: var(--primary) !important;
        transition: all 0.22s ease !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus {
        background: rgba(15,23,42,0.92) !important;
        border-color: rgba(56,189,248,0.45) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(100,116,139,0.55) !important; }
    label[data-testid="stWidgetLabel"] {
        color: var(--text-muted) !important; font-size: 0.65rem !important;
        font-weight: 800 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
    }

    /* ── Alerts ── */
    .stSuccess {
        background: rgba(22,163,74,0.08) !important;
        border: 1px solid rgba(22,163,74,0.18) !important; border-radius: var(--radius-md) !important;
    }
    .stWarning {
        background: rgba(217,119,6,0.08) !important;
        border: 1px solid rgba(217,119,6,0.18) !important; border-radius: var(--radius-md) !important;
    }
    .stError {
        background: rgba(220,38,38,0.08) !important;
        border: 1px solid rgba(220,38,38,0.16) !important; border-radius: var(--radius-md) !important;
    }

    /* ── Progress ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--primary-2), var(--primary-3)) !important;
        background-size: 200% 100% !important;
        animation: shimmer 1.8s linear infinite !important;
        border-radius: 99px !important;
    }
    .stProgress > div { border-radius: 99px !important; background: rgba(15,23,42,0.06) !important; height: 4px !important; }
    @keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(109,40,217,0.05) !important;
        border: 1px solid rgba(109,40,217,0.14) !important;
        border-radius: var(--radius-md) !important;
        color: var(--primary) !important; font-weight: 700 !important; font-size: 0.78rem !important;
        transition: all 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover { background: rgba(109,40,217,0.08) !important; }
    .streamlit-expanderContent {
        background: rgba(15,23,42,0.45) !important;
        border: 1px solid rgba(148,163,184,0.15) !important;
        border-top: none !important; border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }

    /* ── Sidebar helpers ── */
    .sidebar-section {
        font-size: 0.65rem; font-weight: 900; color: var(--primary);
        text-transform: uppercase; letter-spacing: 0.12em;
        padding: 14px 4px 10px; margin-top: 6px;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(56,189,248,0.18) 50%, transparent 100%);
        margin: 14px 0;
    }
    .sidebar-status-item {
        display: flex; align-items: center; gap: 13px;
        padding: 14px 16px; border-radius: var(--radius-lg);
        border: 1px solid rgba(148,163,184,0.14);
        background: rgba(15,23,42,0.50);
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    .sidebar-status-item:hover {
        background: rgba(15,23,42,0.65);
        border-color: rgba(148,163,184,0.22);
    }
    .sidebar-status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .sidebar-status-dot-on  { background: var(--success); box-shadow: 0 0 0 4px rgba(22,163,74,0.16), 0 0 12px rgba(22,163,74,0.25); }
    .sidebar-status-dot-off { background: var(--warning); box-shadow: 0 0 0 4px rgba(217,119,6,0.16), 0 0 12px rgba(217,119,6,0.20); }
    .sidebar-card {
        background: rgba(13, 21, 40, 0.55);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: var(--radius-xl);
        padding: 1.15rem 1.1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }
    .sidebar-mini { font-size: 0.72rem; color: var(--text-muted); line-height: 1.65; }
    .sidebar-kpi {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-top: 14px;
    }
    .sidebar-kpi-item {
        background: rgba(148,163,184,0.06);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 14px;
        padding: 12px 12px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .sidebar-kpi-item:hover {
        background: rgba(148,163,184,0.10);
        border-color: rgba(56,189,248,0.20);
        transform: translateY(-1px);
    }
    .sidebar-kpi-label { font-size: 0.58rem; color: #64748b; font-weight: 900; letter-spacing: 0.10em; text-transform: uppercase; }
    .sidebar-kpi-val { font-size: 0.95rem; color: #e2e8f0; font-weight: 900; letter-spacing: -0.02em; margin-top: 3px; }

    /* Expander tweaks (sidebar) */
    .streamlit-expanderHeader { padding: 0.6rem 0.85rem !important; }
    .streamlit-expanderContent { padding: 0.75rem 0.85rem !important; }

    /* Sidebar suggested query buttons */
    .sidebar-query-btn {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: var(--radius-md);
        background: rgba(15,23,42,0.40); border: 1px solid rgba(148,163,184,0.12);
        font-size: 0.78rem; color: var(--text); cursor: pointer;
        transition: all 0.2s ease; width: 100%; text-align: left;
        margin-bottom: 6px;
    }
    .sidebar-query-btn:hover {
        background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.22);
        color: var(--primary); transform: translateX(3px);
    }

    /* Sidebar file info */
    .sidebar-file-info {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 14px; border-radius: var(--radius-md);
        background: rgba(37,99,235,0.06);
        border: 1px solid rgba(37,99,235,0.14);
        margin: 10px 0;
    }
    .sidebar-file-icon {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(14,165,233,0.14));
        display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
        flex-shrink: 0;
    }

    /* ─────────────────────────────────────
       CHAT AREA
    ───────────────────────────────────── */

    .chat-topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.9rem 1.1rem; border-bottom: 1px solid var(--border);
        background: rgba(15,23,42,0.55); flex-shrink: 0;
    }
    .chat-topbar-left { display: flex; align-items: center; gap: 10px; }
    .chat-topbar-icon {
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, rgba(109,40,217,0.14), rgba(37,99,235,0.10));
        border: 1px solid rgba(109,40,217,0.18);
        display: flex; align-items: center; justify-content: center; font-size: 0.95rem;
        box-shadow: 0 10px 20px rgba(109,40,217,0.10);
    }
    .chat-topbar-title { font-size: 0.9rem; font-weight: 900; color: var(--text); }
    .chat-topbar-sub   { font-size: 0.6rem; color: var(--text-muted); margin-top: 1px; }
    .chat-msg-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: var(--radius-full);
        background: rgba(109,40,217,0.06); border: 1px solid rgba(109,40,217,0.14);
        font-size: 0.6rem; font-weight: 800; color: var(--primary);
    }

    /* Scroll wrapper */
    .chat-messages-wrap {
        flex: 1; overflow-y: auto; padding: 1.2rem 1.1rem;
        scroll-behavior: smooth; scrollbar-width: thin;
        scrollbar-color: rgba(109,40,217,0.25) transparent;
        max-height: 560px;
    }
    .chat-surface {
        background: rgba(13,21,40,0.38);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 22px 60px rgba(0,0,0,0.24);
    }

    /* Message animations */
    .msg-block { margin-bottom: 1.4rem; animation: msgIn 0.32s cubic-bezier(0.22,1,0.36,1) both; }
    @keyframes msgIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

    /* User row */
    .msg-user-row { display: flex; align-items: flex-end; justify-content: flex-end; gap: 10px; margin-bottom: 8px; }
    .msg-user-bubble {
        max-width: 72%;
        background: linear-gradient(145deg, rgba(37,99,235,0.24), rgba(14,165,233,0.18));
        border: 1px solid rgba(56,189,248,0.24); border-radius: 18px 4px 18px 18px;
        padding: 0.75rem 1rem;
        box-shadow: 0 10px 20px rgba(15,23,42,0.06), inset 0 1px 0 rgba(255,255,255,0.20);
    }
    .msg-user-text { font-size: 0.87rem; color: #e0f2fe; line-height: 1.65; margin: 0; }
    .msg-user-avatar {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg, var(--primary), var(--primary-2));
        display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
        box-shadow: 0 10px 18px rgba(109,40,217,0.16); border: 1.5px solid rgba(109,40,217,0.18);
    }

    /* AI row */
    .msg-ai-row { display: flex; align-items: flex-start; justify-content: flex-start; gap: 10px; }
    .msg-ai-avatar {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; margin-top: 18px;
        display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
        border: 1.5px solid rgba(15,23,42,0.08);
    }
    .msg-ai-avatar-rag {
        background: linear-gradient(135deg, rgba(22,163,74,0.92), rgba(14,165,233,0.85));
        box-shadow: 0 10px 18px rgba(22,163,74,0.14); border-color: rgba(22,163,74,0.18);
    }
    .msg-ai-avatar-fallback {
        background: linear-gradient(135deg, rgba(217,119,6,0.92), rgba(245,158,11,0.88));
        box-shadow: 0 10px 18px rgba(217,119,6,0.14); border-color: rgba(217,119,6,0.18);
    }
    .msg-ai-content { max-width: 84%; display: flex; flex-direction: column; gap: 5px; }
    .msg-ai-meta {
        margin-left: 40px;
        max-width: 860px;
    }
    @media (max-width: 900px) {
        .msg-ai-meta { margin-left: 0; max-width: 100%; }
    }

    /* Mode badge */
    .mode-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: var(--radius-full);
        font-size: 0.6rem; font-weight: 800; letter-spacing: 0.04em;
        width: fit-content; margin-bottom: 2px;
    }
    .mode-badge-rag { background: rgba(22,163,74,0.08); border: 1px solid rgba(22,163,74,0.18); color: var(--success); }
    .mode-badge-fallback { background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.18); color: var(--warning); }

    /* AI bubble */
    .msg-ai-bubble {
        background: rgba(15,23,42,0.78); border: 1px solid var(--border);
        border-radius: 4px 18px 18px 18px; padding: 0.85rem 1.05rem;
        box-shadow: var(--shadow-sm); transition: border-color 0.2s ease;
    }
    .msg-ai-bubble-rag     { border-color: rgba(22,163,74,0.12); }
    .msg-ai-bubble-fallback { border-color: rgba(217,119,6,0.12); }
    .msg-ai-bubble:hover   { border-color: var(--border-hover); }

    /* Answer text */
    .answer-box { font-size: 0.88rem; color: #dbeafe; line-height: 1.82; padding: 0; word-wrap: break-word; overflow-wrap: anywhere; }
    .answer-box p { margin: 0.55rem 0; }
    .answer-box p:first-child { margin-top: 0; }
    .answer-box p:last-child { margin-bottom: 0; }
    .answer-box h1, .answer-box h2, .answer-box h3, .answer-box h4 {
        color: #7dd3fc !important; font-weight: 700 !important; margin: 0.65rem 0 0.3rem;
    }
    .answer-box h1 { font-size: 1.05rem !important; }
    .answer-box h2 { font-size: 0.98rem !important; }
    .answer-box h3 { font-size: 0.92rem !important; }
    .answer-box h4 { font-size: 0.86rem !important; }
    .answer-box strong { color: #ffffff; font-weight: 800; }
    .answer-box em     { color: #bae6fd; font-style: italic; }
    .answer-box code {
        background: rgba(14,165,233,0.16); color: #bae6fd;
        font-family: 'JetBrains Mono', monospace; font-size: 0.77rem;
        padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(125,211,252,0.18);
    }
    .answer-box pre {
        background: rgba(2,6,23,0.65); border: 1px solid rgba(148,163,184,0.22);
        border-radius: var(--radius-sm); padding: 0.8rem 1rem; overflow-x: auto; margin: 0.55rem 0;
    }
    .answer-box hr { border: 0; height: 1px; background: rgba(148,163,184,0.14); margin: 0.9rem 0; }
    .answer-box a { color: #7dd3fc; text-decoration: none; border-bottom: 1px solid rgba(125,211,252,0.25); }
    .answer-box a:hover { border-bottom-color: rgba(125,211,252,0.55); }
    .answer-box pre code { background: transparent; border: none; padding: 0; color: #e2e8f0; font-size: 0.76rem; }
    .answer-box ul, .answer-box ol { margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }
    .answer-box li { margin-bottom: 0.25rem; color: #bfdbfe; }
    .answer-box li::marker { color: #38bdf8; }
    .answer-box p { margin: 0.28rem 0; }
    .answer-box blockquote {
        border-left: 3px solid rgba(56,189,248,0.45); margin: 0.4rem 0;
        padding: 0.3rem 0 0.3rem 0.8rem; color: #bae6fd; font-style: italic;
        background: rgba(56,189,248,0.08); border-radius: 0 6px 6px 0;
    }
    /* cite-chip removed — chunk references are stripped from answers */
    .code-lang {
        display: inline-block;
        font-size: 0.62rem;
        color: #94a3b8;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* Confidence bar */
    .conf-wrap { padding: 4px 0 8px; }
    .conf-row  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .conf-label { font-size: 0.58rem; font-weight: 800; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.09em; }
    .conf-val   { font-size: 0.68rem; font-weight: 900; }
    .conf-track { background: rgba(15,23,42,0.06); border-radius: 99px; height: 3px; overflow: hidden; }
    .conf-fill  { height: 100%; border-radius: 99px; transition: width 1.1s cubic-bezier(0.4,0,0.2,1); }
    .conf-high   { background: linear-gradient(90deg,#16a34a,#22c55e); }
    .conf-medium { background: linear-gradient(90deg,#d97706,#f59e0b); }
    .conf-low    { background: linear-gradient(90deg,#dc2626,#fb7185); }

    /* Source card */
    .source-card {
        background: rgba(109,40,217,0.03); border: 1px solid rgba(15,23,42,0.08);
        border-left: 3px solid rgba(109,40,217,0.35); border-radius: var(--radius-sm);
        padding: 0.75rem 0.9rem; margin-bottom: 0.5rem; transition: all 0.2s ease;
    }
    .source-card:hover { background: rgba(109,40,217,0.06); border-left-color: var(--primary); transform: translateX(2px); }
    .source-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .source-label { font-size: 0.58rem; font-weight: 900; color: var(--primary); text-transform: uppercase; letter-spacing: 0.08em; }
    .source-page  { font-size: 0.56rem; color: var(--text-muted); font-weight: 700; background: rgba(15,23,42,0.03); padding: 2px 7px; border-radius: 99px; border: 1px solid rgba(15,23,42,0.08); }
    .source-text  { font-size: 0.73rem; color: #cbd5e1; line-height: 1.62; font-family: 'JetBrains Mono', monospace; margin: 0; }

    /* Message divider */
    .msg-divider { height: 1px; background: rgba(15,23,42,0.07); margin: 0.5rem 0 1.2rem; }

    /* Input area */
    .input-hint {
        font-size: 0.58rem; color: var(--text-dim); text-align: center; margin-top: 6px;
        display: flex; align-items: center; justify-content: center; gap: 6px;
    }
    .composer-shell {
        position: sticky;
        bottom: 0.75rem;
        z-index: 20;
        background: linear-gradient(180deg, rgba(6,11,22,0.0), rgba(6,11,22,0.88) 32%, rgba(6,11,22,0.96) 100%);
        padding-top: 1rem;
    }
    .composer-card {
        background: rgba(13,21,40,0.90);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 22px;
        padding: 0.8rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.32);
        backdrop-filter: blur(12px);
    }
    .kbd {
        background: rgba(15,23,42,0.04); border: 1px solid rgba(15,23,42,0.10);
        padding: 1px 6px; border-radius: 4px; font-size: 0.55rem; font-family: 'JetBrains Mono', monospace;
    }

    /* Thinking indicator */
    .thinking-bar {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: var(--radius-md);
        background: rgba(109,40,217,0.05); border: 1px solid rgba(109,40,217,0.14); margin-bottom: 8px;
    }
    .thinking-dots span {
        display: inline-block; width: 5px; height: 5px; border-radius: 50%;
        background: var(--primary); margin: 0 2px;
        animation: dotBounce 1.4s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.14s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.28s; }
    @keyframes dotBounce { 0%, 80%, 100% { transform: scale(0.5); opacity: 0.35; } 40% { transform: scale(1.1); opacity: 1; } }

    /* Empty state */
    .empty-state {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 4.25rem 2rem;
        background:
            linear-gradient(180deg, rgba(15,23,42,0.72), rgba(15,23,42,0.52)),
            radial-gradient(circle at top, rgba(56,189,248,0.12), transparent 55%);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 24px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.28);
        min-height: 460px;
    }
    .empty-eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 12px; border-radius: 999px;
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.18);
        color: #7dd3fc; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 1rem;
    }
    .empty-icon { font-size: 2.8rem; margin-bottom: 1rem; filter: drop-shadow(0 14px 28px rgba(56,189,248,0.16)); }
    .empty-title { font-size: 2.1rem; font-weight: 900; color: #f8fafc; margin-bottom: 10px; letter-spacing: -0.04em; }
    .empty-sub   { font-size: 0.95rem; color: #94a3b8; max-width: 620px; line-height: 1.8; margin-bottom: 1.6rem; }
    .empty-grid {
        width: 100%;
        max-width: 760px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 0.4rem;
    }
    .empty-card {
        text-align: left;
        background: rgba(2,6,23,0.28);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 14px 15px;
    }
    .empty-card-label {
        font-size: 0.62rem;
        color: #7dd3fc;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .empty-card-text {
        font-size: 0.8rem;
        color: #dbeafe;
        line-height: 1.6;
    }
    .starter-prompt-wrap {
        max-width: 760px;
        width: 100%;
        margin-top: 1rem;
    }
    .starter-prompt-title {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .starter-prompt-note {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.8rem;
    }
    @media (max-width: 900px) {
        .empty-grid { grid-template-columns: 1fr; }
        .empty-title { font-size: 1.6rem; }
        .empty-sub { font-size: 0.88rem; }
    }

    /* Warn bar */
    .warn-bar {
        display: flex; align-items: center; gap: 9px;
        padding: 10px 14px; border-radius: var(--radius-md);
        background: rgba(217,119,6,0.06); border: 1px solid rgba(217,119,6,0.14); margin-bottom: 8px;
    }

    /* File card */
    .file-card {
        background: rgba(37,99,235,0.06); border: 1px solid rgba(37,99,235,0.12);
        border-radius: var(--radius-md); padding: 9px 12px;
        display: flex; align-items: center; gap: 10px; margin-top: 9px;
    }

    /* Ready card */
    .ready-card {
        display: flex; align-items: center; gap: 9px;
        background: rgba(22,163,74,0.06);
        border: 1px solid rgba(22,163,74,0.14); border-radius: var(--radius-md);
        padding: 10px 13px; margin-top: 9px;
        box-shadow: 0 12px 24px rgba(22,163,74,0.07);
        animation: fadeUp 0.4s cubic-bezier(0.22,1,0.36,1);
    }
    .ready-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--success); box-shadow: 0 0 0 3px rgba(22,163,74,0.14); }

    /* Proc steps */
    .proc-step { display: flex; align-items: center; gap: 7px; font-size: 0.7rem; padding: 3px 0; }
    .dot-done   { width: 6px; height: 6px; border-radius: 50%; background: var(--success); flex-shrink: 0; }
    .dot-active { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); flex-shrink: 0; animation: blink 1s ease-in-out infinite; }

    /* Suggested query how-card */
    .how-card { background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.12); border-radius: var(--radius-md); padding: 12px 13px; }
    .how-num  { width: 20px; height: 20px; border-radius: 50%; background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25); display: inline-flex; align-items: center; justify-content: center; font-size: 0.56rem; font-weight: 900; color: #f59e0b; flex-shrink: 0; }

    /* Metric strips */
    .strip-teal    { height: 2px; border-radius: 2px; background: linear-gradient(90deg,var(--primary),var(--primary-2)); margin-top: 4px; opacity: 0.75; }
    .strip-gold    { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#d97706,#f59e0b); margin-top: 4px; opacity: 0.65; }
    .strip-emerald { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#16a34a,#22c55e); margin-top: 4px; opacity: 0.65; }
    .strip-violet  { height: 2px; border-radius: 2px; background: linear-gradient(90deg,var(--primary),#a78bfa); margin-top: 4px; opacity: 0.65; }

    /* Glow divider */
    .glow-div {
        height: 1px; margin: 1rem 0;
        background: linear-gradient(90deg, transparent 0%, rgba(109,40,217,0.30) 25%, rgba(37,99,235,0.18) 65%, transparent 100%);
    }

    /* Footer */
    .footer-bar { text-align: center; padding: 0.6rem 0; font-size: 0.66rem; color: var(--text-muted); font-weight: 600; }

    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "db_ready":      False,
        "last_file":     None,
        "chat_history":  [],
        "total_queries": 0,
        "total_docs":    0,
        "conf_scores":   [],
        "prefill_query": "",
        "auto_submit":   False,
        "input_key":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════
def avg_confidence() -> int:
    s = st.session_state.conf_scores
    return int(sum(s) / len(s)) if s else 0


def strip_chunk_references(text: str) -> str:
    """Remove [Chunk X], [Chunks X, Y], and stray 'and [Chunk X]' from answer text."""
    # Remove patterns like [Chunk 1], [Chunk 2], [Chunks 1, 3], [Chunks 1, 2, 3]
    text = re.sub(r"\[Chunks?\s*[\d,\s]+\]", "", text)
    # Remove leftover 'and' before the removed chunk refs (e.g., "and [Chunk 3]" → "")
    text = re.sub(r"\s+and\s+(?=\s|\.|\,|$)", " ", text)
    # Clean up any double spaces left behind
    text = re.sub(r"  +", " ", text)
    # Clean up spaces before punctuation
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()



def conf_bar_html(value: int) -> str:
    if value >= 80:
        cls, label, color = "conf-high",   "High",   "#34d399"
    elif value >= 60:
        cls, label, color = "conf-medium", "Medium", "#fbbf24"
    else:
        cls, label, color = "conf-low",    "Low",    "#fb7185"
    return (
        '<div class="conf-wrap">'
        '<div class="conf-row">'
        '<span class="conf-label">Confidence</span>'
        f'<span class="conf-val" style="color:{color};">{label} · {value}%</span>'
        '</div>'
        '<div class="conf-track">'
        f'<div class="conf-fill {cls}" style="width:{value}%;"></div>'
        '</div></div>'
    )


def render_markdown_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def replace_code_block(m):
        lang = m.group(1).strip() if m.group(1) else ""
        code = m.group(2)
        lang_label = (
            f'<span class="code-lang">{lang}</span>'
            if lang else ""
        )
        return f"<pre>{lang_label}<code>{code}</code></pre>"

    text = re.sub(r"```(\w*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`",    r"<code>\1</code>", text)
    text = re.sub(r"^####\s+(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s+(.+)$",  r"<h3>\1</h3>",  text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.+)$",   r"<h2>\1</h2>",   text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.+)$",    r"<h1>\1</h1>",    text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$",       "<hr>",            text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__",          r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",          r"<em>\1</em>", text)
    text = re.sub(r"_(.+?)_",            r"<em>\1</em>", text)

    def replace_blockquote(m):
        inner = re.sub(r"^&gt;\s?", "", m.group(0), flags=re.MULTILINE)
        return f"<blockquote>{inner.strip()}</blockquote>"
    text = re.sub(r"(^&gt;.+(\n|$))+", replace_blockquote, text, flags=re.MULTILINE)

    def replace_ul(m):
        items = [re.sub(r"^[\*\-\+]\s+", "", i.strip()) for i in m.group(0).strip().split("\n") if i.strip()]
        return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    text = re.sub(r"(^[\*\-\+] .+(\n|$))+", replace_ul, text, flags=re.MULTILINE)

    def replace_ol(m):
        items = [re.sub(r"^\d+\.\s+", "", i.strip()) for i in m.group(0).strip().split("\n") if i.strip()]
        return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"
    text = re.sub(r"(^\d+\. .+(\n|$))+", replace_ol, text, flags=re.MULTILINE)

    # Chunk references are stripped before rendering — no cite-chip needed.

    block_tags = ("<h1", "<h2", "<h3", "<h4", "<ul", "<ol", "<pre", "<hr", "<blockquote")
    parts = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith(block_tags):
            parts.append(para)
        else:
            parts.append("<p>" + para.replace("\n", "<br>") + "</p>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════
# REUSABLE COMPONENTS
# ═══════════════════════════════════════════════════════
def panel_open(icon: str, title: str, sub: str, icon_cls: str = "p-icon-teal"):
    st.markdown(
        '<div class="panel">'
        '<div class="panel-header">'
        f'<div class="p-icon {icon_cls}">{icon}</div>'
        '<div>'
        f'<div class="p-title">{title}</div>'
        f'<div class="p-sub">{sub}</div>'
        '</div></div>'
        '<div class="panel-body">',
        unsafe_allow_html=True,
    )


def panel_close():
    st.markdown("</div></div>", unsafe_allow_html=True)


def proc_step_html(icon: str, msg: str, color: str, dot_cls: str) -> str:
    return (
        f'<div class="proc-step">'
        f'<div class="{dot_cls}"></div>'
        f'<span style="color:{color};font-size:0.7rem;font-weight:600;">{icon} {msg}</span>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════
# CHAT MESSAGE RENDERER
# ═══════════════════════════════════════════════════════
def render_chat_message(chat: dict, index: int, total: int):
    mode       = chat.get("mode")
    is_rag     = mode == "rag"
    is_not_doc = mode == "not_found"
    mode_cls   = "rag" if is_rag else "fallback"
    if is_rag:
        badge_text = "📄 From Document"
    elif is_not_doc:
        badge_text = "ℹ️ Not in document"
    else:
        badge_text = "🌐 General Knowledge"
    ai_av_cls  = "msg-ai-avatar-rag" if is_rag else "msg-ai-avatar-fallback"
    bubble_cls = f"msg-ai-bubble-{mode_cls}"

    # User message
    st.markdown(
        '<div class="msg-user-row">'
        '<div class="msg-user-bubble">'
        f'<p class="msg-user-text">{html_module.escape(chat["question"])}</p>'
        '</div>'
        '<div class="msg-user-avatar">👤</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Strip chunk references like [Chunk 1], [Chunks 1, 3] from the answer
    clean_answer = strip_chunk_references(chat["answer"])
    rendered_answer = render_markdown_to_html(clean_answer)

    # AI message
    st.markdown(
        '<div class="msg-ai-row">'
        f'<div class="msg-ai-avatar {ai_av_cls}">🤖</div>'
        '<div class="msg-ai-content">'
        f'<span class="mode-badge mode-badge-{mode_cls}">{badge_text}</span>'
        f'<div class="msg-ai-bubble {bubble_cls}">'
        f'<div class="answer-box">{rendered_answer}</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Confidence bar (RAG only)
    if is_rag and chat.get("confidence", 0) > 0:
        st.markdown('<div class="msg-ai-meta">', unsafe_allow_html=True)
        st.markdown(conf_bar_html(chat["confidence"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sources expander
    chat_docs = chat.get("docs", [])
    if is_rag and chat_docs:
        st.markdown('<div class="msg-ai-meta">', unsafe_allow_html=True)
        with st.expander(f"📚 View sources  ·  {len(chat_docs)} source{'s' if len(chat_docs) != 1 else ''}"):
            for idx, doc in enumerate(chat_docs):
                page = doc.get("page", "?")
                snippet = html_module.escape(doc.get("content", "")[:420])
                st.markdown(
                    '<div class="source-card">'
                    '<div class="source-card-header">'
                    f'<span class="source-label">Source {idx + 1}</span>'
                    f'<span class="source-page">Page {page}</span>'
                    '</div>'
                    f'<p class="source-text">{snippet}…</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    if index < total - 1:
        st.markdown('<div class="msg-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # ── Branding header ──
        st.markdown(
            '<div style="display:flex;align-items:center;gap:13px;padding:0.5rem 0 1.25rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title" style="font-size:1.2rem;">RAG Assistant</div>'
            '<div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;font-weight:600;">'
            'Document Intelligence Platform</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # ── Document Upload Section ──
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-section" style="padding:0 0 10px 0;">'
            '📁 &nbsp;Document Upload</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"], label_visibility="collapsed"
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
            fname = uploaded_file.name[:28] + ("…" if len(uploaded_file.name) > 28 else "")
            st.markdown(
                '<div class="sidebar-file-info">'
                '<div class="sidebar-file-icon">📄</div>'
                '<div>'
                f'<div style="font-size:0.78rem;font-weight:800;color:#e2e8f0;">{fname}</div>'
                f'<div style="font-size:0.65rem;color:#94a3b8;margin-top:3px;">{size_str} · PDF</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

            if not st.session_state.db_ready:
                if st.button(
                    "⚡  Process Document",
                    use_container_width=True,
                    key="proc_btn_sidebar",
                ):
                    step_ph = st.empty()
                    bar = st.progress(0)
                    try:
                        for icon, msg, p_start, p_end in [
                            ("🔍", "Parsing PDF…", 5, 20),
                            ("✂️", "Chunking text…", 30, 50),
                            ("🔢", "Generating embeddings…", 60, 85),
                            ("📦", "Indexing FAISS…", 95, 100),
                        ]:
                            step_ph.markdown(
                                proc_step_html(icon, msg, "#38bdf8", "dot-active"),
                                unsafe_allow_html=True,
                            )
                            bar.progress(p_start)
                            if icon == "🔍":
                                docs_raw = load_pdf(file_path)
                            elif icon == "✂️":
                                chunks = split_documents(docs_raw)
                            elif icon == "🔢":
                                store_embeddings(chunks)
                            else:
                                time.sleep(0.3)
                            bar.progress(p_end)

                        step_ph.markdown(
                            proc_step_html("✅", "Document ready!", "#34d399", "dot-done"),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.8)
                        step_ph.empty()
                        bar.empty()
                        st.session_state.db_ready = True
                        st.session_state.total_docs += 1
                        st.rerun()
                    except Exception as exc:
                        step_ph.empty()
                        bar.empty()
                        st.error(f"❌ Processing failed: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)  # close upload card

        # ── Status Section ──
        st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-section" style="padding:0 0 6px 0;">'
            '📊 &nbsp;Status</div>',
            unsafe_allow_html=True,
        )

        db_ready = st.session_state.get("db_ready", False)
        dot_cls = (
            "sidebar-status-dot sidebar-status-dot-on"
            if db_ready
            else "sidebar-status-dot sidebar-status-dot-off"
        )
        title = "Document ready" if db_ready else "No document indexed"
        subtitle = "Chat is now enabled." if db_ready else "Upload and process a PDF."
        st.markdown(
            f'<div class="sidebar-status-item">'
            f'<div class="{dot_cls}"></div>'
            f'<div>'
            f'<div style="font-size:0.82rem;font-weight:800;color:#e2e8f0;">{title}</div>'
            f'<div style="font-size:0.68rem;color:#94a3b8;margin-top:3px;">{subtitle}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-kpi">'
            '<div class="sidebar-kpi-item">'
            '<div class="sidebar-kpi-label">Documents</div>'
            f'<div class="sidebar-kpi-val">{st.session_state.total_docs}</div></div>'
            '<div class="sidebar-kpi-item">'
            '<div class="sidebar-kpi-label">Queries</div>'
            f'<div class="sidebar-kpi-val">{st.session_state.total_queries}</div></div>'
            '<div class="sidebar-kpi-item">'
            '<div class="sidebar-kpi-label">Avg Conf</div>'
            f'<div class="sidebar-kpi-val">{avg_confidence()}%</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)  # close status card

        # ── Suggested Queries Section ──
        st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-section" style="padding:0 0 10px 0;">'
            '⭐ &nbsp;Quick Queries</div>',
            unsafe_allow_html=True,
        )

        if db_ready:
            for q in [
                "What is the main topic?",
                "Summarize the key findings.",
                "What methodology was used?",
                "What are the main conclusions?",
                "List the most important points.",
            ]:
                if st.button(
                    f"›  {q}",
                    use_container_width=True,
                    key=f"sq_sidebar_{q}",
                    type="secondary",
                ):
                    st.session_state.prefill_query = q
                    st.session_state.auto_submit = True
                    st.rerun()
        else:
            st.markdown(
                '<div style="font-size:0.76rem;color:#64748b;padding:6px 0 4px;'
                'line-height:1.65;">'
                'Upload and process a PDF to see suggested queries here.'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)  # close queries card

        # ── Actions Section ──
        if st.session_state.chat_history:
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            if st.button(
                "🗑️  Clear Chat",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.chat_history  = []
                st.session_state.total_queries = 0
                st.session_state.conf_scores   = []
                st.session_state.prefill_query = ""
                st.rerun()


# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
def render_header():
    status_color = "#16a34a" if st.session_state.db_ready else "#6d28d9"
    status_label = "Document ready" if st.session_state.db_ready else "System online"

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding-bottom:1.2rem;'
            'border-bottom:1px solid rgba(15,23,42,0.08);margin-bottom:1.2rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title" style="font-size:1.75rem;">RAG Assistant</div>'
            '<div style="font-size:0.72rem;color:#64748b;margin-top:3px;font-weight:600;">'
            'Document Intelligence &nbsp;·&nbsp; Mistral Small + FAISS + LangChain'
            '</div></div></div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.markdown(
            '<div style="text-align:right;padding-top:8px;">'
            '<div class="status-pill">'
            f'<span class="status-dot" style="background:{status_color};box-shadow:0 0 7px {status_color};"></span>'
            f'<span style="color:{status_color};">{status_label}</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════
def render_metrics():
    m1, m2, m3, m4 = st.columns(4)
    conf_val = f"{avg_confidence()}%" if st.session_state.conf_scores else "—"
    with m1:
        st.metric("📄  Documents",     st.session_state.total_docs)
        st.markdown('<div class="strip-teal"></div>', unsafe_allow_html=True)
    with m2:
        st.metric("💬  Queries",        st.session_state.total_queries)
        st.markdown('<div class="strip-gold"></div>', unsafe_allow_html=True)
    with m3:
        st.metric("📊  Avg Confidence", conf_val)
        st.markdown('<div class="strip-emerald"></div>', unsafe_allow_html=True)
    with m4:
        st.metric("⚡  Model",          "Mistral Small")
        st.markdown('<div class="strip-violet"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LEFT COLUMN
# ═══════════════════════════════════════════════════════
def render_left_column():
    # ── Upload ──────────────────────────────────────────
    panel_open("📁", "Document Upload", "PDF files · up to 50 MB", "p-icon-teal")
    uploaded_file = st.file_uploader("Upload", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        is_new = st.session_state.last_file != uploaded_file.name
        if is_new:
            st.session_state.db_ready  = False
            st.session_state.last_file = uploaded_file.name
            from retrieval.retriever import get_vectorstore
            get_vectorstore.clear()

        os.makedirs("data/docs", exist_ok=True)
        file_path = os.path.join("data/docs", uploaded_file.name)
        if is_new or not os.path.exists(file_path):
            with open(file_path, "wb") as fh:
                fh.write(uploaded_file.getbuffer())

        size_str = f"{round(uploaded_file.size / 1024, 1)} KB"
        fname    = uploaded_file.name[:32] + ("…" if len(uploaded_file.name) > 32 else "")
        st.markdown(
            '<div class="file-card">'
            '<span style="font-size:1.3rem;">📄</span>'
            '<div>'
            f'<div style="font-size:0.74rem;font-weight:700;color:#e2e8f0;">{fname}</div>'
            f'<div style="font-size:0.58rem;color:#475569;margin-top:2px;">{size_str} · PDF</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.db_ready:
            if st.button("⚡  Process Document", use_container_width=True, key="proc_btn"):
                step_ph = st.empty()
                bar     = st.progress(0)
                try:
                    for icon, msg, p_start, p_end in [
                        ("🔍", "Parsing PDF…",          5,  20),
                        ("✂️",  "Chunking text…",        30, 50),
                        ("🔢", "Generating embeddings…", 60, 85),
                        ("📦", "Indexing FAISS…",        95, 100),
                    ]:
                        step_ph.markdown(proc_step_html(icon, msg, "#2dd4bf", "dot-active"), unsafe_allow_html=True)
                        bar.progress(p_start)
                        if   icon == "🔍": docs_raw = load_pdf(file_path)
                        elif icon == "✂️":  chunks   = split_documents(docs_raw)
                        elif icon == "🔢": store_embeddings(chunks)
                        else:              time.sleep(0.3)
                        bar.progress(p_end)

                    step_ph.markdown(proc_step_html("✅", "Document ready!", "#34d399", "dot-done"), unsafe_allow_html=True)
                    time.sleep(0.8)
                    step_ph.empty(); bar.empty()
                    st.session_state.db_ready    = True
                    st.session_state.total_docs += 1
                    st.rerun()
                except Exception as exc:
                    step_ph.empty(); bar.empty()
                    st.error(f"❌ Processing failed: {exc}")

        if st.session_state.db_ready:
            fname_disp = uploaded_file.name[:28] + ("…" if len(uploaded_file.name) > 28 else "")
            st.markdown(
                '<div class="ready-card">'
                '<div class="ready-dot"></div>'
                '<div>'
                '<div style="font-size:0.74rem;font-weight:800;color:#34d399;">Ready for Queries</div>'
                f'<div style="font-size:0.59rem;color:#6ee7b7;opacity:0.75;margin-top:2px;">"{fname_disp}" indexed &amp; loaded</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
    panel_close()

    # ── Suggested Queries ───────────────────────────────
    panel_open("⭐", "Suggested Queries", "Quick-start questions", "p-icon-gold")
    if st.session_state.db_ready:
        for q in [
            "What is the main topic?",
            "Summarize the key findings.",
            "What methodology was used?",
            "What are the main conclusions?",
            "List the most important points.",
        ]:
            if st.button(f"›  {q}", use_container_width=True, key=f"sq_{q}", type="secondary"):
                st.session_state.prefill_query = q
                st.session_state.auto_submit   = True
                st.rerun()
    else:
        st.markdown(
            '<div class="how-card">'
            '<div style="font-size:0.58rem;font-weight:900;color:#f59e0b;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:10px;">💡 How It Works</div>'
            + "".join([
                f'<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:8px;">'
                f'<div class="how-num">{n}</div>'
                f'<div style="font-size:0.74rem;color:#64748b;line-height:1.55;">{t}</div></div>'
                for n, t in [
                    (1, 'Upload your <strong style="color:#fcd34d;">PDF document</strong>'),
                    (2, 'Click <strong style="color:#fcd34d;">Process Document</strong>'),
                    (3, 'Ask questions in the <strong style="color:#fcd34d;">chat</strong>'),
                    (4, 'Get <strong style="color:#fcd34d;">cited answers</strong> with confidence scores'),
                ]
            ])
            + '</div>',
            unsafe_allow_html=True,
        )
    panel_close()

    # (Removed: Model Stack panel) — reduces visual noise without changing workflow.


# ═══════════════════════════════════════════════════════
# CHAT COLUMN
# ═══════════════════════════════════════════════════════
def render_chat_column():
    msg_count = len(st.session_state.chat_history)
    prefill   = st.session_state.get("prefill_query", "")

    # ── Top bar ────────────────────────────────────────
    badge_html = ""
    if msg_count:
        badge_html = (
            f'<span class="chat-msg-badge">'
            f'{msg_count} message{"s" if msg_count != 1 else ""}'
            f'</span>'
        )

    st.markdown('<div class="chat-surface">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-topbar">'
        '<div class="chat-topbar-left">'
        '<div class="chat-topbar-icon">💬</div>'
        '<div>'
        '<div class="chat-topbar-title">Conversation</div>'
        '<div class="chat-topbar-sub">Ask anything about your document</div>'
        '</div>'
        '</div>'
        + badge_html +
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Messages area ──────────────────────────────────
    st.markdown('<div class="chat-messages-wrap">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-eyebrow">AI Document Workspace</div>'
            '<div class="empty-icon">💬</div>'
            '<div class="empty-title">Ask better questions. Get grounded answers.</div>'
            '<div class="empty-sub">Upload a PDF from the sidebar, process it, and chat with your document in a focused workspace. '
            'Responses stay citation-aware, confidence-scored, and easy to verify.</div>'
            '<div class="empty-grid">'
            '<div class="empty-card">'
            '<div class="empty-card-label">Summaries</div>'
            '<div class="empty-card-text">Generate quick executive summaries and extract the main findings.</div>'
            '</div>'
            '<div class="empty-card">'
            '<div class="empty-card-label">Evidence</div>'
            '<div class="empty-card-text">See source-backed answers with confidence signals for each response.</div>'
            '</div>'
            '<div class="empty-card">'
            '<div class="empty-card-label">Fallback</div>'
            '<div class="empty-card-text">If the document lacks context, switch to the external model only when needed.</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="starter-prompt-wrap"><div class="starter-prompt-title">Try these prompts</div></div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            if st.button("Summarize this document", use_container_width=True, key="starter_1", type="secondary"):
                st.session_state.prefill_query = "Summarize this document."
                if st.session_state.db_ready:
                    st.session_state.auto_submit = True
                st.rerun()
        with p2:
            if st.button("What are the key findings?", use_container_width=True, key="starter_2", type="secondary"):
                st.session_state.prefill_query = "What are the key findings?"
                if st.session_state.db_ready:
                    st.session_state.auto_submit = True
                st.rerun()
        with p3:
            if st.button("Explain the methodology", use_container_width=True, key="starter_3", type="secondary"):
                st.session_state.prefill_query = "Explain the methodology used in the document."
                if st.session_state.db_ready:
                    st.session_state.auto_submit = True
                st.rerun()
        if not st.session_state.db_ready:
            st.markdown('<div class="starter-prompt-note">These prompts will auto-run after you upload and process a document.</div>', unsafe_allow_html=True)
    else:
        total = len(st.session_state.chat_history)
        for i, chat in enumerate(st.session_state.chat_history):
            render_chat_message(chat, i, total)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

    # ── Warn / fallback helper ─────────────────────────
    if not st.session_state.db_ready:
        st.markdown(
            '<div class="warn-bar">'
            '<span>⚠️</span>'
            '<span style="font-size:0.78rem;color:#fbbf24;font-weight:500;">'
            'Upload and process a PDF document to start chatting.'
            '</span></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last.get("mode") == "not_found" and last.get("allow_fallback"):
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    '<div class="warn-bar" style="margin-bottom:0;">'
                    '<span>ℹ️</span>'
                    '<span style="font-size:0.78rem;color:#e5e7eb;font-weight:500;">'
                    'This answer is based only on your uploaded document and the required context was not found.'
                    '</span></div>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("🌐 Use web model", use_container_width=True, key="fallback_trigger_btn"):
                    with st.spinner("Fetching answer from web model…"):
                        try:
                            fallback_llm = get_fallback_llm()
                            fb_answer    = fallback_llm.invoke(last["question"]).content
                            last["answer"]         = fb_answer
                            last["mode"]           = "fallback"
                            last["allow_fallback"] = False
                            last["confidence"]     = 0
                            last["docs"]           = []
                            st.session_state.chat_history[-1] = last
                        except Exception as exc:
                            st.error(f"❌ Fallback failed: {exc}")
                    st.rerun()

    # ── Query input ────────────────────────────────────
    st.markdown('<div class="composer-shell"><div class="composer-card">', unsafe_allow_html=True)
    q_l, q_r = st.columns([8, 1])
    with q_l:
        query_input = st.text_input(
            "Query",
            value=prefill,
            placeholder="Ask anything about your document…" if st.session_state.db_ready else "Process a document first…",
            disabled=not st.session_state.db_ready,
            label_visibility="collapsed",
            key=f"query_input_{st.session_state.input_key}",
        )
    with q_r:
        send_clicked = st.button(
            "Send ➤" if st.session_state.db_ready else "🔒",
            use_container_width=True,
            disabled=not st.session_state.db_ready,
            key="send_btn",
        )

    st.markdown(
        '<div class="input-hint">'
        '<span class="kbd">Enter</span> or click <strong style="color:#94a3b8;">Send</strong>'
        ' &nbsp;·&nbsp; Powered by Mistral Small + FAISS RAG'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

    if prefill:
        st.session_state.prefill_query = ""

    # ── Handle submit ──────────────────────────────────
    auto_sub = st.session_state.get("auto_submit", False)
    if auto_sub:
        st.session_state.auto_submit = False

    query_to_run = ""
    if (send_clicked or auto_sub) and query_input and query_input.strip():
        query_to_run = query_input.strip()
    elif query_input and query_input.strip() and query_input != prefill:
        query_to_run = query_input.strip()

    if query_to_run:
        if not st.session_state.db_ready:
            st.warning("⚠️ Please upload and process a document first.")
        else:
            thinking_ph = st.empty()
            thinking_ph.markdown(
                '<div class="thinking-bar">'
                '<div class="thinking-dots"><span></span><span></span><span></span></div>'
                '<span style="font-size:0.8rem;color:#2dd4bf;font-weight:600;">Reasoning through your document…</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            answer, docs, mode, confidence = "", [], "fallback", 0
            try:
                vectorstore = get_vectorstore()
                llm         = get_mistral_llm()
                rag_chain   = create_rag_chain(llm, vectorstore)
                answer_gen, docs, results = rag_chain(query_to_run, st.session_state.chat_history)
                confidence  = calculate_confidence(results)
                answer      = "".join(list(answer_gen))

                # If the RAG chain indicates the answer is not in the document,
                # keep the reply neutral and offer an explicit fallback button.
                if answer.strip().startswith("NOT_FOUND"):
                    mode        = "not_found"
                    docs        = []
                    confidence  = 0
                    answer      = "The required context is not present in the uploaded document."
                else:
                    mode = "rag"

            except TimeoutError as exc:
                thinking_ph.empty()
                st.error(
                    f"⏱️ **Request Timeout**: {str(exc)}\n\n"
                    "The Mistral API took too long. Please try again in a moment."
                )
                answer = ""
            except Exception as exc:
                thinking_ph.empty()
                st.error(f"❌ An error occurred: {str(exc)}\n\nCheck your API key and try again.")
                answer = ""

            thinking_ph.empty()

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


# ═══════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════
def render_footer():
    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="footer-bar">'
        'Powered by '
        '<span style="color:#0f172a;font-weight:800;">Mistral Small</span> · '
        '<span style="color:#0f172a;font-weight:800;">FAISS</span> · '
        '<span style="color:#0f172a;font-weight:800;">LangChain</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    inject_css()
    init_session_state()
    render_sidebar()
    render_chat_column()


if __name__ == "__main__":
    main()