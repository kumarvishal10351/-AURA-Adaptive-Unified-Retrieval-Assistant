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
        --bg: #06080f;
        --surface: #0d1117;
        --surface-2: #161b27;
        --surface-3: #1c2333;
        --teal: #0d9488;
        --teal-lt: #2dd4bf;
        --teal-dim: rgba(13,148,136,0.15);
        --gold: #f59e0b;
        --gold-lt: #fcd34d;
        --violet: #7c3aed;
        --violet-lt: #a78bfa;
        --blue: #1d4ed8;
        --blue-lt: #60a5fa;
        --emerald: #059669;
        --emerald-lt: #34d399;
        --rose: #e11d48;
        --rose-lt: #fb7185;
        --text: #e2e8f0;
        --text-muted: #64748b;
        --text-dim: #334155;
        --border: rgba(255,255,255,0.06);
        --border-hover: rgba(255,255,255,0.12);
        --user-bubble-bg: linear-gradient(145deg, rgba(13,148,136,0.2) 0%, rgba(12,74,110,0.15) 100%);
        --user-bubble-border: rgba(45,212,191,0.25);
        --ai-bubble-bg: rgba(255,255,255,0.028);
        --ai-bubble-border: rgba(255,255,255,0.07);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-full: 9999px;
    }

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Background ── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 10% -10%, rgba(13,148,136,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 90% 80%, rgba(124,58,237,0.05) 0%, transparent 60%),
            linear-gradient(rgba(13,148,136,0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(13,148,136,0.018) 1px, transparent 1px),
            var(--bg);
        background-size: auto, auto, 48px 48px, 48px 48px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(13,148,136,0.4), rgba(124,58,237,0.3));
        border-radius: 99px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1rem 0.8rem !important; }

    /* ── Main container ── */
    .main .block-container {
        padding: 1.5rem 2rem 3rem !important;
        max-width: 100% !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4 { color: #fff !important; font-weight: 800 !important; letter-spacing: -0.025em; }

    /* ── App title ── */
    .app-title {
        background: linear-gradient(135deg, #2dd4bf 0%, #a78bfa 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        letter-spacing: -0.04em;
    }

    /* ── Logo ── */
    .logo-box {
        width: 44px; height: 44px; border-radius: 13px;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #0c4a6e 100%);
        display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
        box-shadow: 0 0 0 1px rgba(45,212,191,0.2), 0 0 24px rgba(13,148,136,0.4);
        animation: logoPulse 3s ease-in-out infinite; flex-shrink: 0;
    }
    @keyframes logoPulse {
        0%, 100% { box-shadow: 0 0 0 1px rgba(45,212,191,0.2), 0 0 24px rgba(13,148,136,0.4); }
        50%       { box-shadow: 0 0 0 1px rgba(45,212,191,0.35), 0 0 40px rgba(13,148,136,0.6); }
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 6px 14px; border-radius: var(--radius-full);
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
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
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35) !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important; font-size: 0.6rem !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 800 !important;
    }
    [data-testid="stMetricValue"] {
        color: #fff !important; font-size: 1.55rem !important;
        font-weight: 900 !important; letter-spacing: -0.04em !important;
    }

    /* ── Panel ── */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin-bottom: 0.85rem;
    }
    .panel-header {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid var(--border);
        display: flex; align-items: center; gap: 9px;
        background: rgba(255,255,255,0.012);
    }
    .panel-body { padding: 1rem; }
    .p-icon {
        width: 28px; height: 28px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center; font-size: 0.8rem;
    }
    .p-icon-teal   { background: rgba(13,148,136,0.18); }
    .p-icon-gold   { background: rgba(245,158,11,0.15); }
    .p-icon-violet { background: rgba(124,58,237,0.15); }
    .p-title { font-size: 0.78rem !important; font-weight: 800 !important; color: #fff !important; }
    .p-sub   { font-size: 0.59rem; color: var(--text-muted); margin-top: 1px; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: radial-gradient(ellipse at 50% 80%, rgba(13,148,136,0.06) 0%, transparent 70%) !important;
        border: 2px dashed rgba(13,148,136,0.3) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.2rem !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(13,148,136,0.55) !important;
        box-shadow: 0 0 28px rgba(13,148,136,0.1) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 55%, #0c4a6e 100%) !important;
        color: white !important;
        border: 1px solid rgba(45,212,191,0.2) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 18px rgba(13,148,136,0.28), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(13,148,136,0.5), 0 0 0 1px rgba(45,212,191,0.35) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        color: #475569 !important; box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(225,29,72,0.07) !important;
        border-color: rgba(225,29,72,0.25) !important;
        color: #fb7185 !important; transform: none !important;
    }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.72rem 1rem !important;
        caret-color: var(--teal-lt) !important;
        transition: all 0.22s ease !important;
    }
    .stTextInput > div > div > input:focus {
        background: rgba(13,148,136,0.05) !important;
        border-color: rgba(13,148,136,0.45) !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(148,163,184,0.3) !important; }
    label[data-testid="stWidgetLabel"] {
        color: var(--text-dim) !important; font-size: 0.65rem !important;
        font-weight: 800 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
    }

    /* ── Alerts ── */
    .stSuccess {
        background: linear-gradient(135deg, rgba(5,150,105,0.1), rgba(13,148,136,0.05)) !important;
        border: 1px solid rgba(5,150,105,0.25) !important; border-radius: var(--radius-md) !important;
    }
    .stWarning {
        background: linear-gradient(135deg, rgba(245,158,11,0.07), rgba(252,211,77,0.03)) !important;
        border: 1px solid rgba(245,158,11,0.22) !important; border-radius: var(--radius-md) !important;
    }
    .stError {
        background: rgba(225,29,72,0.07) !important;
        border: 1px solid rgba(225,29,72,0.22) !important; border-radius: var(--radius-md) !important;
    }

    /* ── Progress ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0d9488, #a78bfa, #34d399) !important;
        background-size: 200% 100% !important;
        animation: shimmer 1.8s linear infinite !important;
        border-radius: 99px !important;
    }
    .stProgress > div { border-radius: 99px !important; background: rgba(255,255,255,0.05) !important; height: 4px !important; }
    @keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(13,148,136,0.05) !important;
        border: 1px solid rgba(13,148,136,0.18) !important;
        border-radius: var(--radius-md) !important;
        color: var(--teal-lt) !important; font-weight: 700 !important; font-size: 0.78rem !important;
        transition: all 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover { background: rgba(13,148,136,0.1) !important; }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.012) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: none !important; border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }

    /* ─────────────────────────────────────
       SIDEBAR NAV
    ───────────────────────────────────── */
    .sidebar-nav-item {
        display: flex; align-items: center; gap: 9px;
        padding: 8px 11px; border-radius: var(--radius-sm); margin-bottom: 3px;
        font-size: 0.78rem; font-weight: 600; color: var(--text-muted);
        border: 1px solid transparent; transition: all 0.18s ease; cursor: pointer;
    }
    .sidebar-nav-item:hover { background: rgba(255,255,255,0.03); color: #94a3b8; }
    .sidebar-nav-item.active {
        background: rgba(13,148,136,0.1); color: var(--teal-lt);
        border-color: rgba(13,148,136,0.22);
    }
    .sidebar-section { font-size: 0.56rem; font-weight: 900; color: #1e3a5f; text-transform: uppercase; letter-spacing: 0.12em; padding: 8px 10px 4px; }
    .sidebar-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 7px 0; }
    .sidebar-status-item { display: flex; align-items: center; gap: 9px; padding: 7px 11px; border-radius: var(--radius-sm); margin-bottom: 3px; }
    .sidebar-status-dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; box-shadow: 0 0 7px #34d399; flex-shrink: 0; }

    /* ─────────────────────────────────────
       CHAT AREA — Complete Redesign
    ───────────────────────────────────── */

    /* Chat container */
    .chat-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        min-height: 560px;
    }

    /* Chat header bar */
    .chat-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.1rem;
        border-bottom: 1px solid var(--border);
        background: rgba(255,255,255,0.015);
        flex-shrink: 0;
    }
    .chat-topbar-left { display: flex; align-items: center; gap: 10px; }
    .chat-topbar-icon {
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, rgba(13,148,136,0.25), rgba(12,74,110,0.2));
        border: 1px solid rgba(13,148,136,0.25);
        display: flex; align-items: center; justify-content: center; font-size: 0.95rem;
        box-shadow: 0 0 14px rgba(13,148,136,0.15);
    }
    .chat-topbar-title { font-size: 0.88rem; font-weight: 800; color: #fff; }
    .chat-topbar-sub   { font-size: 0.6rem; color: var(--text-muted); margin-top: 1px; }
    .chat-msg-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: var(--radius-full);
        background: rgba(13,148,136,0.1); border: 1px solid rgba(13,148,136,0.22);
        font-size: 0.6rem; font-weight: 800; color: var(--teal-lt);
    }

    /* Scroll wrapper */
    .chat-messages-wrap {
        flex: 1;
        overflow-y: auto;
        padding: 1.2rem 1.1rem;
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: rgba(13,148,136,0.3) transparent;
        max-height: 500px;
    }

    /* Message block (one full exchange) */
    .msg-block {
        margin-bottom: 1.4rem;
        animation: msgIn 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    @keyframes msgIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

    /* User row */
    .msg-user-row {
        display: flex; align-items: flex-end; justify-content: flex-end; gap: 10px;
        margin-bottom: 8px;
    }
    .msg-user-bubble {
        max-width: 72%;
        background: linear-gradient(145deg, rgba(13,148,136,0.22), rgba(12,74,110,0.17));
        border: 1px solid rgba(45,212,191,0.22);
        border-radius: 18px 4px 18px 18px;
        padding: 0.75rem 1rem;
        box-shadow: 0 4px 20px rgba(13,148,136,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
    }
    .msg-user-text {
        font-size: 0.87rem; color: #dbeafe; line-height: 1.65; margin: 0;
    }
    .msg-user-avatar {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg, #0d9488, #0c4a6e);
        display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
        box-shadow: 0 0 10px rgba(13,148,136,0.4); border: 1.5px solid rgba(45,212,191,0.2);
    }

    /* AI row */
    .msg-ai-row {
        display: flex; align-items: flex-start; justify-content: flex-start; gap: 10px;
    }
    .msg-ai-avatar {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; margin-top: 18px;
        display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
        border: 1.5px solid rgba(255,255,255,0.1);
    }
    .msg-ai-avatar-rag {
        background: linear-gradient(135deg, #059669, #0d9488);
        box-shadow: 0 0 10px rgba(5,150,105,0.4);
        border-color: rgba(52,211,153,0.25);
    }
    .msg-ai-avatar-fallback {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        box-shadow: 0 0 10px rgba(245,158,11,0.4);
        border-color: rgba(251,191,36,0.25);
    }
    .msg-ai-content { max-width: 84%; display: flex; flex-direction: column; gap: 5px; }

    /* Mode badge */
    .mode-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: var(--radius-full);
        font-size: 0.6rem; font-weight: 800; letter-spacing: 0.04em;
        width: fit-content; margin-bottom: 2px;
    }
    .mode-badge-rag {
        background: rgba(5,150,105,0.1); border: 1px solid rgba(5,150,105,0.28); color: #34d399;
    }
    .mode-badge-fallback {
        background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.28); color: #fbbf24;
    }

    /* AI bubble */
    .msg-ai-bubble {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 4px 18px 18px 18px;
        padding: 0.85rem 1.05rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        transition: border-color 0.2s ease;
    }
    .msg-ai-bubble-rag     { border-color: rgba(5,150,105,0.18); }
    .msg-ai-bubble-fallback { border-color: rgba(245,158,11,0.18); }
    .msg-ai-bubble:hover   { border-color: rgba(255,255,255,0.1); }

    /* Answer text */
    .answer-box {
        font-size: 0.87rem; color: #cbd5e1; line-height: 1.78; padding: 0;
    }
    .answer-box h1, .answer-box h2, .answer-box h3, .answer-box h4 {
        color: #2dd4bf !important; font-weight: 700 !important; margin: 0.65rem 0 0.3rem;
    }
    .answer-box h1 { font-size: 1.05rem !important; }
    .answer-box h2 { font-size: 0.98rem !important; }
    .answer-box h3 { font-size: 0.92rem !important; }
    .answer-box h4 { font-size: 0.86rem !important; }
    .answer-box strong { color: #fcd34d; font-weight: 700; }
    .answer-box em     { color: #a78bfa; font-style: italic; }
    .answer-box code {
        background: rgba(13,148,136,0.1); color: #2dd4bf;
        font-family: 'JetBrains Mono', monospace; font-size: 0.77rem;
        padding: 2px 6px; border-radius: 5px; border: 1px solid rgba(13,148,136,0.18);
    }
    .answer-box pre {
        background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.07);
        border-radius: var(--radius-sm); padding: 0.8rem 1rem; overflow-x: auto; margin: 0.55rem 0;
    }
    .answer-box pre code { background: transparent; border: none; padding: 0; color: #e2e8f0; font-size: 0.76rem; }
    .answer-box ul, .answer-box ol { margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }
    .answer-box li { margin-bottom: 0.25rem; color: #94a3b8; }
    .answer-box li::marker { color: #0d9488; }
    .answer-box p { margin: 0.28rem 0; }
    .answer-box blockquote {
        border-left: 3px solid rgba(13,148,136,0.5); margin: 0.4rem 0;
        padding: 0.3rem 0 0.3rem 0.8rem; color: #64748b; font-style: italic;
        background: rgba(13,148,136,0.035); border-radius: 0 6px 6px 0;
    }

    /* Confidence bar */
    .conf-wrap { padding: 4px 0 8px; }
    .conf-row  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .conf-label { font-size: 0.58rem; font-weight: 800; color: #1e3a5f; text-transform: uppercase; letter-spacing: 0.09em; }
    .conf-val   { font-size: 0.68rem; font-weight: 900; }
    .conf-track { background: rgba(255,255,255,0.04); border-radius: 99px; height: 3px; overflow: hidden; }
    .conf-fill  { height: 100%; border-radius: 99px; transition: width 1.1s cubic-bezier(0.4,0,0.2,1); }
    .conf-high   { background: linear-gradient(90deg,#059669,#34d399); box-shadow: 0 0 8px rgba(52,211,153,0.5); }
    .conf-medium { background: linear-gradient(90deg,#d97706,#f59e0b); box-shadow: 0 0 8px rgba(245,158,11,0.5); }
    .conf-low    { background: linear-gradient(90deg,#be123c,#fb7185); box-shadow: 0 0 8px rgba(225,29,72,0.5); }

    /* Source card */
    .source-card {
        background: rgba(13,148,136,0.03); border: 1px solid rgba(255,255,255,0.05);
        border-left: 3px solid rgba(13,148,136,0.55); border-radius: var(--radius-sm);
        padding: 0.75rem 0.9rem; margin-bottom: 0.5rem; transition: all 0.2s ease;
    }
    .source-card:hover {
        background: rgba(13,148,136,0.07); border-left-color: var(--teal-lt);
        transform: translateX(2px);
    }
    .source-card-header {
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;
    }
    .source-label { font-size: 0.58rem; font-weight: 900; color: #2dd4bf; text-transform: uppercase; letter-spacing: 0.08em; }
    .source-page  { font-size: 0.56rem; color: var(--text-muted); font-weight: 700; background: rgba(255,255,255,0.04); padding: 2px 7px; border-radius: 99px; border: 1px solid rgba(255,255,255,0.07); }
    .source-text  { font-size: 0.73rem; color: #475569; line-height: 1.62; font-family: 'JetBrains Mono', monospace; margin: 0; }

    /* Message divider */
    .msg-divider { height: 1px; background: rgba(255,255,255,0.035); margin: 0.5rem 0 1.2rem; }

    /* ─────────────────────────────────────
       INPUT AREA
    ───────────────────────────────────── */
    .input-area {
        border-top: 1px solid var(--border);
        padding: 0.9rem 1.1rem;
        background: rgba(255,255,255,0.01);
        flex-shrink: 0;
    }
    .input-hint {
        font-size: 0.58rem; color: var(--text-dim); text-align: center; margin-top: 6px;
        display: flex; align-items: center; justify-content: center; gap: 6px;
    }
    .kbd {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        padding: 1px 6px; border-radius: 4px; font-size: 0.55rem; font-family: 'JetBrains Mono', monospace;
    }

    /* ── Thinking indicator ── */
    .thinking-bar {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: var(--radius-md);
        background: rgba(13,148,136,0.05); border: 1px solid rgba(13,148,136,0.18);
        margin-bottom: 8px;
    }
    .thinking-dots span {
        display: inline-block; width: 5px; height: 5px; border-radius: 50%;
        background: var(--teal-lt); margin: 0 2px;
        animation: dotBounce 1.4s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.14s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.28s; }
    @keyframes dotBounce { 0%, 80%, 100% { transform: scale(0.5); opacity: 0.35; } 40% { transform: scale(1.1); opacity: 1; } }

    /* ── Empty state ── */
    .empty-state {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 3.5rem 2rem;
        background: radial-gradient(ellipse at 50% 50%, rgba(13,148,136,0.055) 0%, transparent 65%);
        min-height: 380px;
    }
    .empty-icon {
        font-size: 2.6rem; margin-bottom: 1rem;
        animation: floatIcon 3s ease-in-out infinite;
        filter: drop-shadow(0 0 16px rgba(13,148,136,0.28));
    }
    @keyframes floatIcon { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
    .empty-title { font-size: 1.05rem; font-weight: 900; color: #fff; margin-bottom: 6px; }
    .empty-sub   { font-size: 0.78rem; color: #334155; max-width: 340px; line-height: 1.7; margin-bottom: 1.2rem; }
    .empty-chips { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
    .empty-chip {
        font-size: 0.62rem; padding: 4px 11px; border-radius: var(--radius-full);
        font-weight: 700; cursor: default;
    }

    /* ── Warn bar ── */
    .warn-bar {
        display: flex; align-items: center; gap: 9px;
        padding: 10px 14px; border-radius: var(--radius-md);
        background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.18);
        margin-bottom: 8px;
    }

    /* ── File card ── */
    .file-card {
        background: rgba(29,78,216,0.07); border: 1px solid rgba(29,78,216,0.18);
        border-radius: var(--radius-md); padding: 9px 12px;
        display: flex; align-items: center; gap: 10px; margin-top: 9px;
    }

    /* ── Ready card ── */
    .ready-card {
        display: flex; align-items: center; gap: 9px;
        background: linear-gradient(135deg, rgba(5,150,105,0.11), rgba(13,148,136,0.06));
        border: 1px solid rgba(5,150,105,0.25); border-radius: var(--radius-md);
        padding: 10px 13px; margin-top: 9px;
        box-shadow: 0 0 20px rgba(5,150,105,0.07);
        animation: fadeUp 0.4s cubic-bezier(0.22,1,0.36,1);
    }
    .ready-dot {
        width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
        background: #34d399; box-shadow: 0 0 9px #34d399, 0 0 20px rgba(52,211,153,0.4);
        animation: blink 2s ease-in-out infinite;
    }

    /* ── Proc steps ── */
    .proc-step { display: flex; align-items: center; gap: 7px; font-size: 0.7rem; padding: 3px 0; }
    .dot-done   { width: 6px; height: 6px; border-radius: 50%; background: #34d399; box-shadow: 0 0 6px rgba(52,211,153,0.6); flex-shrink: 0; }
    .dot-active { width: 6px; height: 6px; border-radius: 50%; background: var(--teal-lt); box-shadow: 0 0 6px rgba(45,212,191,0.6); flex-shrink: 0; animation: blink 1s ease-in-out infinite; }

    /* ── Suggested query buttons ── */
    .how-card { background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.12); border-radius: var(--radius-md); padding: 12px 13px; }
    .how-num  { width: 20px; height: 20px; border-radius: 50%; background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25); display: inline-flex; align-items: center; justify-content: center; font-size: 0.56rem; font-weight: 900; color: #f59e0b; flex-shrink: 0; }

    /* ── Model stack ── */
    .model-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .model-row:last-child { border-bottom: none; }

    /* ── Metric strips ── */
    .strip-teal    { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#0d9488,#2dd4bf); box-shadow: 0 0 8px rgba(13,148,136,0.35); margin-top: 4px; }
    .strip-gold    { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#d97706,#f59e0b); box-shadow: 0 0 8px rgba(245,158,11,0.32); margin-top: 4px; }
    .strip-emerald { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#059669,#34d399); box-shadow: 0 0 8px rgba(52,211,153,0.32); margin-top: 4px; }
    .strip-violet  { height: 2px; border-radius: 2px; background: linear-gradient(90deg,#7c3aed,#a78bfa); box-shadow: 0 0 8px rgba(124,58,237,0.32); margin-top: 4px; }

    /* ── Glow divider ── */
    .glow-div {
        height: 1px; margin: 1rem 0;
        background: linear-gradient(90deg, transparent 0%, rgba(13,148,136,0.32) 25%, rgba(124,58,237,0.22) 65%, transparent 100%);
    }

    /* ── Footer ── */
    .footer-bar { text-align: center; padding: 0.6rem 0; font-size: 0.66rem; color: #1e3a5f; font-weight: 500; }

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
            f'<span style="font-size:0.57rem;color:#475569;font-weight:700;">{lang}</span>'
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
    is_rag   = chat.get("mode") == "rag"
    mode_cls = "rag" if is_rag else "fallback"
    badge_text = "📄 From Document" if is_rag else "🌐 General Knowledge"
    ai_av_cls  = "msg-ai-avatar-rag" if is_rag else "msg-ai-avatar-fallback"
    ai_av_icon = "🤖"
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

    rendered_answer = render_markdown_to_html(chat["answer"])

    # AI message
    st.markdown(
        '<div class="msg-ai-row">'
        f'<div class="msg-ai-avatar {ai_av_cls}">{ai_av_icon}</div>'
        '<div class="msg-ai-content">'
        f'<span class="mode-badge mode-badge-{mode_cls}">{badge_text}</span>'
        f'<div class="msg-ai-bubble {bubble_cls}">'
        f'<div class="answer-box">{rendered_answer}</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Confidence bar (only for RAG)
    if is_rag and chat.get("confidence", 0) > 0:
        st.markdown(
            f'<div style="padding-left:40px;">{conf_bar_html(chat["confidence"])}</div>',
            unsafe_allow_html=True,
        )

    # Sources expander
    chat_docs = chat.get("docs", [])
    if is_rag and chat_docs:
        with st.expander(f"📚 View sources  ·  {len(chat_docs)} chunk{'s' if len(chat_docs) != 1 else ''}"):
            for idx, doc in enumerate(chat_docs):
                page    = doc.get("page", "?")
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

    if index < total - 1:
        st.markdown('<div class="msg-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo & brand
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;padding:0.4rem 0 1rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title" style="font-size:1.1rem;">RAG Assistant</div>'
            '<div style="font-size:0.57rem;color:#1e3a5f;margin-top:2px;font-weight:500;">Document Intelligence</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

        for icon, label in [("💬", "Chat"), ("📚", "Knowledge Base"), ("⚙️", "Settings")]:
            active = st.session_state.get("active_tab", "chat") == label.lower().replace(" ", "_")
            st.markdown(
                f'<div class="sidebar-nav-item {"active" if active else ""}">'
                f'<span>{icon}</span>{label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Pipeline Status</div>', unsafe_allow_html=True)

        for icon, name, role, color in [
            ("🤖", "Mistral Small", "Primary LLM",   "#2dd4bf"),
            ("🗄️",  "ChromaDB",     "Vector Store",  "#60a5fa"),
            ("🔢", "HuggingFace",  "Embeddings",    "#fcd34d"),
            ("🔗", "LangChain",    "RAG Pipeline",  "#34d399"),
        ]:
            st.markdown(
                f'<div class="sidebar-status-item">'
                f'<div class="sidebar-status-dot"></div>'
                f'<div>'
                f'<div style="font-size:0.7rem;font-weight:700;color:{color};">{icon} {name}</div>'
                f'<div style="font-size:0.57rem;color:#1e3a5f;">{role}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            if st.button("🗑️  Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.chat_history  = []
                st.session_state.total_queries = 0
                st.session_state.conf_scores   = []
                st.session_state.prefill_query = ""
                st.rerun()


# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
def render_header():
    status_color = "#34d399" if st.session_state.db_ready else "#2dd4bf"
    status_label = "Document Ready" if st.session_state.db_ready else "System Online"

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding-bottom:1.2rem;'
            'border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.2rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title" style="font-size:1.75rem;">RAG Assistant</div>'
            '<div style="font-size:0.67rem;color:#1e3a5f;margin-top:3px;font-weight:500;">'
            'Document Intelligence &nbsp;·&nbsp; Mistral Small + ChromaDB + LangChain'
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
                        ("🔍", "Parsing PDF…",         5,  20),
                        ("✂️",  "Chunking text…",       30, 50),
                        ("🔢", "Generating embeddings…",60, 85),
                        ("📦", "Indexing vector store…",95, 100),
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

    # ── Model Stack ─────────────────────────────────────
    panel_open("🔧", "Model Stack", "AI pipeline components", "p-icon-violet")
    rows = [
        ("LLM",         "Mistral Small", "#2dd4bf"),
        ("Vector Store","ChromaDB",      "#60a5fa"),
        ("Embeddings",  "HuggingFace",   "#fcd34d"),
        ("Pipeline",    "LangChain RAG", "#34d399"),
        ("Fallback",    "General LLM",   "#a78bfa"),
    ]
    st.markdown(
        '<div style="background:rgba(255,255,255,0.018);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:9px 12px;">'
        + "".join([
            f'<div class="model-row">'
            f'<span style="font-size:0.67rem;color:#334155;">{k}</span>'
            f'<span style="font-size:0.67rem;font-weight:700;color:{c};">{v}</span>'
            f'</div>'
            for k, v, c in rows
        ])
        + '</div>',
        unsafe_allow_html=True,
    )
    panel_close()


# ═══════════════════════════════════════════════════════
# CHAT COLUMN — Redesigned
# ═══════════════════════════════════════════════════════
def render_chat_column():
    msg_count  = len(st.session_state.chat_history)
    prefill    = st.session_state.get("prefill_query", "")

    # ── Chat container top bar ─────────────────────────
    badge_html = ""
    if msg_count:
        badge_html = (
            f'<span class="chat-msg-badge">'
            f'{msg_count} message{"s" if msg_count != 1 else ""}'
            f'</span>'
        )

    # Top bar row: title + badge + clear button
    hdr_l, hdr_r = st.columns([8, 1])
    with hdr_l:
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
    with hdr_r:
        if st.session_state.chat_history:
            if st.button("🗑️", key="clear_btn", use_container_width=True, type="secondary"):
                st.session_state.chat_history  = []
                st.session_state.total_queries = 0
                st.session_state.conf_scores   = []
                st.session_state.prefill_query = ""
                st.rerun()

    # ── Messages area ──────────────────────────────────
    st.markdown('<div class="chat-messages-wrap">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">💬</div>'
            '<div class="empty-title">Ready to Assist</div>'
            '<div class="empty-sub">Upload a PDF, process it, then ask any question. '
            'Answers arrive with source citations and live confidence scores.</div>'
            '<div class="empty-chips">'
            '<span class="empty-chip" style="background:rgba(13,148,136,0.1);border:1px solid rgba(13,148,136,0.25);color:#2dd4bf;">🔒 Private</span>'
            '<span class="empty-chip" style="background:rgba(29,78,216,0.1);border:1px solid rgba(29,78,216,0.25);color:#60a5fa;">⚡ Fast RAG</span>'
            '<span class="empty-chip" style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);color:#fcd34d;">📊 Confidence</span>'
            '<span class="empty-chip" style="background:rgba(5,150,105,0.1);border:1px solid rgba(5,150,105,0.25);color:#34d399;">📚 Sources</span>'
            '<span class="empty-chip" style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.25);color:#a78bfa;">🧠 Mistral</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        total = len(st.session_state.chat_history)
        for i, chat in enumerate(st.session_state.chat_history):
            render_chat_message(chat, i, total)

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-messages-wrap

    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

    # ── Warn bar ───────────────────────────────────────
    if not st.session_state.db_ready:
        st.markdown(
            '<div class="warn-bar">'
            '<span>⚠️</span>'
            '<span style="font-size:0.78rem;color:#fbbf24;font-weight:500;">'
            'Upload and process a PDF document to start chatting.'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ── Query input ────────────────────────────────────
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
        '<span class="kbd">Enter</span> or click <strong style="color:#334155;">Send</strong>'
        ' &nbsp;·&nbsp; Powered by Mistral Small + ChromaDB RAG'
        '</div>',
        unsafe_allow_html=True,
    )

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

                if answer.strip().startswith("NOT_FOUND"):
                    fallback_llm = get_fallback_llm()
                    answer       = fallback_llm.invoke(query_to_run).content
                    docs         = []; confidence = 0; mode = "fallback"
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
                    "question":   query_to_run,
                    "answer":     answer,
                    "mode":       mode,
                    "confidence": confidence,
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
        '<span style="color:#0d9488;font-weight:800;">Mistral Small</span> · '
        '<span style="color:#1d4ed8;font-weight:800;">ChromaDB</span> · '
        '<span style="color:#059669;font-weight:800;">LangChain</span>'
        '&nbsp;|&nbsp; RAG Pipeline · v5.1'
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
    render_header()
    render_metrics()

    left_col, chat_col = st.columns([5, 8], gap="large")
    with left_col:
        render_left_column()
    with chat_col:
        render_chat_column()

    render_footer()


if __name__ == "__main__":
    main()