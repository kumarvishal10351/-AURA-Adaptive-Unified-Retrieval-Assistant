import streamlit as st
import os
import re
import time
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import store_embeddings
from app.retrieval.retriever import get_vectorstore
from app.llm.mistral_client import get_mistral_llm
from app.llm.fallback import get_fallback_llm
from app.chains.rag_chain import create_rag_chain
from app.utils.confidence import calculate_confidence
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
# GLOBAL CSS — Modern SaaS Dark Theme
# ═══════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg:#060a14; --surface:#0b1526; --teal:#0d9488; --teal-lt:#2dd4bf;
        --gold:#f59e0b; --gold-lt:#fcd34d; --violet:#7c3aed; --violet-lt:#a78bfa;
        --blue:#1d4ed8; --blue-lt:#60a5fa; --emerald:#059669; --emerald-lt:#34d399;
        --text:#e2e8f0; --muted:#475569; --border:rgba(255,255,255,0.07);
    }

    /* Base */
    html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background:#060a14 !important;color:var(--text) !important;}
    #MainMenu,footer,header{visibility:hidden}
    .stDeployButton{display:none}

    /* Background grid + orbs */
    .stApp{
        background:
            linear-gradient(rgba(13,148,136,0.025) 1px,transparent 1px),
            linear-gradient(90deg,rgba(13,148,136,0.025) 1px,transparent 1px),#060a14;
        background-size:52px 52px;
    }
    .stApp::before{
        content:'';position:fixed;width:700px;height:700px;border-radius:50%;
        background:radial-gradient(circle,rgba(13,148,136,0.12) 0%,transparent 70%);
        top:-220px;left:-180px;filter:blur(110px);pointer-events:none;z-index:0;
        animation:orb1 28s ease-in-out infinite;
    }
    .stApp::after{
        content:'';position:fixed;width:550px;height:550px;border-radius:50%;
        background:radial-gradient(circle,rgba(124,58,237,0.08) 0%,transparent 70%);
        top:40%;right:-160px;filter:blur(110px);pointer-events:none;z-index:0;
        animation:orb2 22s ease-in-out infinite;
    }
    @keyframes orb1{0%,100%{transform:translate(0,0)}33%{transform:translate(80px,-60px)}66%{transform:translate(-45px,80px)}}
    @keyframes orb2{0%,100%{transform:translate(0,0)}50%{transform:translate(-70px,55px)}}

    /* Scrollbar */
    ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}
    ::-webkit-scrollbar-thumb{background:linear-gradient(180deg,rgba(13,148,136,0.5),rgba(124,58,237,0.4));border-radius:99px}

    /* Sidebar */
    [data-testid="stSidebar"]{
        background:var(--surface) !important;
        border-right:1px solid var(--border) !important;
        padding:0 !important;
    }
    [data-testid="stSidebar"] > div:first-child{padding:1rem 0.75rem !important;}
    [data-testid="stSidebarNavItems"]{display:none;}

    /* Main container */
    .main .block-container{
        padding:1.5rem 2rem 3rem !important;
        max-width:100% !important;
    }

    /* Typography */
    h1,h2,h3,h4{color:#fff !important;font-weight:800 !important;letter-spacing:-0.02em}

    /* App title gradient */
    .app-title{
        background:linear-gradient(135deg,#2dd4bf 0%,#a78bfa 45%,#60a5fa 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
        font-size:1.85rem !important;font-weight:900 !important;letter-spacing:-0.045em;
        line-height:1.05;display:inline-block;
    }

    /* Logo */
    .logo-box{
        width:48px;height:48px;border-radius:14px;flex-shrink:0;
        background:linear-gradient(135deg,#0d9488,#0f766e 50%,#0c4a6e);
        display:flex;align-items:center;justify-content:center;font-size:1.4rem;
        box-shadow:0 0 0 1px rgba(45,212,191,0.2),0 0 28px rgba(13,148,136,0.45);
        animation:logoPulse 3s ease-in-out infinite;margin-right:12px;
    }
    @keyframes logoPulse{
        0%,100%{box-shadow:0 0 0 1px rgba(45,212,191,0.2),0 0 28px rgba(13,148,136,0.45)}
        50%{box-shadow:0 0 0 1px rgba(45,212,191,0.35),0 0 50px rgba(13,148,136,0.65)}
    }

    /* Status pill */
    .status-pill{
        display:inline-flex;align-items:center;gap:8px;
        padding:7px 16px;border-radius:99px;
        background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
        font-size:0.7rem;font-weight:800;
    }
    .status-dot{width:7px;height:7px;border-radius:50%;animation:blink 2.2s ease-in-out infinite;display:inline-block;}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

    /* Metric cards */
    [data-testid="stMetric"]{
        background:var(--surface) !important;border:1px solid var(--border) !important;
        border-radius:16px !important;padding:1rem 1.2rem 1.4rem !important;
        transition:all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover{transform:translateY(-3px) !important;box-shadow:0 14px 40px rgba(0,0,0,0.4) !important;}
    [data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:0.62rem !important;text-transform:uppercase !important;letter-spacing:0.09em !important;font-weight:800 !important;}
    [data-testid="stMetricValue"]{color:#fff !important;font-size:1.65rem !important;font-weight:900 !important;letter-spacing:-0.04em !important;}

    /* Panel components */
    .panel-head{
        background:var(--surface);border:1px solid var(--border);
        border-radius:14px 14px 0 0;padding:0.85rem 1.1rem;
        display:flex;align-items:center;gap:9px;
    }
    .panel-body{
        background:var(--surface);border:1px solid var(--border);border-top:none;
        border-radius:0 0 14px 14px;padding:1rem;margin-bottom:0.9rem;
    }
    .p-icon{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;}
    .p-icon-teal{background:rgba(13,148,136,0.18);box-shadow:0 0 12px rgba(13,148,136,0.15);}
    .p-icon-gold{background:rgba(245,158,11,0.15);box-shadow:0 0 12px rgba(245,158,11,0.12);}
    .p-icon-violet{background:rgba(124,58,237,0.15);box-shadow:0 0 12px rgba(124,58,237,0.12);}
    .p-title{font-size:0.8rem !important;font-weight:800 !important;color:#fff !important;}
    .p-sub{font-size:0.61rem;color:var(--muted);margin-top:1px;}

    /* File uploader */
    [data-testid="stFileUploader"]{
        background:radial-gradient(ellipse at 50% 70%,rgba(13,148,136,0.065) 0%,transparent 70%) !important;
        border:2px dashed rgba(13,148,136,0.32) !important;border-radius:14px !important;
        padding:1.4rem !important;transition:all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover{border-color:rgba(13,148,136,0.6) !important;box-shadow:0 0 36px rgba(13,148,136,0.12) !important;}

    /* Buttons */
    .stButton>button{
        background:linear-gradient(135deg,#0d9488 0%,#0f766e 55%,#0c4a6e 100%) !important;
        color:white !important;border:1px solid rgba(45,212,191,0.22) !important;
        border-radius:11px !important;font-weight:800 !important;
        font-family:'Inter',sans-serif !important;font-size:0.84rem !important;
        padding:0.6rem 1.3rem !important;
        box-shadow:0 4px 20px rgba(13,148,136,0.3),inset 0 1px 0 rgba(255,255,255,0.1) !important;
        transition:all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    }
    .stButton>button:hover{
        box-shadow:0 8px 35px rgba(13,148,136,0.55),0 0 0 1px rgba(45,212,191,0.4) !important;
        transform:translateY(-2px) !important;
    }
    .stButton>button[kind="secondary"]{
        background:rgba(255,255,255,0.03) !important;
        border:1px solid rgba(255,255,255,0.06) !important;
        color:#475569 !important;box-shadow:none !important;
    }
    .stButton>button[kind="secondary"]:hover{
        background:rgba(225,29,72,0.07) !important;
        border-color:rgba(225,29,72,0.25) !important;
        color:#fb7185 !important;transform:none !important;
    }

    /* Text input */
    .stTextInput>div>div>input{
        background:rgba(255,255,255,0.035) !important;
        border:1px solid rgba(255,255,255,0.09) !important;
        border-radius:12px !important;color:#e2e8f0 !important;
        font-family:'Inter',sans-serif !important;font-size:0.88rem !important;
        padding:0.7rem 1rem !important;caret-color:var(--teal-lt) !important;
        transition:all 0.25s ease !important;
    }
    .stTextInput>div>div>input:focus{
        background:rgba(13,148,136,0.055) !important;
        border-color:rgba(13,148,136,0.48) !important;
        box-shadow:0 0 0 3px rgba(13,148,136,0.11),0 0 18px rgba(13,148,136,0.07) !important;
    }
    .stTextInput>div>div>input::placeholder{color:rgba(148,163,184,0.36) !important;}
    label[data-testid="stWidgetLabel"]{
        color:#64748b !important;font-size:0.68rem !important;
        font-weight:800 !important;letter-spacing:0.07em !important;text-transform:uppercase !important;
    }

    /* Alerts */
    .stSuccess{background:linear-gradient(135deg,rgba(5,150,105,0.11),rgba(13,148,136,0.06)) !important;border:1px solid rgba(5,150,105,0.28) !important;border-radius:11px !important;}
    .stWarning{background:linear-gradient(135deg,rgba(245,158,11,0.08),rgba(252,211,77,0.04)) !important;border:1px solid rgba(245,158,11,0.26) !important;border-radius:11px !important;}
    .stError{background:rgba(225,29,72,0.08) !important;border:1px solid rgba(225,29,72,0.26) !important;border-radius:11px !important;}

    /* Progress bar */
    .stProgress>div>div>div>div{
        background:linear-gradient(90deg,#0d9488,#a78bfa,#f59e0b,#34d399) !important;
        background-size:300% 100% !important;
        animation:progShimmer 2s linear infinite !important;
        border-radius:99px !important;
    }
    .stProgress>div{border-radius:99px !important;background:rgba(255,255,255,0.05) !important;height:5px !important;}
    @keyframes progShimmer{0%{background-position:100% 0}100%{background-position:-100% 0}}

    /* Expander */
    .streamlit-expanderHeader{
        background:rgba(13,148,136,0.06) !important;
        border:1px solid rgba(13,148,136,0.2) !important;
        border-radius:11px !important;color:var(--teal-lt) !important;
        font-weight:700 !important;font-size:0.8rem !important;transition:all 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover{background:rgba(13,148,136,0.11) !important;}
    .streamlit-expanderContent{
        background:rgba(255,255,255,0.015) !important;
        border:1px solid rgba(255,255,255,0.055) !important;
        border-top:none !important;border-radius:0 0 11px 11px !important;
    }

    /* Sidebar nav items */
    .sidebar-nav-item{
        display:flex;align-items:center;gap:10px;
        padding:9px 12px;border-radius:10px;margin-bottom:4px;
        cursor:pointer;font-size:0.8rem;font-weight:600;color:#475569;
        border:1px solid transparent;transition:all 0.2s ease;
    }
    .sidebar-nav-item:hover{background:rgba(255,255,255,0.04);color:#94a3b8;}
    .sidebar-nav-item.active{
        background:rgba(13,148,136,0.12);color:#2dd4bf;
        border-color:rgba(13,148,136,0.25);
    }
    .sidebar-section-label{
        font-size:0.58rem;font-weight:800;color:#1e3a5f;
        text-transform:uppercase;letter-spacing:0.1em;
        padding:8px 10px 4px;margin-bottom:2px;
    }
    .sidebar-divider{height:1px;background:rgba(255,255,255,0.05);margin:8px 0;}
    .sidebar-status-item{
        display:flex;align-items:center;gap:9px;
        padding:8px 12px;border-radius:10px;margin-bottom:4px;
    }
    .sidebar-status-dot{
        width:7px;height:7px;border-radius:50%;
        background:#34d399;box-shadow:0 0 8px #34d399;flex-shrink:0;
    }

    /* Chat bubbles */
    .chat-user{
        background:linear-gradient(135deg,rgba(13,148,136,0.22),rgba(12,74,110,0.18));
        border:1px solid rgba(13,148,136,0.28);border-radius:18px 4px 18px 18px;
        padding:0.85rem 1.05rem;box-shadow:0 4px 20px rgba(13,148,136,0.1);
        animation:slideRight 0.35s cubic-bezier(0.22,1,0.36,1);
    }
    .chat-ai{
        background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
        border-radius:4px 18px 18px 18px;padding:0.85rem 1.05rem;
        box-shadow:0 4px 18px rgba(0,0,0,0.2);animation:slideLeft 0.35s cubic-bezier(0.22,1,0.36,1);
    }
    .chat-ai.rag{border-color:rgba(5,150,105,0.2);}
    .chat-ai.fallback{border-color:rgba(245,158,11,0.2);}
    @keyframes slideRight{from{opacity:0;transform:translateX(18px) scale(0.97)}to{opacity:1;transform:none}}
    @keyframes slideLeft{from{opacity:0;transform:translateX(-18px) scale(0.97)}to{opacity:1;transform:none}}

    /* Answer box */
    .answer-box{border:none;background:transparent;padding:0;font-size:0.86rem;color:#e2e8f0;line-height:1.8;}
    .answer-box h1,.answer-box h2,.answer-box h3,.answer-box h4{color:#2dd4bf !important;font-weight:700 !important;margin:0.7rem 0 0.35rem;}
    .answer-box h1{font-size:1.1rem !important}.answer-box h2{font-size:1rem !important}
    .answer-box h3{font-size:0.95rem !important}.answer-box h4{font-size:0.88rem !important}
    .answer-box strong{color:#fcd34d;font-weight:700}
    .answer-box em{color:#a78bfa;font-style:italic}
    .answer-box code{
        background:rgba(13,148,136,0.12);color:#2dd4bf;
        font-family:'JetBrains Mono',monospace;font-size:0.78rem;
        padding:2px 6px;border-radius:5px;border:1px solid rgba(13,148,136,0.2);
    }
    .answer-box pre{
        background:rgba(0,0,0,0.32);border:1px solid rgba(255,255,255,0.07);
        border-radius:10px;padding:0.85rem 1rem;overflow-x:auto;margin:0.6rem 0;
    }
    .answer-box pre code{background:transparent;border:none;padding:0;color:#e2e8f0;font-size:0.77rem;}
    .answer-box ul,.answer-box ol{margin:0.35rem 0 0.35rem 1rem;padding:0;}
    .answer-box li{margin-bottom:0.28rem;color:#cbd5e1;}
    .answer-box li::marker{color:#0d9488;}
    .answer-box p{margin:0.32rem 0;}
    .answer-box blockquote{
        border-left:3px solid rgba(13,148,136,0.52);margin:0.45rem 0;
        padding:0.35rem 0 0.35rem 0.85rem;color:#94a3b8;font-style:italic;
        background:rgba(13,148,136,0.04);border-radius:0 8px 8px 0;
    }

    /* Scrollable chat area */
    .chat-scroll-wrap{
        max-height:520px;overflow-y:auto;padding-right:4px;
        scrollbar-width:thin;scrollbar-color:rgba(13,148,136,0.4) transparent;
    }

    /* Confidence bars */
    .conf-wrap{padding:3px 0 7px}
    .conf-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
    .conf-label{font-size:0.6rem;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:0.08em}
    .conf-val{font-size:0.7rem;font-weight:900}
    .conf-track{background:rgba(255,255,255,0.05);border-radius:99px;height:4px;overflow:hidden}
    .conf-fill{height:100%;border-radius:99px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1)}
    .conf-high{background:linear-gradient(90deg,#059669,#34d399);box-shadow:0 0 9px rgba(52,211,153,0.6)}
    .conf-medium{background:linear-gradient(90deg,#d97706,#f59e0b);box-shadow:0 0 9px rgba(245,158,11,0.6)}
    .conf-low{background:linear-gradient(90deg,#be123c,#fb7185);box-shadow:0 0 9px rgba(225,29,72,0.6)}

    /* Badges */
    .badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:99px;font-size:0.63rem;font-weight:800;letter-spacing:0.04em;margin-bottom:5px}
    .badge-rag{background:rgba(5,150,105,0.12);border:1px solid rgba(5,150,105,0.3);color:#34d399}
    .badge-fallback{background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#fbbf24}

    /* Source cards */
    .source-card{
        background:rgba(13,148,136,0.04);border:1px solid rgba(255,255,255,0.055);
        border-left:3px solid rgba(13,148,136,0.62);border-radius:10px;
        padding:0.85rem 1rem;margin-bottom:0.55rem;transition:all 0.22s ease;
    }
    .source-card:hover{background:rgba(13,148,136,0.08);border-left-color:var(--teal-lt);}

    /* Query input wrapper */
    .query-wrap{
        background:var(--surface);border:1px solid var(--border);
        border-radius:16px;padding:0.75rem;
        box-shadow:0 0 36px rgba(13,148,136,0.04);transition:all 0.3s ease;
    }
    .query-wrap:focus-within{
        border-color:rgba(13,148,136,0.4);
        box-shadow:0 0 0 3px rgba(13,148,136,0.09),0 0 36px rgba(13,148,136,0.08);
    }

    /* Thinking animation */
    .thinking-bar{
        display:flex;align-items:center;gap:10px;padding:11px 15px;border-radius:12px;
        background:rgba(13,148,136,0.055);border:1px solid rgba(13,148,136,0.2);margin-bottom:9px;
    }
    .thinking-dots span{
        display:inline-block;width:5px;height:5px;border-radius:50%;background:#2dd4bf;
        margin:0 2px;animation:dotBounce 1.4s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2){animation-delay:0.16s}
    .thinking-dots span:nth-child(3){animation-delay:0.32s}
    @keyframes dotBounce{0%,80%,100%{transform:scale(0.5);opacity:0.4}40%{transform:scale(1);opacity:1}}

    /* File card */
    .file-card{
        background:rgba(29,78,216,0.07);border:1px solid rgba(29,78,216,0.2);
        border-radius:11px;padding:9px 12px;display:flex;align-items:center;gap:10px;
        margin-top:9px;box-shadow:0 0 18px rgba(29,78,216,0.07);
    }

    /* Ready card */
    .ready-card{
        display:flex;align-items:center;gap:9px;
        background:linear-gradient(135deg,rgba(5,150,105,0.13),rgba(13,148,136,0.07));
        border:1px solid rgba(5,150,105,0.3);border-radius:11px;
        padding:10px 13px;margin-top:9px;box-shadow:0 0 22px rgba(5,150,105,0.09);
        animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1);
    }
    .ready-dot{
        width:8px;height:8px;border-radius:50%;
        background:#34d399;box-shadow:0 0 10px #34d399,0 0 22px rgba(52,211,153,0.45);
        flex-shrink:0;animation:blink 2s ease-in-out infinite;
    }

    /* Processing steps */
    .proc-step{display:flex;align-items:center;gap:7px;font-size:0.7rem;padding:3px 0;color:#475569;}
    .proc-dot-done{width:6px;height:6px;border-radius:50%;background:#34d399;box-shadow:0 0 7px rgba(52,211,153,0.6);flex-shrink:0;}
    .proc-dot-active{width:6px;height:6px;border-radius:50%;background:var(--teal-lt);box-shadow:0 0 7px rgba(45,212,191,0.6);flex-shrink:0;animation:blink 1s ease-in-out infinite;}

    /* Suggested query buttons */
    .how-card{background:rgba(245,158,11,0.04);border:1px solid rgba(245,158,11,0.13);border-radius:13px;padding:13px 14px;}
    .how-num{width:21px;height:21px;border-radius:50%;background:rgba(245,158,11,0.13);border:1px solid rgba(245,158,11,0.28);display:inline-flex;align-items:center;justify-content:center;font-size:0.58rem;font-weight:900;color:#f59e0b;flex-shrink:0;}

    /* Model stack rows */
    .model-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);}
    .model-row:last-child{border-bottom:none;}

    /* Warn bar */
    .warn-bar{
        display:flex;align-items:center;gap:9px;padding:10px 14px;border-radius:11px;
        background:rgba(245,158,11,0.055);border:1px solid rgba(245,158,11,0.2);margin-bottom:9px;
    }

    /* Empty state */
    .empty-state{
        text-align:center;padding:3rem 2rem;
        background:radial-gradient(ellipse at 50% 55%,rgba(13,148,136,0.065) 0%,transparent 65%);
        border-radius:18px;border:1px solid rgba(255,255,255,0.04);
    }
    .empty-icon{font-size:3rem;margin-bottom:14px;display:inline-block;animation:pulseScale 2.8s ease-in-out infinite;filter:drop-shadow(0 0 18px rgba(13,148,136,0.3));}
    @keyframes pulseScale{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}

    /* Glow divider */
    .glow-div{height:1px;margin:1.1rem 0;background:linear-gradient(90deg,transparent 0%,rgba(13,148,136,0.38) 25%,rgba(124,58,237,0.28) 60%,rgba(245,158,11,0.18) 80%,transparent 100%);}

    /* Footer */
    .footer-bar{text-align:center;padding:0.65rem 0;font-size:0.68rem;color:#1e3a5f;font-weight:500;}

    /* Metric color strips */
    .strip-teal{height:3px;border-radius:2px;background:linear-gradient(90deg,#0d9488,#2dd4bf);box-shadow:0 0 10px rgba(13,148,136,0.4);margin-top:4px;}
    .strip-gold{height:3px;border-radius:2px;background:linear-gradient(90deg,#d97706,#f59e0b);box-shadow:0 0 10px rgba(245,158,11,0.38);margin-top:4px;}
    .strip-emerald{height:3px;border-radius:2px;background:linear-gradient(90deg,#059669,#34d399);box-shadow:0 0 10px rgba(52,211,153,0.38);margin-top:4px;}
    .strip-violet{height:3px;border-radius:2px;background:linear-gradient(90deg,#7c3aed,#a78bfa);box-shadow:0 0 10px rgba(124,58,237,0.38);margin-top:4px;}

    @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "db_ready":       False,
        "last_file":      None,
        "chat_history":   [],
        "total_queries":  0,
        "total_docs":     0,
        "conf_scores":    [],
        "prefill_query":  "",
        "active_tab":     "chat",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════
def avg_confidence() -> int:
    scores = st.session_state.conf_scores
    return int(sum(scores) / len(scores)) if scores else 0


def conf_bar_html(value: int) -> str:
    if value >= 80:
        cls, label, color = "conf-high",   "High",   "#34d399"
    elif value >= 60:
        cls, label, color = "conf-medium", "Medium", "#fbbf24"
    else:
        cls, label, color = "conf-low",    "Low",    "#fb7185"
    return (
        '<div class="conf-wrap">'
        '<div class="conf-meta">'
        '<span class="conf-label">Confidence Score</span>'
        f'<span class="conf-val" style="color:{color};">{label} · {value}%</span>'
        '</div>'
        '<div class="conf-track">'
        f'<div class="conf-fill {cls}" style="width:{value}%;"></div>'
        '</div></div>'
    )


def render_markdown_to_html(text: str) -> str:
    """Convert markdown text to safe HTML for use inside answer-box divs."""
    text = text.replace("&", "&").replace("<", "<").replace(">", ">")

    def replace_code_block(m):
        lang = m.group(1).strip() if m.group(1) else ""
        code = m.group(2)
        lang_label = (
            f'<span style="font-size:0.58rem;color:#475569;font-weight:700;">{lang}</span>'
            if lang else ""
        )
        return f"<pre>{lang_label}<code>{code}</code></pre>"

    text = re.sub(r"```(\w*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    text = re.sub(r"^####\s+(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s+(.+)$",  r"<h3>\1</h3>",  text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.+)$",   r"<h2>\1</h2>",   text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.+)$",    r"<h1>\1</h1>",    text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$",       "<hr>",                text, flags=re.MULTILINE)
    text = re.sub(r"^\*{3,}$",      "<hr>",                text, flags=re.MULTILINE)

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
# REUSABLE UI COMPONENTS
# ═══════════════════════════════════════════════════════
def panel_open(icon: str, title: str, sub: str, icon_cls: str = "p-icon-teal"):
    st.markdown(
        f'<div class="panel-head">'
        f'<div class="p-icon {icon_cls}">{icon}</div>'
        f'<div><div class="p-title">{title}</div>'
        f'<div class="p-sub">{sub}</div></div></div>'
        f'<div class="panel-body">',
        unsafe_allow_html=True,
    )


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def proc_step_html(icon: str, msg: str, color: str, dot_cls: str) -> str:
    return (
        f'<div class="proc-step">'
        f'<div class="{dot_cls}"></div>'
        f'<span style="color:{color};font-size:0.7rem;font-weight:600;">{icon} {msg}</span>'
        f'</div>'
    )


def render_thinking_indicator():
    return (
        '<div class="thinking-bar">'
        '<div class="thinking-dots"><span></span><span></span><span></span></div>'
        '<span style="font-size:0.8rem;color:#2dd4bf;font-weight:600;">Reasoning through your document…</span>'
        '</div>'
    )


def render_chat_message(chat: dict, index: int, total: int):
    """Render a single chat exchange (user + AI bubble)."""
    is_rag     = chat.get("mode") == "rag"
    bubble_cls = "rag" if is_rag else "fallback"
    badge_html = (
        '<span class="badge badge-rag">📄 From Document</span>'
        if is_rag else
        '<span class="badge badge-fallback">🌐 General Knowledge</span>'
    )
    av_style = (
        "background:linear-gradient(135deg,#059669,#0d9488);box-shadow:0 0 12px rgba(5,150,105,0.42);"
        if is_rag else
        "background:linear-gradient(135deg,#d97706,#f59e0b);box-shadow:0 0 12px rgba(245,158,11,0.42);"
    )

    # User bubble
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;align-items:flex-end;gap:9px;margin-bottom:7px;">'
        '<div style="max-width:76%;">'
        '<div class="chat-user">'
        f'<p style="margin:0;font-size:0.86rem;color:#e2e8f0;line-height:1.65;">{chat["question"]}</p>'
        '</div></div>'
        '<div style="width:32px;height:32px;border-radius:50%;'
        'background:linear-gradient(135deg,#0d9488,#0c4a6e);'
        'display:flex;align-items:center;justify-content:center;font-size:0.78rem;flex-shrink:0;'
        'box-shadow:0 0 12px rgba(13,148,136,0.42);">👤</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    rendered_answer = render_markdown_to_html(chat["answer"])

    # AI bubble
    st.markdown(
        '<div style="display:flex;justify-content:flex-start;align-items:flex-start;gap:9px;margin-bottom:5px;">'
        f'<div style="width:32px;height:32px;border-radius:50%;{av_style}'
        'display:flex;align-items:center;justify-content:center;font-size:0.78rem;flex-shrink:0;margin-top:20px;">🤖</div>'
        '<div style="max-width:84%;">'
        + badge_html +
        f'<div class="chat-ai {bubble_cls}">'
        f'<div class="answer-box">{rendered_answer}</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    if is_rag and chat.get("confidence", 0) > 0:
        st.markdown(conf_bar_html(chat["confidence"]), unsafe_allow_html=True)

    # Sources expander
    chat_docs = chat.get("docs", [])
    if is_rag and chat_docs:
        with st.expander(f"📚  Sources for this answer ({len(chat_docs)} chunks)"):
            for idx, doc in enumerate(chat_docs):
                page_label = f"Page {doc.get('page', '?')}"
                snippet    = doc.get("content", "")[:400]
                st.markdown(
                    '<div class="source-card">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                    f'<span style="font-size:0.6rem;font-weight:900;color:#2dd4bf;text-transform:uppercase;letter-spacing:0.08em;">📄 Source {idx + 1}</span>'
                    f'<span style="font-size:0.58rem;color:#475569;font-weight:700;">{page_label}</span>'
                    '</div>'
                    f'<p style="margin:0;font-size:0.75rem;color:#94a3b8;line-height:1.65;font-family:\'JetBrains Mono\',monospace;">{snippet}…</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    if index < total - 1:
        st.markdown(
            '<div style="height:1px;background:rgba(255,255,255,0.04);margin:9px 0;"></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            '<div style="display:flex;align-items:center;gap:9px;padding:0.5rem 0 1rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title" style="font-size:1.15rem;">RAG Assistant</div>'
            '<div style="font-size:0.58rem;color:#334155;margin-top:2px;">Document Intelligence</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

        # Nav buttons
        tabs = [("💬", "Chat"), ("📚", "Knowledge Base"), ("⚙️", "Settings")]
        for icon, label in tabs:
            active = st.session_state.get("active_tab", "chat") == label.lower().replace(" ", "_")
            active_cls = "active" if active else ""
            st.markdown(
                f'<div class="sidebar-nav-item {active_cls}">'
                f'<span>{icon}</span> {label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Pipeline Status</div>', unsafe_allow_html=True)

        components = [
            ("🤖", "Mistral 7B",   "Primary LLM",   "#2dd4bf"),
            ("🗄️",  "FAISS",        "Vector Store",  "#60a5fa"),
            ("🔢", "HuggingFace",  "Embeddings",    "#fcd34d"),
            ("🔗", "LangChain",    "RAG Pipeline",  "#34d399"),
        ]
        for icon, name, role, color in components:
            st.markdown(
                f'<div class="sidebar-status-item">'
                f'<div class="sidebar-status-dot"></div>'
                f'<div><div style="font-size:0.72rem;font-weight:700;color:{color};">{icon} {name}</div>'
                f'<div style="font-size:0.58rem;color:#334155;">{role}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Clear chat
        if st.session_state.chat_history:
            if st.button("🗑️  Clear Chat History", use_container_width=True, type="secondary"):
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
            '<div style="display:flex;align-items:center;padding-bottom:1.25rem;'
            'border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.25rem;">'
            '<div class="logo-box">🧠</div>'
            '<div>'
            '<div class="app-title">RAG Assistant</div>'
            '<div style="font-size:0.68rem;color:#3d5475;font-weight:500;margin-top:3px;">'
            'Document Intelligence &nbsp;·&nbsp; Mistral 7B + FAISS + LangChain'
            '</div></div></div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.markdown(
            '<div style="text-align:right;padding-top:8px;">'
            '<div class="status-pill">'
            f'<span class="status-dot" style="background:{status_color};box-shadow:0 0 8px {status_color},0 0 18px {status_color}40;"></span>'
            f'<span style="color:{status_color};">{status_label}</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
# METRICS ROW
# ═══════════════════════════════════════════════════════
def render_metrics():
    m1, m2, m3, m4 = st.columns(4)
    conf_val = f"{avg_confidence()}%" if st.session_state.conf_scores else "—"

    with m1:
        st.metric("📄  Documents", st.session_state.total_docs)
        st.markdown('<div class="strip-teal"></div>', unsafe_allow_html=True)
    with m2:
        st.metric("💬  Queries", st.session_state.total_queries)
        st.markdown('<div class="strip-gold"></div>', unsafe_allow_html=True)
    with m3:
        st.metric("📊  Avg Confidence", conf_val)
        st.markdown('<div class="strip-emerald"></div>', unsafe_allow_html=True)
    with m4:
        st.metric("⚡  Model", "Mistral 7B")
        st.markdown('<div class="strip-violet"></div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LEFT COLUMN — Upload + Suggestions + Model Stack
# ═══════════════════════════════════════════════════════
def render_left_column():
    # ── Document Upload ────────────────────────────────
    panel_open("📁", "Document Upload", "PDF files · up to 50 MB", "p-icon-teal")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.db_ready  = False
            st.session_state.last_file = uploaded_file.name

        os.makedirs("data/docs", exist_ok=True)
        file_path = os.path.join("data/docs", uploaded_file.name)
        with open(file_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())

        size_str = f"{round(uploaded_file.size / 1024, 1)} KB"
        fname    = uploaded_file.name[:32] + ("…" if len(uploaded_file.name) > 32 else "")
        st.markdown(
            '<div class="file-card">'
            '<span style="font-size:1.4rem;">📄</span>'
            '<div>'
            f'<div style="font-size:0.76rem;font-weight:700;color:#e2e8f0;">{fname}</div>'
            f'<div style="font-size:0.6rem;color:#475569;margin-top:2px;">{size_str} · PDF</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.db_ready:
            if st.button("⚡  Process Document", use_container_width=True, key="proc_btn"):
                step_ph = st.empty()
                bar     = st.progress(0)
                try:
                    steps = [
                        ("🔍", "Parsing PDF structure…",      5,  20),
                        ("✂️",  "Chunking text segments…",     30, 50),
                        ("🔢", "Generating embeddings…",      60, 85),
                        ("📦", "Indexing FAISS vector store…", 95, 100),
                    ]
                    for icon, msg, p_start, p_end in steps:
                        step_ph.markdown(
                            proc_step_html(icon, msg, "#2dd4bf", "proc-dot-active"),
                            unsafe_allow_html=True,
                        )
                        bar.progress(p_start)

                        if icon == "🔍":
                            docs_raw = load_pdf(file_path)
                        elif icon == "✂️":
                            chunks = split_documents(docs_raw)
                        elif icon == "🔢":
                            store_embeddings(chunks)
                        elif icon == "📦":
                            time.sleep(0.3)

                        bar.progress(p_end)

                    step_ph.markdown(
                        proc_step_html("✅", "Document ready!", "#34d399", "proc-dot-done"),
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.8)
                    step_ph.empty()
                    bar.empty()
                    st.session_state.db_ready    = True
                    st.session_state.total_docs += 1
                    st.rerun()

                except Exception as exc:
                    step_ph.empty()
                    bar.empty()
                    st.error(f"❌ Processing failed: {exc}")

        if st.session_state.db_ready:
            fname_disp = uploaded_file.name[:28] + ("…" if len(uploaded_file.name) > 28 else "")
            st.markdown(
                '<div class="ready-card">'
                '<div class="ready-dot"></div>'
                '<div>'
                '<div style="font-size:0.76rem;font-weight:800;color:#34d399;">Ready for Queries</div>'
                f'<div style="font-size:0.6rem;color:#6ee7b7;opacity:0.75;margin-top:2px;">"{fname_disp}" indexed &amp; loaded</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
    panel_close()

    # ── Suggested Queries ──────────────────────────────
    panel_open("⭐", "Suggested Queries", "Quick-start questions", "p-icon-gold")
    if st.session_state.db_ready:
        suggestions = [
            "What is the main topic of this document?",
            "Summarize the key findings.",
            "What methodology was used?",
            "What are the main conclusions?",
            "List the most important points.",
        ]
        for q in suggestions:
            if st.button(f"›  {q}", use_container_width=True, key=f"sq_{hash(q)}"):
                st.session_state.prefill_query = q
                st.rerun()
    else:
        st.markdown(
            '<div class="how-card">'
            '<div style="font-size:0.6rem;font-weight:900;color:#f59e0b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">💡 How It Works</div>'
            '<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:8px;"><div class="how-num">1</div>'
            '<div style="font-size:0.75rem;color:#94a3b8;line-height:1.55;">Upload your <strong style="color:#fcd34d;">PDF document</strong></div></div>'
            '<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:8px;"><div class="how-num">2</div>'
            '<div style="font-size:0.75rem;color:#94a3b8;line-height:1.55;">Click <strong style="color:#fcd34d;">Process Document</strong></div></div>'
            '<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:8px;"><div class="how-num">3</div>'
            '<div style="font-size:0.75rem;color:#94a3b8;line-height:1.55;">Ask questions in the <strong style="color:#fcd34d;">chat panel</strong></div></div>'
            '<div style="display:flex;align-items:flex-start;gap:9px;"><div class="how-num">4</div>'
            '<div style="font-size:0.75rem;color:#94a3b8;line-height:1.55;">Get <strong style="color:#fcd34d;">cited answers</strong> with confidence scores</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    panel_close()

    # ── Model Stack ────────────────────────────────────
    panel_open("🔧", "Model Stack", "AI pipeline components", "p-icon-violet")
    st.markdown(
        '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:11px;padding:10px 13px;">'
        '<div class="model-row"><span style="font-size:0.68rem;color:#475569;">LLM</span><span style="font-size:0.68rem;font-weight:700;color:#2dd4bf;">Mistral 7B Instruct</span></div>'
        '<div class="model-row"><span style="font-size:0.68rem;color:#475569;">Vector Store</span><span style="font-size:0.68rem;font-weight:700;color:#60a5fa;">FAISS</span></div>'
        '<div class="model-row"><span style="font-size:0.68rem;color:#475569;">Embeddings</span><span style="font-size:0.68rem;font-weight:700;color:#fcd34d;">HuggingFace</span></div>'
        '<div class="model-row"><span style="font-size:0.68rem;color:#475569;">Pipeline</span><span style="font-size:0.68rem;font-weight:700;color:#34d399;">LangChain RAG</span></div>'
        '<div class="model-row"><span style="font-size:0.68rem;color:#475569;">Fallback</span><span style="font-size:0.68rem;font-weight:700;color:#a78bfa;">General LLM</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    panel_close()


# ═══════════════════════════════════════════════════════
# CHAT COLUMN
# ═══════════════════════════════════════════════════════
def render_chat_column():
    msg_count = len(st.session_state.chat_history)

    # ── Chat header ────────────────────────────────────
    ch_l, ch_r = st.columns([7, 1])
    with ch_l:
        badge_html = ""
        if msg_count > 0:
            label = f"{msg_count} message{'s' if msg_count != 1 else ''}"
            badge_html = (
                f'<span style="font-size:0.63rem;padding:3px 10px;border-radius:99px;font-weight:800;'
                f'background:rgba(13,148,136,0.13);border:1px solid rgba(13,148,136,0.28);color:#2dd4bf;margin-left:8px;">'
                f'{label}</span>'
            )
        st.markdown(
            '<div style="display:flex;align-items:center;margin-bottom:0.9rem;">'
            '<div class="p-icon p-icon-teal" style="width:30px;height:30px;font-size:0.85rem;margin-right:9px;">💬</div>'
            '<span style="font-size:0.88rem;font-weight:900;color:#fff;">Conversation</span>'
            + badge_html + '</div>',
            unsafe_allow_html=True,
        )
    with ch_r:
        if st.session_state.chat_history:
            if st.button("🗑️", use_container_width=True, key="clear_btn", type="secondary"):
                st.session_state.chat_history  = []
                st.session_state.total_queries = 0
                st.session_state.conf_scores   = []
                st.session_state.prefill_query = ""
                st.rerun()

    # ── Query input ────────────────────────────────────
    prefill = st.session_state.get("prefill_query", "")
    st.markdown('<div class="query-wrap">', unsafe_allow_html=True)
    q_l, q_r = st.columns([7, 1])
    with q_l:
        query_input = st.text_input(
            "",
            value=prefill,
            placeholder="Ask anything about your document…" if st.session_state.db_ready else "Process a document first…",
            disabled=not st.session_state.db_ready,
            label_visibility="collapsed",
            key="query_input_field",
        )
    with q_r:
        send_clicked = st.button(
            "Send ➤" if st.session_state.db_ready else "🔒",
            use_container_width=True,
            disabled=not st.session_state.db_ready,
            key="send_btn",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if prefill:
        st.session_state.prefill_query = ""

    st.markdown(
        '<div style="font-size:0.6rem;color:#1e3a5f;text-align:center;margin-top:4px;">'
        'Press <kbd style="background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:4px;font-size:0.58rem;">Enter</kbd>'
        ' or click Send &nbsp;·&nbsp; Powered by Mistral 7B RAG'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

    # ── Warn bar ───────────────────────────────────────
    if not st.session_state.db_ready:
        st.markdown(
            '<div class="warn-bar">'
            '<span>⚠️</span>'
            '<span style="font-size:0.8rem;color:#fbbf24;font-weight:500;">'
            'Upload and process a PDF document to unlock the chat interface.'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ── Handle query ───────────────────────────────────
    query_to_run = query_input.strip() if (send_clicked and query_input) else ""
    if query_to_run:
        if not st.session_state.db_ready:
            st.warning("⚠️ Please upload and process a document first.")
        else:
            thinking_ph = st.empty()
            thinking_ph.markdown(render_thinking_indicator(), unsafe_allow_html=True)

            answer, docs, mode, confidence = "", [], "fallback", 0
            try:
                vectorstore = get_vectorstore()
                llm         = get_mistral_llm()
                rag_chain   = create_rag_chain(llm, vectorstore)
                answer, docs, results = rag_chain(query_to_run, st.session_state.chat_history)
                confidence  = calculate_confidence(results)

                if "NOT_FOUND" in answer:
                    fallback_llm = get_fallback_llm()
                    answer       = fallback_llm.invoke(query_to_run).content
                    docs         = []
                    confidence   = 0
                    mode         = "fallback"
                else:
                    mode = "rag"

            except Exception as exc:
                thinking_ph.empty()
                st.error(f"❌ An error occurred: {exc}")
                answer = ""

            thinking_ph.empty()

            if answer:
                st.session_state.total_queries += 1
                if mode == "rag" and confidence > 0:
                    st.session_state.conf_scores.append(confidence)

                docs_store = [
                    {
                        "content": d.page_content,
                        "page":    d.metadata.get("page", "?") if hasattr(d, "metadata") else "?",
                    }
                    for d in docs
                ]
                st.session_state.chat_history.append({
                    "question":   query_to_run,
                    "answer":     answer,
                    "mode":       mode,
                    "confidence": confidence,
                    "docs":       docs_store,
                })
                st.rerun()

    # ── Chat history / empty state ─────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">💬</div>'
            '<div style="font-size:1.05rem;font-weight:900;color:#fff;margin-bottom:7px;">Ready to Assist</div>'
            '<div style="font-size:0.8rem;color:#3d5475;max-width:380px;margin:0 auto;line-height:1.7;">'
            'Upload a PDF and start asking questions. The AI answers from your document '
            'with cited sources and real-time confidence scores.'
            '</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-top:1.2rem;">'
            '<span style="font-size:0.65rem;padding:4px 12px;border-radius:99px;font-weight:700;background:rgba(13,148,136,0.12);border:1px solid rgba(13,148,136,0.28);color:#2dd4bf;">🔒 Private</span>'
            '<span style="font-size:0.65rem;padding:4px 12px;border-radius:99px;font-weight:700;background:rgba(29,78,216,0.12);border:1px solid rgba(29,78,216,0.28);color:#60a5fa;">⚡ Fast RAG</span>'
            '<span style="font-size:0.65rem;padding:4px 12px;border-radius:99px;font-weight:700;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.28);color:#fcd34d;">📊 Confidence</span>'
            '<span style="font-size:0.65rem;padding:4px 12px;border-radius:99px;font-weight:700;background:rgba(5,150,105,0.12);border:1px solid rgba(5,150,105,0.28);color:#34d399;">📚 Sources</span>'
            '<span style="font-size:0.65rem;padding:4px 12px;border-radius:99px;font-weight:700;background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.28);color:#a78bfa;">🧠 Mistral 7B</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-scroll-wrap">', unsafe_allow_html=True)
        total = len(st.session_state.chat_history)
        for i, chat in enumerate(st.session_state.chat_history):
            render_chat_message(chat, i, total)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════
def render_footer():
    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="footer-bar">'
        'Powered by '
        '<span style="color:#0d9488;font-weight:800;">Mistral 7B</span> · '
        '<span style="color:#1d4ed8;font-weight:800;">FAISS</span> · '
        '<span style="color:#059669;font-weight:800;">LangChain</span>'
        '&nbsp;|&nbsp; RAG Pipeline · Production Build · v4.0'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════
# MAIN ENTRYPOINT
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


if __name__ == "__main__" or True:
    main()