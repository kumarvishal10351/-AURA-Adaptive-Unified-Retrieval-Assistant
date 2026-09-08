"""
app/main.py
───────────
Viora — Grounded Research Workspace
Serves the exact Stitch "Archival Research Workspace" design system,
integrating with the RAG pipeline and FastAPI service.
"""

import os
import sys
import threading
import time
from pathlib import Path

# Ensure project root and app folder are in sys.path for reliable imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st
import streamlit.components.v1 as components
import uvicorn

try:
    from app.api import app as fastapi_app
except ImportError:
    from api import app as fastapi_app

# Set page configuration
st.set_page_config(
    page_title="Viora — Grounded Research Workspace",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all Streamlit chrome & make canvas full-bleed
st.markdown("""
<style>
/* Remove all Streamlit default headers, bars, padding, and footers */
#MainMenu, footer, header, header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], .stDeployButton,
[data-testid="stHeaderActionElements"], [data-testid="stSidebar"] {
    display: none !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
.main, [data-testid="stMainBlockContainer"], .block-container {
    background-color: #faf9fb !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

iframe {
    width: 100vw !important;
    min-height: 100vh !important;
    height: 100vh !important;
    border: none !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)


def _start_api_server():
    """Start FastAPI backend on port 8502 if not already listening."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = sock.connect_ex(("127.0.0.1", 8502)) == 0
    sock.close()

    if not is_open:
        config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=8502,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(0.5)


# Start API server in background daemon
_start_api_server()

# Load the exact Stitch HTML application
static_html_path = Path(__file__).parent / "static" / "index.html"
if static_html_path.exists():
    with open(static_html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
else:
    html_code = "<h1>Viora — Grounded Research Workspace</h1>"

# Render the application full-screen
components.html(html_code, height=1300, scrolling=True)