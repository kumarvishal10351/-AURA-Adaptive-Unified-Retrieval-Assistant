"""
app/main.py
───────────
Viora — Grounded Research Workspace
Main application entrypoint running the FastAPI server and serving the React UI.
"""

import os
import sys
from pathlib import Path
import uvicorn

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from app.api import app

def run():
    port = int(os.environ.get("PORT", 8000))
    print(f"--> [Render Startup] Launching Viora on 0.0.0.0:{port}...")
    sys.stdout.flush()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run()