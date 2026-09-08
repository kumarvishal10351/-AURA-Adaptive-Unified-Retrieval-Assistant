"""
Root conftest.py
────────────────
Ensures the `app/` directory is on sys.path so that test imports
like `from app.utils.confidence import ...` work consistently
regardless of working directory.
"""

import os
import sys

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Ensure the app directory is on sys.path for bare imports (config.settings, etc.)
_app_dir = os.path.join(_project_root, "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
