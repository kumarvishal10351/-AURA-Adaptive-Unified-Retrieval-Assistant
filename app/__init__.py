import os
import sys

# Ensure app directory is always on sys.path regardless of execution context
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
