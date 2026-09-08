"""
experiment/artifacts.py
───────────────────────
Serialization and artifact management for evaluation results and diagnostics.
"""

from __future__ import annotations
import json
import os
from typing import Any


def save_diagnostic_report(report_data: dict[str, Any], output_path: str) -> str:
    """Save an evaluation or diagnostic report to disk as formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    return output_path


def load_diagnostic_report(input_path: str) -> dict[str, Any]:
    """Load an existing diagnostic report from disk."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Report file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
