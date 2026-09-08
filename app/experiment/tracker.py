"""
experiment/tracker.py
─────────────────────
Metrics and parameter tracker across the RAG execution lifecycle.
"""

from __future__ import annotations
import time
from typing import Any

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


class ExperimentTracker:
    """
    Records parameters, metrics, and latency timings across ingestion and retrieval.
    """

    def __init__(self, experiment_name: str = "AURA-RAG"):
        self.experiment_name = experiment_name
        self._timers: dict[str, float] = {}

    def start_timer(self, label: str) -> None:
        self._timers[label] = time.time()

    def stop_timer(self, label: str) -> float:
        start = self._timers.pop(label, None)
        if start is None:
            return 0.0
        elapsed = round(time.time() - start, 4)
        mlflow_logger.log_metric(f"{label}_seconds", elapsed)
        return elapsed

    def log_parameter(self, key: str, value: Any) -> None:
        mlflow_logger.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow_logger.log_metric(key, value, step=step)
