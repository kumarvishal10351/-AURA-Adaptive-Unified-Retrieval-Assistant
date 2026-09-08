"""
Experiment Manager

Central entry point for the complete experiment lifecycle.

Responsibilities
----------------
1. Start experiment
2. Finish experiment
3. Log parameters
4. Log metrics
5. Log artifacts
6. Log tags

The manager hides the underlying tracking implementation
(MLflow today, another backend tomorrow).
"""

from __future__ import annotations

try:
    from app.utils import mlflow_logger
except ImportError:
    from utils import mlflow_logger


class ExperimentManager:
    """
    Central controller for experiment tracking.
    """

    def __init__(self, experiment_name: str = "AURA-RAG"):
        self.experiment_name = experiment_name
        self.is_active = False

    def start(self):
        if self.is_active:
            return

        mlflow_logger.start_experiment(self.experiment_name)
        self.is_active = True

    def finish(self):
        if not self.is_active:
            return

        mlflow_logger.end_run()
        self.is_active = False

    def log_param(self, key, value):
        if self.is_active:
            mlflow_logger.log_param(key, value)

    def log_metric(self, key, value):
        if self.is_active:
            mlflow_logger.log_metric(key, value)

    def active(self) -> bool:
        return self.is_active
