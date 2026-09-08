"""
utils/mlflow_logger.py
──────────────────────
Thin wrapper around MLflow for experiment tracking with full graceful degradation.

Ensures that if MLflow is not installed, the SQLite database is locked,
or network errors occur, the core application continues working seamlessly.
"""

from __future__ import annotations

import logging
from config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger(__name__)

_active = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def is_available() -> bool:
    """Return whether MLflow is installed and operational."""
    return _HAS_MLFLOW


def start_experiment(experiment_name: str | None = None) -> None:
    """
    Start (or resume) an MLflow run under the given experiment.
    Safe to call multiple times without creating duplicate runs.
    """
    global _active
    if not _HAS_MLFLOW:
        logger.debug("MLflow not installed — skipping experiment tracking.")
        return

    exp_name = experiment_name or MLFLOW_EXPERIMENT_NAME

    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(exp_name)
        if not mlflow.active_run():
            mlflow.start_run()
        _active = True
    except Exception as exc:
        logger.warning("Could not initialize MLflow run (%s) — continuing gracefully.", exc)
        _active = False


def log_param(key: str, value) -> None:
    """Log a single parameter to the active MLflow run (no-op if inactive)."""
    if not _active or not _HAS_MLFLOW:
        return
    try:
        mlflow.log_param(key, value)
    except Exception as exc:
        logger.debug("MLflow log_param(%s) skipped: %s", key, exc)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a single metric to the active MLflow run (no-op if inactive)."""
    if not _active or not _HAS_MLFLOW:
        return
    try:
        mlflow.log_metric(key, float(value), step=step)
    except Exception as exc:
        logger.debug("MLflow log_metric(%s) skipped: %s", key, exc)


def end_run() -> None:
    """End the current MLflow run safely, if one is active."""
    global _active
    if not _active or not _HAS_MLFLOW:
        _active = False
        return
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception as exc:
        logger.debug("MLflow end_run() skipped: %s", exc)
    finally:
        _active = False
