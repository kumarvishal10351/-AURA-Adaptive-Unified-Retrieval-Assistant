"""
utils/mlflow_logger.py
──────────────────────
Thin wrapper around MLflow for experiment tracking.

Provides convenience functions so callers don't need to manage
run lifecycle or handle the case where MLflow is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_active = False

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def start_experiment(experiment_name: str = "rag-assistant") -> None:
    """
    Start (or resume) an MLflow run under the given experiment.

    Safe to call multiple times — only the first call in a session
    actually creates a new run.
    """
    global _active
    if not _HAS_MLFLOW:
        logger.debug("MLflow not installed — skipping experiment tracking.")
        return

    try:
        mlflow.set_experiment(experiment_name)
        if not mlflow.active_run():
            mlflow.start_run()
        _active = True
    except Exception as exc:
        logger.warning("Could not start MLflow run: %s", exc)
        _active = False


def log_param(key: str, value) -> None:
    """Log a single parameter to the active MLflow run (no-op if inactive)."""
    if not _active:
        return
    try:
        mlflow.log_param(key, value)
    except Exception as exc:
        logger.debug("MLflow log_param(%s) failed: %s", key, exc)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a single metric to the active MLflow run (no-op if inactive)."""
    if not _active:
        return
    try:
        mlflow.log_metric(key, value, step=step)
    except Exception as exc:
        logger.debug("MLflow log_metric(%s) failed: %s", key, exc)


def end_run() -> None:
    """End the current MLflow run, if one is active."""
    global _active
    if not _active:
        return
    try:
        mlflow.end_run()
    except Exception as exc:
        logger.debug("MLflow end_run failed: %s", exc)
    finally:
        _active = False
