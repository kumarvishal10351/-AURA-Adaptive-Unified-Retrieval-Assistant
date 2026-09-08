"""
Tests for MLflow graceful degradation.

Critical requirement: MLflow failures must NEVER crash the core application.
All MLflow operations should be no-ops when MLflow is unavailable or inactive.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.utils import mlflow_logger
from app.experiment.manager import ExperimentManager
from app.experiment.tracker import ExperimentTracker


class TestMLflowGracefulDegradation:
    """Verify MLflow failures don't crash the application."""

    def test_log_param_without_active_run(self):
        """log_param should be a safe no-op when no run is active."""
        mlflow_logger._active = False
        mlflow_logger.log_param("test_key", "test_value")  # Should not raise

    def test_log_metric_without_active_run(self):
        """log_metric should be a safe no-op when no run is active."""
        mlflow_logger._active = False
        mlflow_logger.log_metric("test_metric", 42.0)  # Should not raise

    def test_end_run_without_active_run(self):
        """end_run should be safe to call without an active run."""
        mlflow_logger._active = False
        mlflow_logger.end_run()  # Should not raise

    def test_double_end_run(self):
        """Calling end_run twice should not raise."""
        mlflow_logger.end_run()
        mlflow_logger.end_run()

    def test_log_param_with_various_types(self):
        """log_param should handle various value types without crashing."""
        mlflow_logger._active = False
        mlflow_logger.log_param("str_key", "value")
        mlflow_logger.log_param("int_key", 42)
        mlflow_logger.log_param("float_key", 3.14)
        mlflow_logger.log_param("bool_key", True)
        mlflow_logger.log_param("none_key", None)

    def test_is_available_returns_bool(self):
        result = mlflow_logger.is_available()
        assert isinstance(result, bool)


class TestMLflowUnavailable:
    """Test behavior when MLflow import fails."""

    def test_logger_functions_safe_without_mlflow(self):
        """All functions should be safe no-ops when _HAS_MLFLOW is False."""
        original = mlflow_logger._HAS_MLFLOW
        try:
            mlflow_logger._HAS_MLFLOW = False
            mlflow_logger._active = False
            
            mlflow_logger.start_experiment("test")
            mlflow_logger.log_param("key", "val")
            mlflow_logger.log_metric("metric", 1.0)
            mlflow_logger.end_run()
        finally:
            mlflow_logger._HAS_MLFLOW = original


class TestExperimentManagerLifecycle:
    """Test the ExperimentManager lifecycle operations."""

    def test_manager_starts_inactive(self):
        mgr = ExperimentManager("test-exp")
        assert mgr.active() is False

    def test_manager_start_finish_cycle(self):
        mgr = ExperimentManager("test-exp")
        mgr.start()
        assert mgr.active() is True
        mgr.finish()
        assert mgr.active() is False

    def test_manager_double_start_idempotent(self):
        mgr = ExperimentManager("test-exp")
        mgr.start()
        mgr.start()  # Second start should not error
        assert mgr.active() is True
        mgr.finish()

    def test_manager_double_finish_safe(self):
        mgr = ExperimentManager("test-exp")
        mgr.finish()  # Finish without start
        mgr.finish()  # Double finish

    def test_manager_log_when_inactive(self):
        mgr = ExperimentManager("test-exp")
        # Logging when inactive should be no-op
        mgr.log_param("key", "val")
        mgr.log_metric("metric", 1.0)


class TestExperimentTracker:
    """Test the ExperimentTracker timing functionality."""

    def test_timer_roundtrip(self):
        tracker = ExperimentTracker()
        tracker.start_timer("test_op")
        elapsed = tracker.stop_timer("test_op")
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)

    def test_stop_nonexistent_timer(self):
        tracker = ExperimentTracker()
        elapsed = tracker.stop_timer("nonexistent")
        assert elapsed == 0.0

    def test_timer_not_reusable(self):
        """Stopping a timer should remove it."""
        tracker = ExperimentTracker()
        tracker.start_timer("op")
        tracker.stop_timer("op")
        elapsed = tracker.stop_timer("op")  # Second stop
        assert elapsed == 0.0
