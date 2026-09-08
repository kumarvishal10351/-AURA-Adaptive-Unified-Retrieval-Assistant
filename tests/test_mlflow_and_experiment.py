"""
Tests for MLflow logging, graceful degradation, and the experiment tracking framework.
"""

from app.experiment.manager import ExperimentManager
from app.experiment.tracker import ExperimentTracker
from app.experiment.artifacts import save_diagnostic_report, load_diagnostic_report
from app.utils import mlflow_logger


def test_experiment_manager_lifecycle():
    mgr = ExperimentManager(experiment_name="test-experiment")
    assert mgr.active() is False

    mgr.start()
    assert mgr.active() is True

    mgr.log_param("test_param", "value")
    mgr.log_metric("test_metric", 42.0)

    mgr.finish()
    assert mgr.active() is False


def test_experiment_tracker_timing():
    tracker = ExperimentTracker()
    tracker.start_timer("retrieval")
    elapsed = tracker.stop_timer("retrieval")
    assert elapsed >= 0.0


def test_mlflow_logger_safe_noops():
    # Calling log_param, log_metric, end_run without an active run should never raise
    mlflow_logger.end_run()
    mlflow_logger.log_param("safe_key", "safe_val")
    mlflow_logger.log_metric("safe_metric", 123.4)
    mlflow_logger.end_run()


def test_diagnostic_artifact_roundtrip(tmp_path):
    report_file = str(tmp_path / "diag.json")
    data = {"system": "AURA", "accuracy": 0.96, "grounded": True}

    save_diagnostic_report(data, report_file)
    loaded = load_diagnostic_report(report_file)

    assert loaded["system"] == "AURA"
    assert loaded["accuracy"] == 0.96
    assert loaded["grounded"] is True
