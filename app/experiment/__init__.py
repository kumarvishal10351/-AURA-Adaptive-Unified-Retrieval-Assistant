"""
experiment package
──────────────────
MLOps lifecycle, tracking, and evaluation suite for AURA.
"""

try:
    from app.experiment.manager import ExperimentManager
    from app.experiment.tracker import ExperimentTracker
    from app.experiment.evaluator import AnswerEvaluator
except ImportError:
    from experiment.manager import ExperimentManager
    from experiment.tracker import ExperimentTracker
    from experiment.evaluator import AnswerEvaluator

__all__ = ["ExperimentManager", "ExperimentTracker", "AnswerEvaluator"]
