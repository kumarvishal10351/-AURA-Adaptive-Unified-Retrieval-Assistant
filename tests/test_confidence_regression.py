"""
Regression tests for confidence scoring.

Critical invariant: confidence must ALWAYS be computed from cosine similarity
scores (bounded [0,1]), NEVER from CrossEncoder logits (unbounded reals).

These tests protect against the most important AURA correctness bug documented
in the architecture guide.
"""

import pytest
from langchain_core.documents import Document
from app.utils.confidence import calculate_confidence, confidence_level
from app.config.settings import COSINE_FULL_CONFIDENCE, CONFIDENCE_FLOOR


class TestConfidenceFromCosineOnly:
    """Regression: confidence must use cosine scores, not CE logits."""

    def test_cosine_score_produces_reasonable_confidence(self):
        """Cosine 0.55 should produce ~75% confidence, not 55%."""
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.55)])
        # Formula: 20 + (0.55/0.80) * 80 = 20 + 55 = 75
        assert conf == 75

    def test_crossencoder_logit_would_produce_absurd_confidence(self):
        """
        If a CE logit like 4.8 were accidentally passed, it would produce
        a clamped 100% for everything. This test documents why the isolation
        matters — CE logits are unbounded.
        """
        doc = Document(page_content="test")
        # CE logit of 4.8 — if mistakenly passed:
        # clamped to 1.0 → 20 + (1.0/0.80) * 80 = 120 → clamped to 100
        conf = calculate_confidence([(doc, 4.8)])
        assert conf == 100  # Would always be 100 — useless for discrimination

        # CE logit of -1.2 — if mistakenly passed:
        # clamped to 0.0 → 20
        conf_neg = calculate_confidence([(doc, -1.2)])
        assert conf_neg == int(CONFIDENCE_FLOOR)  # Would always be floor — also useless

    def test_cosine_score_discriminates_quality(self):
        """Cosine scores produce meaningful differentiation between match qualities."""
        doc = Document(page_content="test")
        
        low = calculate_confidence([(doc, 0.25)])
        medium = calculate_confidence([(doc, 0.50)])
        high = calculate_confidence([(doc, 0.75)])
        
        assert low < medium < high
        assert confidence_level(low) == "Low"
        assert confidence_level(medium) == "Medium"
        assert confidence_level(high) == "High"


class TestCosineThresholdBoundaryConfidence:
    """Tests around the 0.20 cosine threshold and its confidence mapping."""

    def test_at_threshold_019(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.19)])
        # 20 + (0.19/0.80) * 80 = 20 + 19 = 39
        assert conf == 39
        assert confidence_level(conf) == "Low"

    def test_at_threshold_020(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.20)])
        # 20 + (0.20/0.80) * 80 = 20 + 20 = 40
        assert conf == 40
        assert confidence_level(conf) == "Low"

    def test_at_threshold_021(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.21)])
        # 20 + (0.21/0.80) * 80 = 20 + 21 = 41
        assert conf == 41
        assert confidence_level(conf) == "Low"


class TestConfidenceEdgeCases:
    """Edge cases for confidence scoring."""

    def test_perfect_cosine_score(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 1.0)])
        assert conf == 100

    def test_zero_cosine_score(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.0)])
        assert conf == int(CONFIDENCE_FLOOR)

    def test_negative_cosine_clamped(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, -0.5)])
        assert conf == int(CONFIDENCE_FLOOR)

    def test_above_one_clamped(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 1.5)])
        assert conf == 100

    def test_multiple_results_uses_first(self):
        """Confidence uses the top result's score (index 0)."""
        docs = [Document(page_content=f"doc{i}") for i in range(3)]
        results = [(docs[0], 0.60), (docs[1], 0.80), (docs[2], 0.40)]
        conf = calculate_confidence(results)
        # Uses first result's score (0.60), not the highest (0.80)
        assert conf == 80  # 20 + (0.60/0.80) * 80

    def test_none_input(self):
        assert calculate_confidence(None) == 0

    def test_empty_input(self):
        assert calculate_confidence([]) == 0

    def test_output_is_integer(self):
        doc = Document(page_content="test")
        conf = calculate_confidence([(doc, 0.37)])
        assert isinstance(conf, int)

    def test_output_in_valid_range(self):
        """Confidence must always be in [0, 100]."""
        doc = Document(page_content="test")
        for score in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            conf = calculate_confidence([(doc, score)])
            assert 0 <= conf <= 100, f"score={score} produced conf={conf}"
