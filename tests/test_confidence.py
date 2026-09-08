"""
Unit and regression tests for confidence scoring calculation and qualitative levels.
"""

import pytest
from app.utils.confidence import calculate_confidence, confidence_level, average_confidence
from app.config.settings import COSINE_FULL_CONFIDENCE, CONFIDENCE_FLOOR


class MockDoc:
    def __init__(self, page_content="test content"):
        self.page_content = page_content
        self.metadata = {}


def test_confidence_empty_results():
    assert calculate_confidence([]) == 0
    assert calculate_confidence(None) == 0


def test_confidence_malformed_input():
    assert calculate_confidence([None]) == 0
    assert calculate_confidence([("doc", "not_a_number")]) == 0
    assert calculate_confidence(["not_a_tuple"]) == 0


def test_confidence_boundary_zero():
    doc = MockDoc()
    # At cosine 0.0, formula gives 20% floor
    conf = calculate_confidence([(doc, 0.0)])
    assert conf == int(CONFIDENCE_FLOOR)
    assert confidence_level(conf) == "Low"


def test_confidence_boundary_full():
    doc = MockDoc()
    # At cosine 0.80 (COSINE_FULL_CONFIDENCE), maps to 100%
    conf = calculate_confidence([(doc, COSINE_FULL_CONFIDENCE)])
    assert conf == 100
    assert confidence_level(conf) == "High"


def test_confidence_above_full_clamped():
    doc = MockDoc()
    # Cosine > 0.80 must never exceed 100
    conf = calculate_confidence([(doc, 0.95)])
    assert conf == 100
    conf_max = calculate_confidence([(doc, 1.0)])
    assert conf_max == 100


def test_confidence_negative_clamped():
    doc = MockDoc()
    # Negative cosine similarity clamped to 0.0
    conf = calculate_confidence([(doc, -0.5)])
    assert conf == int(CONFIDENCE_FLOOR)


def test_confidence_exact_formula():
    doc = MockDoc()
    # At cosine 0.40 (halfway to 0.80) -> 20 + 0.5 * 80 = 60%
    conf = calculate_confidence([(doc, 0.40)])
    assert conf == 60
    assert confidence_level(conf) == "Medium"


def test_confidence_levels():
    assert confidence_level(0) == "None"
    assert confidence_level(-1) == "None"
    assert confidence_level(35) == "Low"
    assert confidence_level(59) == "Low"
    assert confidence_level(60) == "Medium"
    assert confidence_level(79) == "Medium"
    assert confidence_level(80) == "High"
    assert confidence_level(100) == "High"


def test_average_confidence():
    assert average_confidence([]) == 0
    assert average_confidence([80, 90, 70]) == 80
    assert average_confidence([100, 0, 80]) == 90  # 0 is excluded as uncalibrated/not_found
