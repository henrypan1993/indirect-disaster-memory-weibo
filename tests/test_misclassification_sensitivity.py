"""Algebra checks for the misclassification correction used in Appendix B."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

sens = importlib.import_module("14_misclassification_sensitivity")


def test_rogan_gladen_inverts_the_observation_equation() -> None:
    prevalence = 0.30
    sensitivity = 0.50
    false_positive_rate = 0.04
    observed = false_positive_rate + (sensitivity - false_positive_rate) * prevalence
    corrected = sens.rogan_gladen(observed, sensitivity, false_positive_rate)
    assert corrected == pytest.approx(prevalence)


def test_rogan_gladen_rejects_nonpositive_denominator() -> None:
    assert math.isnan(sens.rogan_gladen(0.2, 0.1, 0.2))


def test_crude_or_matches_odds_ratio_of_two_proportions() -> None:
    expected = (0.294 / 0.706) / (0.140 / 0.860)
    assert sens.crude_or(0.140, 0.294) == pytest.approx(expected)


def test_conditional_probabilities_stay_inside_the_unit_interval() -> None:
    sensitivity = 0.5
    false_positive_rate = 0.04
    prevalence = 0.22
    observed = false_positive_rate + (sensitivity - false_positive_rate) * prevalence
    ppv, p_false_neg = sens.conditional_probabilities(
        sensitivity, false_positive_rate, prevalence, observed
    )
    assert 0 <= ppv <= 1
    assert 0 <= p_false_neg <= 1
    assert ppv > p_false_neg


def test_synthetic_confusion_matrix_rates() -> None:
    """Invented pairs: sensitivity 3/6 and false-positive rate 2/10 in T1."""
    rows: list[dict[str, str]] = []
    rows += [{"period": "T1", "label_expression": "Direct", "llm_label_expression": "Direct"}] * 8
    rows += [
        {"period": "T1", "label_expression": "Direct", "llm_label_expression": "Indirect-Mixed"}
    ] * 2
    rows += [
        {
            "period": "T1",
            "label_expression": "Indirect-Mixed",
            "llm_label_expression": "Indirect-Mixed",
        }
    ] * 3
    rows += [
        {"period": "T1", "label_expression": "Indirect-Mixed", "llm_label_expression": "Direct"}
    ] * 3
    rows += [{"period": "T1", "label_expression": "Unclear", "llm_label_expression": "Direct"}]
    frame = pd.DataFrame(rows)
    pairs = sens._binary_pairs(frame)
    rates = sens.period_rates(pairs, "T1")
    assert rates["n_gold_indirect"] == 6
    assert rates["n_gold_direct"] == 10
    assert rates["sensitivity"] == pytest.approx(3 / 6)
    assert rates["false_positive_rate"] == pytest.approx(2 / 10)
    corrected = sens.rogan_gladen(0.25, rates["sensitivity"], rates["false_positive_rate"])
    assert corrected == pytest.approx((0.25 - 0.2) / (0.5 - 0.2))
