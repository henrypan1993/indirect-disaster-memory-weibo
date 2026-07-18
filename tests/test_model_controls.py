"""Tests for confirmatory H1 / H2 / D1 model specifications."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

common = importlib.import_module("common")


def test_main_specs_are_h1_h2_m1_d1_m1_only():
    specs = common.main_model_specs()
    ids = {s.model_id for s in specs}
    assert ids == {
        "h1_engagement_indirect",
        "h2_indirect_period",
        "d1_indirect_peripheral",
    }
    assert {s.default_spec_id for s in specs} == {
        "main",
        "h2_m1_period",
        "d1_m1_total_association",
    }
    for s in specs:
        assert "entropy" not in s.model_id
        assert s.y_col != "entropy_norm"


def test_h2_m1_has_topic_fe_no_composition():
    spec = common.spec_by_spec_id("h2_m1_period")
    formula = common.build_formula(spec)
    assert "C(topic_id" in formula
    assert "Treatment(reference='0')" in formula
    assert "label_narrative_clean" not in formula
    assert "label_emotion_clean" not in formula
    assert "relative_window_day" not in formula
    assert "verified" not in formula


def test_h2_m2_has_composition_controls_and_fixed_refs():
    spec = common.spec_by_spec_id("h2_m2_composition")
    formula = common.build_formula(spec)
    assert "relative_window_day" in formula
    assert "Trauma-Help-Loss" in formula
    assert "Neutral-Informational" in formula
    assert "C(topic_id" in formula


def test_d1_m1_excludes_verified_and_log_followers():
    spec = common.spec_by_spec_id("d1_m1_total_association")
    formula = common.build_formula(spec)
    assert "peripheral" in formula
    assert "t2" in formula
    assert "C(topic_id" in formula
    assert "verified" not in formula
    assert "log_followers" not in formula


def test_d1_m2_includes_verified_and_log_followers():
    spec = common.spec_by_spec_id("d1_m2_conditional_threshold")
    formula = common.build_formula(spec)
    assert "verified" in formula
    assert "log_followers" in formula
    assert "C(topic_id" in formula


def test_h1_peripheral_controls_exclude_verified():
    spec = common.spec_by_spec_id("main")
    formula = common.build_formula(spec)
    assert "verified" not in formula
    assert "indirect_clean" in formula
    assert "C(topic_id" in formula


def test_legacy_alias_resolution():
    assert common.spec_by_model_id("h2b_indirect_reactivation").model_id == ("h2_indirect_period")
    assert common.spec_by_model_id("e1b_indirect_peripheral").model_id == ("d1_indirect_peripheral")


def test_relative_window_day():
    dates = pd.Series(["2021-07-18", "2021-07-20", "2025-08-04", "2025-08-10"])
    t2 = pd.Series([0, 0, 1, 1])
    out = common.relative_window_day_series(dates, t2)
    assert list(out) == [0.0, 2.0, 0.0, 6.0]


def test_zero_variance_cols_raise():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0, 1, 0],
            "verified": [0, 0, 0],
            "account_id": ["a", "b", "c"],
        }
    )
    formula = "y ~ x + verified"
    diag = common.diagnose_model_frame(
        df,
        formula,
        fitter="ols",
        y_col="y",
        x_cols=["x"],
        control_cols=("verified",),
    )
    assert diag["zero_variance_cols"] == ["verified"]
    with pytest.raises(common.ZeroVarianceDiagnosticError):
        if diag["zero_variance_cols"]:
            raise common.ZeroVarianceDiagnosticError(diag["zero_variance_cols"])


def test_glm_cluster_no_fallback():
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "x": [0, 1, 0, 1],
            "account_id": ["a", "a", "b", "b"],
        }
    )
    formula = "y ~ x"
    with patch.object(common.sm.GLM, "from_formula") as mock_from:
        mock_model = MagicMock()
        mock_from.return_value = mock_model
        mock_model.fit.side_effect = ValueError("cluster failed")
        with pytest.raises(common.ClusterCovarianceError):
            common.fit_glm_binomial_cluster(df, formula)


def test_appendix_without_t2_preserves_controls():
    for spec in common.appendix_without_t2_specs():
        assert "t2" not in spec.x_cols
