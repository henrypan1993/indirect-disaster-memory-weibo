
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

common = importlib.import_module("common")


def test_peripheral_specs_exclude_verified_in_formula():
    for mid in (
        "h1_engagement_indirect",
        "h2a_entropy_reactivation",
        "h2b_indirect_reactivation",
        "e2_entropy_increment",
    ):
        spec = common.spec_by_model_id(mid)
        formula = common.build_formula(spec)
        assert "verified" not in formula
        assert spec.control_cols == common.PERIPHERAL_SAMPLE_CONTROLS


def test_full_sample_specs_include_verified():
    for mid in ("e1a_entropy_peripheral", "e1b_indirect_peripheral"):
        spec = common.spec_by_model_id(mid)
        formula = common.build_formula(spec)
        assert "verified" in formula
        assert spec.control_cols == common.FULL_SAMPLE_CONTROLS


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


def test_appendix_specs_preserve_control_cols():
    main = {s.model_id: s.control_cols for s in common.main_model_specs()}
    for spec in common.appendix_without_t2_specs():
        assert spec.control_cols == main[spec.model_id]
