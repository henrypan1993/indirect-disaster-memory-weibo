"""Tests for residual deviance case selection (script 10)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "export_deviance",
    SCRIPTS / "10_export_engagement_deviance_cases.py",
)
assert _spec is not None and _spec.loader is not None
dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev)


def test_expectation_main_formula_excludes_composition_terms():
    spec = common.spec_by_spec_id("expectation_main")
    formula = common.build_formula(spec)
    assert "indirect_clean" in formula
    assert "t2" in formula
    assert "C(topic_id" in formula
    assert "relative_window_day" not in formula
    assert "label_narrative_clean" not in formula
    assert "label_emotion_clean" not in formula
    assert "verified" not in formula


def test_expectation_robust_adds_relative_window_day():
    spec = common.spec_by_spec_id("expectation_with_relative_day")
    formula = common.build_formula(spec)
    assert "relative_window_day" in formula
    assert "label_narrative_clean" not in formula
    assert "label_emotion_clean" not in formula


def _synthetic_core(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        indirect = 1 if i % 2 == 0 else 0
        followers = 100 + i
        hour = i % 24
        tags = i % 3
        length = 20 + i
        t2 = 1 if i >= n // 2 else 0
        topic = str(i % 3)
        # Engagement depends on structure + expression + noise
        log_eng = (
            0.1 * np.log1p(followers)
            + 0.2 * indirect
            - 0.1 * t2
            + 0.01 * length
            + float(rng.normal(0, 0.3))
        )
        # Inject one viral Indirect-Mixed and one needs_review
        likes = int(np.expm1(log_eng))
        if i == 2:
            likes = 10_000_000
            log_eng = float(np.log1p(likes))
        needs = i == 4
        rows.append(
            {
                "mid": str(i),
                "account_id": f"a{i % 8}",  # shared accounts for cap test
                "sample_h1_eligible": True,
                "include_main": 1,
                "peripheral": 1,
                "disaster_impact_related": 1,
                "indirect_clean": float(indirect),
                "label_expression_clean": "Indirect-Mixed" if indirect else "Direct",
                "label_narrative_clean": "Trauma-Help-Loss",
                "label_emotion_clean": "Neutral-Informational",
                "t2": t2,
                "relative_window_day": float(i % 10),
                "log_followers": float(np.log1p(followers)),
                "followers_count": followers,
                "post_hour_cst": hour,
                "hashtag_count": tags,
                "text_length": length,
                "topic_id": topic,
                "log_engagement": log_eng,
                "likes_count": likes,
                "comments_count": 0,
                "reposts_count": 0,
                "high_clarity": True,
                "needs_manual_review": needs,
                "verified": 0,
            }
        )
    return pd.DataFrame(rows)


def test_residuals_equal_y_minus_yhat():
    df = _synthetic_core()
    spec = common.spec_by_spec_id("expectation_main")
    frame, _fit, _summary = dev.fit_expectation_with_residuals(df, spec)
    err = frame["log_engagement"] - frame["predicted_log_engagement"]
    assert np.allclose(err.to_numpy(), frame["residual_engagement"].to_numpy(), atol=1e-8)


def test_screening_flags_viral_and_review():
    df = _synthetic_core()
    spec = common.spec_by_spec_id("expectation_main")
    frame, _, _ = dev.fit_expectation_with_residuals(df, spec)
    screened = dev.apply_preselect_screens(frame, viral_percentile=99.5)
    reasons = set(screened.loc[~screened["screen_pass"], "screen_reason"])
    assert "screen_viral_event" in reasons
    assert "screen_needs_review" in reasons


def test_select_deviance_indirect_mixed_and_account_cap():
    df = _synthetic_core(n=40)
    spec = common.spec_by_spec_id("expectation_main")
    frame, _, _ = dev.fit_expectation_with_residuals(df, spec)
    screened = dev.apply_preselect_screens(frame, viral_percentile=99.5)
    pos, neg = dev.select_deviance_cases(screened, k=5, max_per_account=1)
    assert len(pos) <= 5
    assert len(neg) <= 5
    assert (pos["indirect_clean"] == 1).all()
    assert (neg["indirect_clean"] == 1).all()
    assert pos["account_id"].nunique() == len(pos)
    assert neg["account_id"].nunique() == len(neg)
    # Viral mid=2 should not appear
    assert "2" not in set(pos["mid"].astype(str)) | set(neg["mid"].astype(str))
    # Plan A: negative side excludes zero engagement
    assert (pd.to_numeric(neg["raw_engagement"], errors="coerce") > 0).all()


def test_negative_deviance_excludes_zero_engagement():
    """Lowest residuals that are zero-engagement must not enter negative list."""
    df = _synthetic_core(n=40)
    spec = common.spec_by_spec_id("expectation_main")
    frame, _, _ = dev.fit_expectation_with_residuals(df, spec)
    # Force several Indirect-Mixed rows to zero engagement with low residuals
    im = frame["indirect_clean"] == 1
    idxs = frame.index[im][:6]
    frame.loc[idxs, "likes_count"] = 0
    frame.loc[idxs, "comments_count"] = 0
    frame.loc[idxs, "reposts_count"] = 0
    frame.loc[idxs, "raw_engagement"] = 0
    frame.loc[idxs, "log_engagement"] = 0.0
    frame.loc[idxs, "residual_engagement"] = -10.0  # would dominate without Plan A
    screened = dev.apply_preselect_screens(frame, viral_percentile=99.5)
    _pos, neg = dev.select_deviance_cases(screened, k=5, max_per_account=1)
    assert len(neg) > 0
    assert (pd.to_numeric(neg["raw_engagement"], errors="coerce") > 0).all()
    assert set(neg["mid"].astype(str)).isdisjoint(set(frame.loc[idxs, "mid"].astype(str)))


def test_rank_stability_metrics_compute():
    df = _synthetic_core(n=40)
    main_spec = common.spec_by_spec_id("expectation_main")
    rob_spec = common.spec_by_spec_id("expectation_with_relative_day")
    main_f, _, _ = dev.fit_expectation_with_residuals(df, main_spec)
    rob_f, _, _ = dev.fit_expectation_with_residuals(df, rob_spec)
    stab_df, summary = dev.residual_rank_stability(
        main_f,
        rob_f,
        k=5,
        max_per_account=1,
        viral_percentile=99.5,
    )
    assert len(stab_df) == 2
    assert "spearman_rho" in summary
    assert np.isfinite(summary["spearman_rho"])
